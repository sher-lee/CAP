"""
Scan Worker
============
QThread that runs the full scan pipeline (preliminary focus →
capture → focus stack → disk write) off the UI thread.

Communicates with the UI via ScanSignals and FocusSignals.
The scan_control screen creates this worker, connects its signals,
and calls start().

Usage:
    worker = ScanWorker(app_context, scan_region)
    worker.signals.progress.connect(self.on_progress)
    worker.signals.scan_complete.connect(self.on_complete)
    worker.start()
    # Later:
    worker.request_pause()
    worker.request_stop()
"""

from __future__ import annotations

import time
import json
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread

from cap.common.dataclasses import ScanRegion, ScanProgress, FocusMapResult
from cap.common.logging_setup import get_logger
from cap.layer6_ui.signals import ScanSignals, FocusSignals
from cap.layer5_data import crud
from cap.layer5_data.audit import AuditLogger, EventType

if TYPE_CHECKING:
    from cap.app import AppContext

logger = get_logger("workers.scan")


class ScanWorker(QThread):
    """
    Runs the scan pipeline in a background thread.

    Lifecycle:
        1. Preliminary focus (builds focus map)
        2. ScanPipeline.run() (capture → stack → disk)
        3. Update DB with scan results
        4. Emit scan_complete or scan_failed

    The UI thread must not call pipeline methods directly —
    use request_pause() / request_stop() which are thread-safe.
    """

    def __init__(
        self,
        app_context: AppContext,
        scan_region_vertices: list[tuple[int, int]],
    ) -> None:
        super().__init__()

        self._ctx = app_context
        self._region_vertices = scan_region_vertices

        # Public signal objects — connect these in the UI
        self.scan_signals = ScanSignals()
        self.focus_signals = FocusSignals()

        # Pipeline reference (set during run, used for pause/stop)
        self._pipeline = None
        self._stop_requested = False
        self._pause_requested = False

    def run(self) -> None:
        """
        Thread entry point. Runs the full scan pipeline.
        Do NOT call this directly — use self.start().
        """
        slide_id = getattr(self._ctx, "current_slide_id", None)
        config = self._ctx.config
        audit = AuditLogger(self._ctx.db)

        try:
            # ---------------------------------------------------------------
            # Step 1: Build ScanRegion from vertices
            # ---------------------------------------------------------------
            from cap.layer1_hardware.scan_region import ScanRegionManager

            region_mgr = ScanRegionManager(config)
            scan_region = region_mgr.create_from_vertices(
                self._region_vertices,
            )

            logger.info(
                "Scan region created: %d fields, est. %.0f sec",
                scan_region.field_count,
                scan_region.estimated_scan_time_sec,
            )

            # ---------------------------------------------------------------
            # Step 2: Preliminary focus
            # ---------------------------------------------------------------
            self.focus_signals.preliminary_focus_started.emit()
            audit.log(EventType.SCAN_STARTED, details=f"Slide {slide_id}")

            focus_module = self._ctx.focus
            focus_result = focus_module.run(
                motor=self._ctx.motor,
                camera=self._ctx.camera,
                scan_region=scan_region,
            )

            self.focus_signals.preliminary_focus_complete.emit(focus_result)

            logger.info(
                "Preliminary focus complete: residual=%.4f",
                focus_result.fit_residual,
            )

            # Store focus map in DB
            if slide_id:
                crud.update_slide_scan_complete(
                    self._ctx.db,
                    slide_id=slide_id,
                    scan_duration=0,  # updated later with real duration
                    focus_map_json=json.dumps(focus_result.sample_points),
                    focus_map_grid_size=f"{focus_result.grid_size[0]}x{focus_result.grid_size[1]}",
                )

            if self._stop_requested:
                self._emit_stopped(slide_id, audit)
                return

            # ---------------------------------------------------------------
            # Step 3: Insert field records in DB
            # ---------------------------------------------------------------
            if slide_id:
                field_positions = [
                    (pos[0], pos[1]) for pos in scan_region.field_positions
                ]
                field_ids = crud.insert_fields_batch(
                    self._ctx.db,
                    slide_id,
                    field_positions,
                )
                # Store mapping for inference worker to use later
                self._ctx.field_id_map = {
                    pos: fid for pos, fid in zip(field_positions, field_ids)
                }

                crud.update_slide_status(self._ctx.db, slide_id, "scanning")

            # ---------------------------------------------------------------
            # Step 4: Run scan pipeline
            # ---------------------------------------------------------------
            from cap.layer2_acquisition.pipeline import ScanPipeline

            self._pipeline = ScanPipeline(
                config=config,
                motor_controller=self._ctx.motor,
                camera_interface=self._ctx.camera,
                autofocus=self._ctx.autofocus,
                focus_map=focus_result,
                scan_region=scan_region,
            )

            def on_progress(progress: ScanProgress):
                """Forward pipeline progress to the UI thread via signal."""
                self.scan_signals.progress.emit(progress)
                self.scan_signals.field_captured.emit(
                    progress.fields_completed, progress.fields_total,
                )

            scan_start_time = time.monotonic()

            pipeline_result = self._pipeline.run(
                slide_id=slide_id or 0,
                progress_callback=on_progress,
            )

            scan_duration = time.monotonic() - scan_start_time

            # ---------------------------------------------------------------
            # Step 5: Update DB with scan results
            # ---------------------------------------------------------------
            if slide_id:
                crud.update_slide_scan_complete(
                    self._ctx.db,
                    slide_id=slide_id,
                    scan_duration=scan_duration,
                    focus_map_json=json.dumps(focus_result.sample_points),
                    focus_map_grid_size=f"{focus_result.grid_size[0]}x{focus_result.grid_size[1]}",
                )

                # Update field statuses to 'stacked'
                for sf in pipeline_result.get("stacked_fields", []):
                    pos_key = (sf.field_x, sf.field_y)
                    field_id = self._ctx.field_id_map.get(pos_key)
                    if field_id:
                        crud.update_field_status(
                            self._ctx.db,
                            field_id,
                            "stacked",
                            focus_score=float(sf.sharpness_map.mean()),
                        )

            # Store pipeline result in context for inference worker
            self._ctx.last_pipeline_result = pipeline_result
            self._ctx.last_scan_region = scan_region

            audit.log(
                EventType.SCAN_COMPLETED,
                details=(
                    f"Slide {slide_id}: "
                    f"{pipeline_result.get('fields_completed', 0)} fields, "
                    f"{scan_duration:.1f}s"
                ),
            )

            logger.info(
                "Scan complete: slide=%s, %d fields, %.1f sec",
                slide_id,
                pipeline_result.get("fields_completed", 0),
                scan_duration,
            )

            self.scan_signals.scan_complete.emit(slide_id or 0)

        except Exception as exc:
            logger.error("Scan failed: %s", exc, exc_info=True)
            audit.log(
                EventType.SCAN_FAILED,
                details=f"Slide {slide_id}: {exc}",
            )
            self.scan_signals.scan_failed.emit(str(exc))

    # -------------------------------------------------------------------
    # Thread-safe control methods (called from UI thread)
    # -------------------------------------------------------------------

    def request_pause(self) -> None:
        """Request the scan to pause. Thread-safe."""
        self._pause_requested = True
        if self._pipeline:
            self._pipeline.pause()
        logger.info("Pause requested")

    def request_resume(self) -> None:
        """Request the scan to resume. Thread-safe."""
        self._pause_requested = False
        if self._pipeline:
            self._pipeline.resume()
        logger.info("Resume requested")

    def request_stop(self) -> None:
        """Request the scan to stop. Thread-safe."""
        self._stop_requested = True
        if self._pipeline:
            self._pipeline.stop()
        logger.info("Stop requested")

    def _emit_stopped(self, slide_id, audit: AuditLogger) -> None:
        """Handle a user-requested stop."""
        if slide_id:
            crud.update_slide_status(self._ctx.db, slide_id, "scan_complete")
        audit.log(EventType.SCAN_COMPLETED, details=f"Slide {slide_id}: stopped by user")
        self.scan_signals.scan_complete.emit(slide_id or 0)
        logger.info("Scan stopped by user request")
