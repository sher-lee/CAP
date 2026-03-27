"""
Capture Sequencer
==================
Orchestrates the full slide scan. Moves the motors through the
serpentine field grid, runs per-field autofocus, captures Z-stacks,
and passes frames to the focus stacker. Tracks field status and
emits progress callbacks for the UI.

Usage:
    sequencer = CaptureSequencer(config, motor, camera, focus_map, scan_region)
    sequencer.run(progress_callback=on_progress)
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import numpy as np

from cap.common.dataclasses import (
    FieldStatus, FocusMapResult, RawFrame, ScanProgress, ScanRegion,
)
from cap.common.logging_setup import get_logger
from cap.layer1_hardware.per_field_autofocus import PerFieldAutofocus, AutofocusResult

logger = get_logger("capture")


class CaptureSequencer:
    """
    Executes the serpentine raster scan within the defined scan region.
    At each field: move → autofocus → capture Z-stack → pass to stacker.
    """

    def __init__(
        self,
        config: object,
        motor_controller,
        camera_interface,
        autofocus: PerFieldAutofocus,
        focus_map: FocusMapResult,
        scan_region: ScanRegion,
    ) -> None:
        self._config = config
        self._motor = motor_controller
        self._camera = camera_interface
        self._autofocus = autofocus
        self._focus_map = focus_map
        self._region = scan_region

        if hasattr(config, "scan"):
            self._max_retries = config.scan.max_capture_retries
            self._fields_per_sec = config.scan.fields_per_second
        else:
            scan = config.get("scan", {})
            self._max_retries = scan.get("max_capture_retries", 3)
            self._fields_per_sec = scan.get("fields_per_second", 2)

        # State
        self._is_running = False
        self._is_paused = False
        self._stop_requested = False
        self._fields_completed = 0
        self._fields_total = len(scan_region.field_positions)
        self._start_time: Optional[float] = None

        # Results per field: field_pos → AutofocusResult
        self._field_results: dict[tuple[int, int], AutofocusResult] = {}
        self._field_statuses: dict[tuple[int, int], FieldStatus] = {}

        # Initialize all fields as pending
        for pos in scan_region.field_positions:
            self._field_statuses[pos] = FieldStatus.PENDING

        logger.info(
            "CaptureSequencer initialized: %d fields, max_retries=%d",
            self._fields_total, self._max_retries,
        )

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def fields_completed(self) -> int:
        return self._fields_completed

    @property
    def fields_total(self) -> int:
        return self._fields_total

    def pause(self) -> None:
        """Pause the scan after the current field completes."""
        self._is_paused = True
        logger.info("Scan pause requested")

    def resume(self) -> None:
        """Resume a paused scan."""
        self._is_paused = False
        logger.info("Scan resumed")

    def stop(self) -> None:
        """Stop the scan after the current field completes."""
        self._stop_requested = True
        logger.info("Scan stop requested")

    def run(
        self,
        progress_callback: Optional[Callable[[ScanProgress], None]] = None,
        field_complete_callback: Optional[Callable[[tuple[int, int], AutofocusResult], None]] = None,
    ) -> dict:
        """
        Execute the full scan sequence.

        Parameters
        ----------
        progress_callback : callable, optional
            Called after each field with a ScanProgress object.
        field_complete_callback : callable, optional
            Called after each field with (field_pos, AutofocusResult).
            The AutofocusResult contains all captured Z-stack frames
            for the focus stacker to process.

        Returns
        -------
        dict
            Scan summary: fields_completed, fields_failed, duration_sec,
            field_results dict.
        """
        self._is_running = True
        self._stop_requested = False
        self._is_paused = False
        self._fields_completed = 0
        self._start_time = time.monotonic()

        fields_failed = 0

        logger.info("Scan started: %d fields to capture", self._fields_total)

        for field_idx, (fx, fy) in enumerate(self._region.field_positions):
            # Check for stop/pause
            if self._stop_requested:
                logger.info("Scan stopped at field %d/%d", field_idx, self._fields_total)
                break

            while self._is_paused:
                time.sleep(0.1)
                if self._stop_requested:
                    break

            if self._stop_requested:
                break

            # Update status
            self._field_statuses[(fx, fy)] = FieldStatus.CAPTURING

            # Attempt capture with retries
            success = False
            for attempt in range(self._max_retries + 1):
                try:
                    result = self._capture_field(fx, fy)
                    self._field_results[(fx, fy)] = result
                    self._field_statuses[(fx, fy)] = FieldStatus.CAPTURED
                    success = True

                    # Notify callback with the Z-stack frames
                    if field_complete_callback:
                        field_complete_callback((fx, fy), result)

                    break

                except Exception as e:
                    if attempt < self._max_retries:
                        logger.warning(
                            "Field (%d, %d) capture failed (attempt %d/%d): %s",
                            fx, fy, attempt + 1, self._max_retries + 1, e,
                        )
                    else:
                        logger.error(
                            "Field (%d, %d) capture FAILED after %d attempts: %s",
                            fx, fy, self._max_retries + 1, e,
                        )
                        self._field_statuses[(fx, fy)] = FieldStatus.FAILED
                        fields_failed += 1

            if success:
                self._fields_completed += 1

            # Emit progress
            if progress_callback:
                elapsed = time.monotonic() - self._start_time
                remaining = self._fields_total - (field_idx + 1)

                if self._fields_completed > 0:
                    avg_time = elapsed / self._fields_completed
                    eta = remaining * avg_time
                else:
                    eta = 0.0

                progress = ScanProgress(
                    fields_completed=self._fields_completed,
                    fields_total=self._fields_total,
                    current_x=fx,
                    current_y=fy,
                    eta_seconds=eta,
                    current_field_status=f"Field ({fx}, {fy}) — {self._field_statuses[(fx, fy)].value}",
                )
                progress_callback(progress)

        # Scan complete
        duration = time.monotonic() - self._start_time
        self._is_running = False

        logger.info(
            "Scan complete: %d/%d fields captured, %d failed, %.1f sec total",
            self._fields_completed, self._fields_total, fields_failed, duration,
        )

        return {
            "fields_completed": self._fields_completed,
            "fields_failed": fields_failed,
            "fields_total": self._fields_total,
            "duration_sec": duration,
            "field_results": self._field_results,
            "field_statuses": self._field_statuses,
        }

    def _capture_field(self, field_x: int, field_y: int) -> AutofocusResult:
        """
        Capture a single field: move to position, run autofocus,
        and return the Z-stack frames.

        Parameters
        ----------
        field_x, field_y : int
            Motor coordinates of the field center.

        Returns
        -------
        AutofocusResult
            Contains all Z-stack frames, sharpness scores, and drift info.
        """
        # Move to field position
        self._motor.move_to("x", field_x)
        self._motor.move_to("y", field_y)

        # Run autofocus and capture Z-stack
        # This moves Z to multiple positions and captures a frame at each
        result = self._autofocus.find_best_z(
            self._motor,
            self._camera,
            self._focus_map,
            field_x,
            field_y,
        )

        logger.debug(
            "Field (%d, %d): Z=%.0f, drift=%.1f, %d frames captured",
            field_x, field_y, result.actual_z, result.drift, len(result.frames),
        )

        return result

    def get_field_status(self, field_pos: tuple[int, int]) -> FieldStatus:
        """Get the status of a specific field."""
        return self._field_statuses.get(field_pos, FieldStatus.PENDING)

    def get_all_statuses(self) -> dict[tuple[int, int], FieldStatus]:
        """Get status dict for all fields."""
        return dict(self._field_statuses)

    def get_progress(self) -> ScanProgress:
        """Get current scan progress snapshot."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        if self._fields_completed > 0:
            avg_time = elapsed / self._fields_completed
            remaining = self._fields_total - self._fields_completed
            eta = remaining * avg_time
        else:
            eta = 0.0

        return ScanProgress(
            fields_completed=self._fields_completed,
            fields_total=self._fields_total,
            current_x=0,
            current_y=0,
            eta_seconds=eta,
            current_field_status="running" if self._is_running else "idle",
        )

    def get_resume_point(self) -> int:
        """
        Get the index of the first incomplete field for resume capability.
        Returns the index into scan_region.field_positions.
        """
        for i, pos in enumerate(self._region.field_positions):
            status = self._field_statuses.get(pos, FieldStatus.PENDING)
            if status in (FieldStatus.PENDING, FieldStatus.FAILED):
                return i
        return self._fields_total
