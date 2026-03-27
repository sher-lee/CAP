"""
Scan Pipeline Manager
=======================
Coordinates the full scan-to-disk pipeline using producer/consumer
threads with queue-based handoff:

    Capture thread → Stack queue → Stacking thread → Disk queue → Disk I/O thread

This threading model prevents the camera from waiting on stacking
or disk writes, and prevents stacking from blocking the next capture.

Usage:
    pipeline = ScanPipeline(config, motor, camera, autofocus, focus_map, scan_region)
    result = pipeline.run(progress_callback=on_progress)
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from cap.common.dataclasses import (
    FocusMapResult, ScanProgress, ScanRegion, StackedField,
)
from cap.common.logging_setup import get_logger
from cap.layer1_hardware.per_field_autofocus import PerFieldAutofocus, AutofocusResult
from cap.layer2_acquisition.capture_sequencer import CaptureSequencer
from cap.layer2_acquisition.focus_stacker import FocusStacker

logger = get_logger("pipeline")

# Sentinel value to signal thread shutdown
_SENTINEL = object()


class ScanPipeline:
    """
    Full scan-to-disk pipeline with threaded stages.
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

        # Configuration
        if hasattr(config, "storage"):
            self._image_root = config.storage.image_root
            self._stacked_format = config.storage.stacked_format
            self._jpeg_quality = config.storage.stacked_jpeg_quality
        else:
            storage = config.get("storage", {})
            self._image_root = storage.get("image_root", "./data/slides/")
            self._stacked_format = storage.get("stacked_format", "jpeg")
            self._jpeg_quality = storage.get("stacked_jpeg_quality", 95)

        # Components
        self._sequencer = CaptureSequencer(
            config, motor_controller, camera_interface,
            autofocus, focus_map, scan_region,
        )
        self._stacker = FocusStacker(config)

        # Queues
        self._stack_queue: queue.Queue = queue.Queue(maxsize=4)
        self._disk_queue: queue.Queue = queue.Queue(maxsize=8)

        # Results
        self._stacked_fields: list[StackedField] = []
        self._lock = threading.Lock()

        logger.info("ScanPipeline initialized: %d fields", len(scan_region.field_positions))

    def run(
        self,
        slide_id: int = 0,
        progress_callback: Optional[Callable[[ScanProgress], None]] = None,
    ) -> dict:
        """
        Run the full pipeline: capture → stack → save to disk.

        Can run in two modes:
        - Threaded (default): capture, stacking, and disk I/O run in
          separate threads for maximum throughput
        - Sequential: for debugging, runs everything in the main thread

        Parameters
        ----------
        slide_id : int
            Slide ID for file path construction and metadata.
        progress_callback : callable, optional
            Called with ScanProgress after each field.

        Returns
        -------
        dict
            Pipeline results: scan summary + list of StackedField objects.
        """
        logger.info("Pipeline starting: slide_id=%d", slide_id)

        # Ensure output directories exist
        slide_dir = os.path.join(self._image_root, str(slide_id))
        os.makedirs(slide_dir, exist_ok=True)

        # Start worker threads
        stack_thread = threading.Thread(
            target=self._stacking_worker,
            args=(slide_id,),
            name="stacking-worker",
            daemon=True,
        )
        disk_thread = threading.Thread(
            target=self._disk_worker,
            args=(slide_id,),
            name="disk-worker",
            daemon=True,
        )

        stack_thread.start()
        disk_thread.start()

        # Run capture sequencer in the main thread (it controls motors)
        def on_field_complete(field_pos, autofocus_result):
            """Pass captured Z-stack to stacking queue."""
            self._stack_queue.put((field_pos, autofocus_result))

        scan_result = self._sequencer.run(
            progress_callback=progress_callback,
            field_complete_callback=on_field_complete,
        )

        # Signal stacking thread to finish
        self._stack_queue.put(_SENTINEL)
        stack_thread.join(timeout=60)

        # Signal disk thread to finish
        self._disk_queue.put(_SENTINEL)
        disk_thread.join(timeout=60)

        logger.info(
            "Pipeline complete: %d fields captured, %d stacked, %.1f sec",
            scan_result["fields_completed"],
            len(self._stacked_fields),
            scan_result["duration_sec"],
        )

        return {
            **scan_result,
            "stacked_fields": list(self._stacked_fields),
            "slide_dir": slide_dir,
        }

    def run_sequential(
        self,
        slide_id: int = 0,
        progress_callback: Optional[Callable[[ScanProgress], None]] = None,
    ) -> dict:
        """
        Run the pipeline sequentially (no threading).
        Useful for debugging and testing.
        """
        logger.info("Pipeline starting (sequential mode): slide_id=%d", slide_id)

        slide_dir = os.path.join(self._image_root, str(slide_id))
        os.makedirs(slide_dir, exist_ok=True)

        stacked_results = []

        def on_field_complete(field_pos, autofocus_result):
            fx, fy = field_pos

            # Stack immediately
            stacked = self._stacker.stack(
                frames=autofocus_result.frames,
                slide_id=slide_id,
                field_x=fx,
                field_y=fy,
            )

            # Save to disk immediately
            self._save_stacked_field(stacked, slide_id)

            with self._lock:
                self._stacked_fields.append(stacked)
                stacked_results.append(stacked)

        scan_result = self._sequencer.run(
            progress_callback=progress_callback,
            field_complete_callback=on_field_complete,
        )

        logger.info(
            "Pipeline complete (sequential): %d fields, %d stacked, %.1f sec",
            scan_result["fields_completed"],
            len(stacked_results),
            scan_result["duration_sec"],
        )

        return {
            **scan_result,
            "stacked_fields": stacked_results,
            "slide_dir": slide_dir,
        }

    # ----- Worker threads -----

    def _stacking_worker(self, slide_id: int) -> None:
        """
        Stacking worker thread. Reads (field_pos, autofocus_result) from
        the stack queue, runs focus stacking, and puts the result on
        the disk queue.
        """
        logger.debug("Stacking worker started")

        while True:
            item = self._stack_queue.get()

            if item is _SENTINEL:
                logger.debug("Stacking worker received shutdown signal")
                self._disk_queue.put(_SENTINEL)
                break

            field_pos, autofocus_result = item
            fx, fy = field_pos

            try:
                stacked = self._stacker.stack(
                    frames=autofocus_result.frames,
                    slide_id=slide_id,
                    field_x=fx,
                    field_y=fy,
                )

                with self._lock:
                    self._stacked_fields.append(stacked)

                self._disk_queue.put(stacked)

                logger.debug(
                    "Stacked field (%d, %d): %.1f ms, %d Z-depths used",
                    fx, fy, stacked.stacking_duration_ms,
                    len(stacked.z_distribution),
                )

            except Exception as e:
                logger.error("Stacking failed for field (%d, %d): %s", fx, fy, e)

            self._stack_queue.task_done()

        logger.debug("Stacking worker stopped")

    def _disk_worker(self, slide_id: int) -> None:
        """
        Disk I/O worker thread. Reads StackedField objects from the
        disk queue and saves composite images to the file system.
        """
        logger.debug("Disk I/O worker started")

        while True:
            item = self._disk_queue.get()

            if item is _SENTINEL:
                logger.debug("Disk I/O worker received shutdown signal")
                break

            try:
                self._save_stacked_field(item, slide_id)
            except Exception as e:
                logger.error(
                    "Disk write failed for field (%d, %d): %s",
                    item.field_x, item.field_y, e,
                )

            self._disk_queue.task_done()

        logger.debug("Disk I/O worker stopped")

    def _save_stacked_field(self, stacked: StackedField, slide_id: int) -> str:
        """
        Save a stacked composite to disk.

        Returns the file path written.
        """
        field_dir = os.path.join(
            self._image_root,
            str(slide_id),
            f"{stacked.field_x}_{stacked.field_y}",
            "stacked",
        )
        os.makedirs(field_dir, exist_ok=True)

        if self._stacked_format == "jpeg":
            filename = "composite.jpg"
            filepath = os.path.join(field_dir, filename)
            cv2.imwrite(
                filepath,
                stacked.composite,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
        elif self._stacked_format == "png":
            filename = "composite.png"
            filepath = os.path.join(field_dir, filename)
            cv2.imwrite(filepath, stacked.composite)
        elif self._stacked_format == "tiff":
            filename = "composite.tiff"
            filepath = os.path.join(field_dir, filename)
            cv2.imwrite(filepath, stacked.composite)
        else:
            filename = "composite.jpg"
            filepath = os.path.join(field_dir, filename)
            cv2.imwrite(filepath, stacked.composite)

        # Save stacking metadata as JSON sidecar
        meta_path = os.path.join(field_dir, "stacking_meta.json")
        meta = {
            "block_size": stacked.block_size,
            "z_distribution": stacked.z_distribution,
            "stacking_duration_ms": stacked.stacking_duration_ms,
            "registration_shifts": stacked.registration_shifts,
            "avg_sharpness": float(np.mean(stacked.sharpness_map)),
            "min_sharpness": float(np.min(stacked.sharpness_map)),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.debug("Saved stacked field (%d, %d) → %s", stacked.field_x, stacked.field_y, filepath)
        return filepath

    # ----- Control -----

    def pause(self) -> None:
        """Pause the scan."""
        self._sequencer.pause()

    def resume(self) -> None:
        """Resume the scan."""
        self._sequencer.resume()

    def stop(self) -> None:
        """Stop the scan."""
        self._sequencer.stop()

    @property
    def is_running(self) -> bool:
        return self._sequencer.is_running

    @property
    def stacked_fields(self) -> list[StackedField]:
        with self._lock:
            return list(self._stacked_fields)
