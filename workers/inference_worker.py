"""
Inference Worker
=================
QThread that runs AI inference after the scan pipeline completes.
Loads the model, runs batch inference on all processed fields,
post-processes detections, aggregates slide-level results, and
stores everything in the database.

In AI-disabled mode (no model file or inference disabled), emits
inference_skipped and stores empty/None results so the rest of
the pipeline flows without error.

Usage:
    worker = InferenceWorker(app_context)
    worker.signals.inference_complete.connect(self.on_inference_done)
    worker.signals.inference_skipped.connect(self.on_inference_skipped)
    worker.start()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread

from cap.common.dataclasses import ProcessedFrame
from cap.common.logging_setup import get_logger
from cap.layer6_ui.signals import InferenceSignals
from cap.layer5_data import crud
from cap.layer5_data.audit import AuditLogger, EventType

if TYPE_CHECKING:
    from cap.app import AppContext

logger = get_logger("workers.inference")


class InferenceWorker(QThread):
    """
    Runs AI inference in a background thread.

    Lifecycle:
        1. Load model (or detect AI-disabled mode)
        2. Build ProcessedFrame list from pipeline stacked fields
        3. Run batch inference
        4. Post-process → Detection dataclasses
        5. Aggregate → SlideResults
        6. Store detections + results in DB
        7. Emit inference_complete or inference_skipped
    """

    def __init__(self, app_context: AppContext) -> None:
        super().__init__()
        self._ctx = app_context
        self.signals = InferenceSignals()

    def run(self) -> None:
        """Thread entry point."""
        slide_id = getattr(self._ctx, "current_slide_id", None)
        config = self._ctx.config
        audit = AuditLogger(self._ctx.db)

        try:
            # ---------------------------------------------------------------
            # Step 1: Load model
            # ---------------------------------------------------------------
            from cap.layer4_inference.model_loader import load_model, get_model_version

            model = load_model(config)
            model_version = get_model_version(model)

            if model is None:
                # AI-disabled mode
                reason = (
                    "AI disabled in config"
                    if not config.inference.enabled
                    else "Model file not found"
                )
                logger.info("Inference skipped: %s", reason)

                # Store empty results in DB
                from cap.layer4_inference.ai_disabled_mode import get_disabled_results

                disabled_results = get_disabled_results(slide_id or 0)

                if slide_id:
                    crud.insert_results(
                        self._ctx.db,
                        slide_id=slide_id,
                        organism_counts=disabled_results.organism_counts,
                        severity_score="0",
                        severity_grades=None,
                        model_version="none",
                        plain_english_summary=None,
                    )
                    crud.update_slide_status(self._ctx.db, slide_id, "complete")

                audit.log(
                    EventType.INFERENCE_SKIPPED,
                    details=f"Slide {slide_id}: {reason}",
                )

                self.signals.inference_skipped.emit(reason)
                return

            # ---------------------------------------------------------------
            # Step 2: Build ProcessedFrame list from stacked fields
            # ---------------------------------------------------------------
            self.signals.inference_started.emit(slide_id or 0)
            audit.log(EventType.INFERENCE_STARTED, details=f"Slide {slide_id}")

            if slide_id:
                crud.update_slide_status(self._ctx.db, slide_id, "inferring")

            pipeline_result = getattr(self._ctx, "last_pipeline_result", None)
            if pipeline_result is None:
                raise RuntimeError("No pipeline result found — was the scan completed?")

            stacked_fields = pipeline_result.get("stacked_fields", [])
            if not stacked_fields:
                raise RuntimeError("No stacked fields — scan may have failed")

            frames: list[ProcessedFrame] = []
            for sf in stacked_fields:
                frames.append(ProcessedFrame(
                    slide_id=slide_id or 0,
                    field_x=sf.field_x,
                    field_y=sf.field_y,
                    rgb_data=sf.composite,
                    stacked=True,
                    focus_score=float(sf.sharpness_map.mean()),
                ))

            total_fields = len(frames)
            logger.info("Inference starting: %d fields, model=%s", total_fields, model_version)

            # ---------------------------------------------------------------
            # Step 3: Run batch inference
            # ---------------------------------------------------------------
            from cap.layer4_inference.inference import run_inference

            inference_start = time.monotonic()
            results_paired = run_inference(model, frames, config)
            inference_duration = time.monotonic() - inference_start

            self.signals.inference_progress.emit(total_fields, total_fields)

            # ---------------------------------------------------------------
            # Step 4: Post-process
            # ---------------------------------------------------------------
            from cap.layer4_inference.postprocess import extract_all_detections

            field_id_map = getattr(self._ctx, "field_id_map", {})
            detections = extract_all_detections(
                results_paired, field_id_map, config, model_version,
            )

            logger.info(
                "Inference complete: %d detections from %d fields in %.1fs",
                len(detections), total_fields, inference_duration,
            )

            # ---------------------------------------------------------------
            # Step 5: Aggregate
            # ---------------------------------------------------------------
            from cap.layer4_inference.aggregator import aggregate_slide_results

            # Compute grid size from stacked fields
            if stacked_fields:
                max_x = max(sf.field_x for sf in stacked_fields) + 1
                max_y = max(sf.field_y for sf in stacked_fields) + 1
                grid_size = (max_y, max_x)
            else:
                grid_size = None

            slide_results = aggregate_slide_results(
                slide_id=slide_id or 0,
                detections=detections,
                config=config,
                field_grid_size=grid_size,
                model_version=model_version,
            )

            # ---------------------------------------------------------------
            # Step 6: Store in DB
            # ---------------------------------------------------------------
            if slide_id:
                # Store individual detections
                if detections:
                    det_dicts = [{
                        "field_id": d.field_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox_x": d.bbox[0],
                        "bbox_y": d.bbox[1],
                        "bbox_w": d.bbox[2],
                        "bbox_h": d.bbox[3],
                        "model_version": d.model_version,
                    } for d in detections]
                    crud.insert_detections_batch(self._ctx.db, det_dicts)

                # Store aggregated results
                severity_grades_serializable = None
                if slide_results.severity_grades is not None:
                    severity_grades_serializable = {
                        cls: grade.value
                        for cls, grade in slide_results.severity_grades.items()
                    }

                crud.insert_results(
                    self._ctx.db,
                    slide_id=slide_id,
                    organism_counts=slide_results.organism_counts,
                    severity_score=(
                        slide_results.overall_severity.value
                        if slide_results.overall_severity
                        else "0"
                    ),
                    severity_grades=severity_grades_serializable,
                    flagged_field_ids=slide_results.flagged_field_ids,
                    model_version=model_version,
                    plain_english_summary=slide_results.plain_english_summary,
                )

                crud.update_slide_status(self._ctx.db, slide_id, "complete")

            # Store results in context for downstream use
            self._ctx.last_slide_results = slide_results

            audit.log(
                EventType.INFERENCE_COMPLETED,
                details=(
                    f"Slide {slide_id}: {len(detections)} detections, "
                    f"severity={slide_results.overall_severity.value if slide_results.overall_severity else '0'}, "
                    f"{inference_duration:.1f}s"
                ),
            )

            self.signals.inference_complete.emit(slide_id or 0, slide_results)

        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            audit.log(
                EventType.INFERENCE_COMPLETED,
                details=f"Slide {slide_id}: FAILED — {exc}",
            )
            self.signals.inference_failed.emit(str(exc))
