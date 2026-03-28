"""
Post-Processing
================
Converts raw Ultralytics YOLO Results objects into CAP Detection
dataclasses. Applies confidence thresholding and NMS (both are
also applied by Ultralytics during predict(), but this module
provides a second pass with CAP-specific thresholds and handles
the None → empty-list conversion for AI-disabled mode).

Usage:
    from cap.layer4_inference.postprocess import extract_detections
    detections = extract_detections(frame, yolo_result, config)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from cap.common.dataclasses import Detection, ProcessedFrame
from cap.common.logging_setup import get_logger

if TYPE_CHECKING:
    from cap.config.config_loader import CAPConfig

logger = get_logger("inference.postprocess")


def extract_detections(
    frame: ProcessedFrame,
    yolo_result: Optional[object],
    field_id: int,
    config: CAPConfig,
    model_version: str = "unknown",
) -> list[Detection]:
    """
    Convert a single YOLO Results object into a list of Detection dataclasses.

    Parameters
    ----------
    frame : ProcessedFrame
        The frame that was inferred on (used for metadata only).
    yolo_result : ultralytics Results object or None
        Raw output from model.predict(). None in AI-disabled mode.
    field_id : int
        Database field_id for this field (from Layer 5).
    config : CAPConfig
        Configuration (uses confidence_threshold, nms_iou_threshold, classes).
    model_version : str
        Version tag to stamp on each Detection.

    Returns
    -------
    list of Detection
        Filtered detections for this field. Empty list if yolo_result
        is None or no detections passed the threshold.
    """
    if yolo_result is None:
        return []

    conf_threshold = config.inference.confidence_threshold
    known_classes = set(config.inference.classes)
    detections: list[Detection] = []

    try:
        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i])

            # Second-pass confidence filter (Ultralytics already filters,
            # but config may have been updated between predict and here)
            if confidence < conf_threshold:
                continue

            # Get class name from model's class map
            class_id = int(boxes.cls[i])
            class_name = yolo_result.names.get(class_id, f"class_{class_id}")

            # Skip classes not in our configured list
            if class_name not in known_classes:
                logger.debug(
                    "Skipping unknown class '%s' (id=%d) on field %d",
                    class_name, class_id, field_id,
                )
                continue

            # Extract bounding box — Ultralytics xywh format
            # boxes.xywh gives center_x, center_y, width, height
            xywh = boxes.xywh[i]
            cx, cy, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])

            # Convert to top-left x, y for storage (matches crud.insert_detection)
            bbox_x = cx - w / 2.0
            bbox_y = cy - h / 2.0

            detections.append(Detection(
                field_id=field_id,
                class_name=class_name,
                confidence=round(confidence, 4),
                bbox=(bbox_x, bbox_y, w, h),
                model_version=model_version,
            ))

    except Exception as exc:
        logger.error(
            "Error extracting detections for field %d: %s",
            field_id, exc,
        )
        return []

    logger.debug(
        "Field %d: %d detections after filtering (threshold=%.2f)",
        field_id, len(detections), conf_threshold,
    )

    return detections


def extract_all_detections(
    results_paired: list[tuple[ProcessedFrame, Optional[object]]],
    field_id_map: dict[tuple[int, int], int],
    config: CAPConfig,
    model_version: str = "unknown",
) -> list[Detection]:
    """
    Batch-convert all inference results into Detection dataclasses.

    Parameters
    ----------
    results_paired : list of (ProcessedFrame, Results)
        Output from inference.run_inference().
    field_id_map : dict
        Mapping of (field_x, field_y) → database field_id.
        Fields not in this map are skipped with a warning.
    config : CAPConfig
        Configuration.
    model_version : str
        Version tag for all detections.

    Returns
    -------
    list of Detection
        All detections across all fields. Empty list in AI-disabled mode.
    """
    all_detections: list[Detection] = []

    for frame, yolo_result in results_paired:
        field_key = (frame.field_x, frame.field_y)
        field_id = field_id_map.get(field_key)

        if field_id is None:
            logger.warning(
                "No field_id found for field position (%d, %d) — skipping",
                frame.field_x, frame.field_y,
            )
            continue

        field_detections = extract_detections(
            frame, yolo_result, field_id, config, model_version,
        )
        all_detections.extend(field_detections)

    logger.info(
        "Post-processing complete: %d total detections from %d fields",
        len(all_detections), len(results_paired),
    )

    return all_detections
