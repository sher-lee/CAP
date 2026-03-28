"""
Inference Runner
=================
Runs YOLOv11 batch inference on processed field images.
Returns raw model output that postprocess.py will filter.

If the model is None (AI-disabled mode), all functions return
empty lists so downstream code flows without error.

Usage:
    from cap.layer4_inference.inference import run_inference
    raw_results = run_inference(model, frames, config)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from cap.common.dataclasses import ProcessedFrame
from cap.common.logging_setup import get_logger

if TYPE_CHECKING:
    from cap.config.config_loader import CAPConfig

logger = get_logger("inference.runner")


def run_inference(
    model: Optional[object],
    frames: list[ProcessedFrame],
    config: CAPConfig,
) -> list[tuple[ProcessedFrame, object]]:
    """
    Run batch inference on a list of processed frames.

    Parameters
    ----------
    model : ultralytics.YOLO or None
        The loaded YOLO model. If None, returns empty results
        for every frame (AI-disabled mode).
    frames : list of ProcessedFrame
        Processed field images ready for inference.
    config : CAPConfig
        Configuration (uses batch_size, confidence_threshold).

    Returns
    -------
    list of (ProcessedFrame, Results)
        Paired list of frames and their raw YOLO Results objects.
        In AI-disabled mode, the Results element is None for each frame.
    """
    if model is None:
        logger.debug(
            "Model is None — returning empty results for %d frames",
            len(frames),
        )
        return [(frame, None) for frame in frames]

    if not frames:
        return []

    batch_size = config.inference.batch_size
    conf_threshold = config.inference.confidence_threshold
    results_paired: list[tuple[ProcessedFrame, object]] = []

    # Process in batches
    for batch_start in range(0, len(frames), batch_size):
        batch_frames = frames[batch_start : batch_start + batch_size]
        batch_images = [frame.rgb_data for frame in batch_frames]

        logger.debug(
            "Running inference batch: frames %d–%d of %d",
            batch_start,
            batch_start + len(batch_frames) - 1,
            len(frames),
        )

        try:
            # Ultralytics YOLO.predict() accepts a list of numpy arrays
            batch_results = model.predict(
                source=batch_images,
                conf=conf_threshold,
                verbose=False,
            )

            for frame, result in zip(batch_frames, batch_results):
                results_paired.append((frame, result))

        except Exception as exc:
            logger.error(
                "Inference failed on batch starting at frame %d: %s",
                batch_start,
                exc,
            )
            # Mark the entire batch as having no detections
            for frame in batch_frames:
                results_paired.append((frame, None))

    logger.info(
        "Inference complete: %d frames processed in %d batches",
        len(frames),
        (len(frames) + batch_size - 1) // batch_size,
    )

    return results_paired


def run_single_inference(
    model: Optional[object],
    image: np.ndarray,
    config: CAPConfig,
) -> Optional[object]:
    """
    Run inference on a single image. Convenience wrapper for
    real-time or on-demand re-inference of individual fields.

    Parameters
    ----------
    model : ultralytics.YOLO or None
        The loaded model.
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    config : CAPConfig
        Configuration.

    Returns
    -------
    ultralytics Results object or None
        Raw YOLO results, or None if model is unavailable.
    """
    if model is None:
        return None

    try:
        results = model.predict(
            source=image,
            conf=config.inference.confidence_threshold,
            verbose=False,
        )
        return results[0] if results else None
    except Exception as exc:
        logger.error("Single-frame inference failed: %s", exc)
        return None
