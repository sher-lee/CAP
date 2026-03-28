"""
Layer 4: AI Inference Engine
=============================
YOLOv11 model loading, batch inference, post-processing, and
slide-level result aggregation.

Supports AI-disabled mode: when the model file is missing or
inference is disabled in config, all functions return None or
empty results so the rest of the pipeline flows without error.

Public API:
    load_model(config) → model or None
    run_inference(model, frames, config) → [(frame, result), ...]
    extract_all_detections(results, field_id_map, config) → [Detection, ...]
    aggregate_slide_results(slide_id, detections, config) → SlideResults
    get_disabled_results(slide_id) → SlideResults (empty/None fields)
    is_ai_available(config_or_model) → bool
"""

from cap.layer4_inference.model_loader import load_model, get_model_version
from cap.layer4_inference.inference import run_inference, run_single_inference
from cap.layer4_inference.postprocess import extract_detections, extract_all_detections
from cap.layer4_inference.aggregator import aggregate_slide_results
from cap.layer4_inference.ai_disabled_mode import get_disabled_results, is_ai_available

__all__ = [
    "load_model",
    "get_model_version",
    "run_inference",
    "run_single_inference",
    "extract_detections",
    "extract_all_detections",
    "aggregate_slide_results",
    "get_disabled_results",
    "is_ai_available",
]
