
"""
Model Loader
=============
Loads a YOLOv11 (Ultralytics) model from a .pt file.
Returns None if the model file is missing, inference is disabled,
or if ultralytics/torch are not installed — allowing the system
to run fully in AI-disabled mode.

Usage:
    from cap.layer4_inference.model_loader import load_model
    model = load_model(config)
    if model is None:
        # AI-disabled mode: skip inference
        ...
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from cap.common.logging_setup import get_logger

if TYPE_CHECKING:
    from cap.config.config_loader import CAPConfig

logger = get_logger("inference.model_loader")


def load_model(config: CAPConfig) -> Optional[object]:
    """
    Load a YOLOv11 model from the path specified in config.

    Parameters
    ----------
    config : CAPConfig
        Application configuration. Uses ``config.inference.enabled``
        and ``config.inference.model_path``.

    Returns
    -------
    ultralytics.YOLO or None
        The loaded model object, or None if:
        - ``config.inference.enabled`` is False
        - The model file does not exist
        - ultralytics or torch could not be imported
    """
    if not config.inference.enabled:
        logger.info("AI inference is disabled in configuration")
        return None

    model_path = config.inference.model_path

    if not os.path.isfile(model_path):
        logger.warning(
            "Model file not found: %s — running in AI-disabled mode",
            os.path.abspath(model_path),
        )
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning(
            "ultralytics package not installed — running in AI-disabled mode. "
            "Install with: pip install ultralytics"
        )
        return None

    try:
        model = YOLO(model_path)
        logger.info(
            "Model loaded successfully: %s (%d classes)",
            model_path,
            len(model.names) if hasattr(model, "names") else -1,
        )
        return model
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", model_path, exc)
        return None


def get_model_version(model: object) -> str:
    """
    Extract a version string from a loaded YOLO model.

    Falls back to 'unknown' if the model metadata is not available.

    Parameters
    ----------
    model : ultralytics.YOLO
        A loaded YOLO model.

    Returns
    -------
    str
        A version identifier string.
    """
    if model is None:
        return "none"

    # Ultralytics stores training metadata in model.ckpt or model.overrides
    try:
        if hasattr(model, "ckpt") and isinstance(model.ckpt, dict):
            train_args = model.ckpt.get("train_args", {})
            name = train_args.get("name", "")
            if name:
                return str(name)

        # Fall back to the filename stem
        if hasattr(model, "ckpt_path"):
            return os.path.splitext(os.path.basename(model.ckpt_path))[0]
    except Exception:
        pass

    return "unknown"
