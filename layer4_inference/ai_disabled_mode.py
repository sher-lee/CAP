"""
AI-Disabled Mode
=================
Convenience functions for running the system without a trained model.
When the model file is missing or inference is disabled in config,
the rest of the pipeline still needs valid (but empty/None) data
structures to flow through Layer 5 storage and Layer 7 visualization.

Usage:
    from cap.layer4_inference.ai_disabled_mode import get_disabled_results
    if model is None:
        results = get_disabled_results(slide_id)
        # results.severity_grades is None
        # results.plain_english_summary is None
"""

from __future__ import annotations

from cap.common.dataclasses import SlideResults
from cap.common.logging_setup import get_logger

logger = get_logger("inference.ai_disabled")


def get_disabled_results(slide_id: int) -> SlideResults:
    """
    Return a SlideResults object representing AI-disabled mode.

    All severity and summary fields are None. organism_counts is
    an empty dict. This allows downstream code to check:
        if results.severity_grades is not None:
            # display AI results
        else:
            # show "AI not available" in the UI

    Parameters
    ----------
    slide_id : int
        Database slide_id.

    Returns
    -------
    SlideResults
        Empty results with None severity fields.
    """
    logger.info(
        "Slide %d: generating AI-disabled placeholder results",
        slide_id,
    )
    return SlideResults(
        slide_id=slide_id,
        organism_counts={},
        severity_grades=None,
        overall_severity=None,
        flagged_field_ids=[],
        density_map=None,
        model_version="none",
        plain_english_summary=None,
    )


def is_ai_available(config_or_model) -> bool:
    """
    Quick check for whether AI inference is available.

    Accepts either a CAPConfig (checks config.inference.enabled and
    model file existence) or a model object (checks for None).

    Parameters
    ----------
    config_or_model : CAPConfig or object
        Either the app config or a loaded model (possibly None).

    Returns
    -------
    bool
        True if AI inference can run, False otherwise.
    """
    if config_or_model is None:
        return False

    # If it's a config object, check the enabled flag
    if hasattr(config_or_model, "inference"):
        import os
        cfg = config_or_model
        if not cfg.inference.enabled:
            return False
        if not os.path.isfile(cfg.inference.model_path):
            return False
        return True

    # Otherwise assume it's a model object — non-None means available
    return True
