
"""
Aggregator
===========
Computes slide-level aggregated results from per-field detections.
Calls the existing severity calculator (Layer 7) to produce severity
grades and plain-English summaries, then assembles a SlideResults
dataclass.

In AI-disabled mode (no detections), returns a SlideResults with
empty counts and None for all severity/summary fields.

Usage:
    from cap.layer4_inference.aggregator import aggregate_slide_results
    slide_results = aggregate_slide_results(slide_id, detections, field_grid, config)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional, TYPE_CHECKING

import numpy as np

from cap.common.dataclasses import Detection, SlideResults, SeverityGrade
from cap.common.logging_setup import get_logger
from cap.layer7_visualization.severity import (
    compute_all_severities,
    get_overall_severity,
    generate_summary,
)

if TYPE_CHECKING:
    from cap.config.config_loader import CAPConfig

logger = get_logger("inference.aggregator")


def aggregate_slide_results(
    slide_id: int,
    detections: list[Detection],
    config: CAPConfig,
    field_grid_size: Optional[tuple[int, int]] = None,
    model_version: str = "unknown",
) -> SlideResults:
    """
    Aggregate per-field detections into slide-level results.

    Parameters
    ----------
    slide_id : int
        Database slide_id.
    detections : list of Detection
        All detections for this slide (across all fields).
        Empty list in AI-disabled mode.
    config : CAPConfig
        Configuration (uses severity_thresholds, classes).
    field_grid_size : (rows, cols) or None
        Grid dimensions for the density map. If None, the density
        map is not computed.
    model_version : str
        Model version tag for the results.

    Returns
    -------
    SlideResults
        Aggregated results. In AI-disabled mode (empty detections),
        organism_counts is {}, and severity_grades, overall_severity,
        density_map, and plain_english_summary are all None.
    """
    # --- AI-disabled mode: no detections at all ---
    if not detections:
        logger.info("Slide %d: no detections — returning empty results", slide_id)
        return SlideResults(
            slide_id=slide_id,
            organism_counts={},
            severity_grades=None,
            overall_severity=None,
            flagged_field_ids=[],
            density_map=None,
            model_version=model_version,
            plain_english_summary=None,
        )

    # --- Count detections per class ---
    organism_counts: dict[str, int] = Counter()
    for det in detections:
        organism_counts[det.class_name] += 1

    # Ensure all configured classes appear (with 0 if not detected)
    for cls in config.inference.classes:
        if cls not in organism_counts:
            organism_counts[cls] = 0

    # Remove empty_artifact from severity consideration
    # (keep it in counts for informational purposes)
    counts_for_severity = {
        cls: count for cls, count in organism_counts.items()
        if cls != "empty_artifact"
    }

    # --- Severity grades via existing Layer 7 calculator ---
    severity_thresholds = config.inference.severity_thresholds
    severity_grades = compute_all_severities(counts_for_severity, severity_thresholds)
    overall = get_overall_severity(severity_grades)
    summary = generate_summary(counts_for_severity, severity_grades, overall)

    logger.info(
        "Slide %d: %d detections, overall severity %s",
        slide_id, len(detections), overall.value,
    )

    # --- Density map ---
    density_map = None
    if field_grid_size is not None:
        density_map = _build_density_map(detections, field_grid_size)

    # --- Flag high-density fields ---
    flagged = _flag_fields(detections, config)

    return SlideResults(
        slide_id=slide_id,
        organism_counts=dict(organism_counts),
        severity_grades=severity_grades,
        overall_severity=overall,
        flagged_field_ids=flagged,
        density_map=density_map,
        model_version=model_version,
        plain_english_summary=summary,
    )


def _build_density_map(
    detections: list[Detection],
    grid_size: tuple[int, int],
) -> np.ndarray:
    """
    Build a 2D density map of detection counts per grid cell.

    The density map is indexed by field position. Each cell contains
    the total number of detections (all classes) for that field.

    Parameters
    ----------
    detections : list of Detection
        All detections with field_id set.
    grid_size : (rows, cols)
        Dimensions of the field grid.

    Returns
    -------
    np.ndarray
        Shape (rows, cols), dtype float32. Each cell is the total
        detection count for that field position.
    """
    rows, cols = grid_size
    density = np.zeros((rows, cols), dtype=np.float32)

    # Group detections by field_id, then we'd need field positions.
    # Since Detection only carries field_id (not x/y grid coords),
    # we count per field_id and distribute evenly. The caller can
    # provide a field_id → (row, col) mapping in a future version.
    # For now, build a flat histogram by field_id.
    counts_by_field: dict[int, int] = Counter()
    for det in detections:
        counts_by_field[det.field_id] += 1

    # Distribute into the grid in raster order
    field_ids_sorted = sorted(counts_by_field.keys())
    for idx, fid in enumerate(field_ids_sorted):
        row = idx // cols
        col = idx % cols
        if row < rows and col < cols:
            density[row, col] = counts_by_field[fid]

    return density


def _flag_fields(
    detections: list[Detection],
    config: CAPConfig,
) -> list[int]:
    """
    Flag field_ids that have unusually high detection density.

    A field is flagged if its total detection count exceeds the
    4+ threshold for *any* organism class (i.e. it has "packed"
    levels of at least one organism in a single field).

    Also flags any field with ear_mite detections, since these
    are a rare and clinically significant finding.

    Parameters
    ----------
    detections : list of Detection
        All detections for the slide.
    config : CAPConfig
        Configuration (uses severity_thresholds).

    Returns
    -------
    list of int
        Sorted list of flagged field_ids.
    """
    # Count per (field_id, class)
    field_class_counts: dict[int, dict[str, int]] = defaultdict(Counter)
    for det in detections:
        field_class_counts[det.field_id][det.class_name] += 1

    severity_thresholds = config.inference.severity_thresholds
    default_thresholds = severity_thresholds.get("default", [1, 5, 15, 30])
    flagged: set[int] = set()

    for field_id, class_counts in field_class_counts.items():
        for cls, count in class_counts.items():
            if cls == "empty_artifact":
                continue

            # Flag ear mites at any count
            if cls == "ear_mites" and count > 0:
                flagged.add(field_id)
                continue

            # Flag if count hits the 4+ threshold for this class
            thresholds = severity_thresholds.get(cls, default_thresholds)
            if len(thresholds) >= 4 and count >= thresholds[3]:
                flagged.add(field_id)

    return sorted(flagged)
