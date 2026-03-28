"""
CAP Common Dataclasses
======================
Formal data contracts between layers. Every object passed through
inter-layer queues or callbacks is defined here. No layer should
invent its own ad-hoc dicts or tuples for cross-layer communication.

These dataclasses are intentionally simple containers — no business
logic, no database access, no imports from any layer module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FieldStatus(Enum):
    """Status of a single field during the scan lifecycle."""
    PENDING = "pending"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    PROCESSING = "processing"
    PROCESSED = "processed"
    STACKING = "stacking"
    STACKED = "stacked"
    FAILED = "failed"
    SKIPPED = "skipped"          # outside scan region polygon


class SlideStatus(Enum):
    """Status of a slide through the full pipeline."""
    PENDING = "pending"
    SCANNING = "scanning"
    SCAN_COMPLETE = "scan_complete"
    INFERRING = "inferring"
    COMPLETE = "complete"
    FAILED = "failed"


class SeverityGrade(Enum):
    """Semi-quantitative veterinary ear cytology severity scale."""
    NONE = "0"
    RARE = "1+"
    MODERATE = "2+"
    MANY = "3+"
    PACKED = "4+"


# ---------------------------------------------------------------------------
# Layer 1 → Layer 2: Focus map from preliminary focus routine
# ---------------------------------------------------------------------------

@dataclass
class FocusMapResult:
    """
    Output of the PreliminaryFocus module. Describes the slide's focal
    surface so the CaptureSequencer can predict the best Z for each field.

    Passed: Layer 1 → Layer 2 (CaptureSequencer) and Layer 5 (stored in DB).
    """
    sample_points: list[tuple[int, int, float]]
    """List of (motor_x, motor_y, best_z) measured during focus map sampling."""

    surface_coefficients: np.ndarray
    """Fitted 2nd-order polynomial coefficients [a, b, c, d, e, f] for
    z = a + b*x + c*y + d*x² + e*y² + f*x*y."""

    grid_size: tuple[int, int]
    """Grid dimensions used for sampling, e.g. (3, 3) or (5, 5)."""

    fit_residual: float
    """RMS residual of the surface fit. High values indicate irregular
    slide surface that the polynomial model cannot capture well."""


# ---------------------------------------------------------------------------
# Layer 2 → Layer 3: Raw camera frame with metadata
# ---------------------------------------------------------------------------

@dataclass
class RawFrame:
    """
    A single raw Bayer frame captured from the camera at one Z-depth.
    Passed through the image queue from the capture thread to the
    processing thread.

    Passed: Layer 2 (CaptureSequencer) → Layer 3 (Processing pipeline).
    """
    slide_id: int
    field_x: int
    """X grid coordinate of this field in the scan region."""
    field_y: int
    """Y grid coordinate of this field in the scan region."""
    z_depth: int
    """Z-depth index (0–5 for 6-depth capture)."""
    timestamp: float
    """Time of capture (time.monotonic() for relative timing)."""
    bayer_data: np.ndarray
    """Raw Bayer frame, shape (H, W), dtype uint8 (RG8) or uint16 (RG10)."""
    motor_position: tuple[int, int, int]
    """Actual motor position (x, y, z) in microsteps at time of capture."""


# ---------------------------------------------------------------------------
# Layer 3 → Layer 4: Processed RGB frame ready for inference
# ---------------------------------------------------------------------------

@dataclass
class ProcessedFrame:
    """
    A debayered, normalized, noise-reduced image ready for AI inference.
    May be a single Z-depth frame or a focus-stacked composite.

    Passed: Layer 3 (Processing) → Layer 4 (Inference queue).
    """
    slide_id: int
    field_x: int
    field_y: int
    rgb_data: np.ndarray
    """Processed RGB image, shape (H, W, 3), dtype uint8."""
    stacked: bool
    """True if this is a focus-stacked composite; False if single Z-depth."""
    focus_score: float
    """Overall sharpness score (Laplacian variance of the full frame)."""


# ---------------------------------------------------------------------------
# Layer 2 stacker → Layer 5 / Layer 7: Focus-stacked composite with metadata
# ---------------------------------------------------------------------------

@dataclass
class StackedField:
    """
    A fully focus-stacked composite image for one field, with metadata
    about the stacking process. Written to disk and recorded in the DB.

    Passed: Layer 2 (FocusStacker) → Layer 5 (Data) and Layer 7 (Visualization).
    """
    slide_id: int
    field_x: int
    field_y: int
    composite: np.ndarray
    """All-in-focus composite image, shape (H, W, 3), dtype uint8."""
    sharpness_map: np.ndarray
    """Per-block average sharpness scores, shape (blocks_y, blocks_x), dtype float32."""
    z_distribution: dict[int, int]
    """Histogram: {z_depth_index: number_of_blocks_selected_from_this_depth}."""
    stacking_duration_ms: float
    """Time taken for registration + stacking (milliseconds)."""
    registration_shifts: list[tuple[float, float]]
    """Sub-pixel (dx, dy) shifts computed for frames 1–5 relative to frame 0.
    Empty list if registration was skipped."""
    block_size: int
    """Block size used for this stacking operation (e.g. 16 or 32)."""


# ---------------------------------------------------------------------------
# Layer 4 → Layer 5: Individual organism detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    A single organism detection from YOLOv11 inference on one field.

    Passed: Layer 4 (Inference) → Layer 5 (Data, written to detections table).
    """
    field_id: int
    """Database field_id (assigned after the field is recorded in Layer 5)."""
    class_name: str
    """Organism class label, e.g. 'cocci_small', 'yeast', 'rods'."""
    confidence: float
    """Model confidence score, 0.0–1.0."""
    bbox: tuple[float, float, float, float]
    """Bounding box (x, y, width, height) in pixel coordinates on the processed frame."""
    model_version: str
    """Version tag of the model that produced this detection."""


# ---------------------------------------------------------------------------
# Layer 4 → Layer 5 / Layer 7: Slide-level aggregated results
# ---------------------------------------------------------------------------

@dataclass
class SlideResults:
    """
    Aggregated detection results for an entire slide. Computed after
    inference completes on all fields.

    Passed: Layer 4 (Aggregator) → Layer 5 (Data) and Layer 7 (PDF report).
    """
    slide_id: int
    organism_counts: dict[str, int]
    """Per-class total detection counts, e.g. {'cocci_small': 42, 'yeast': 17}."""
    severity_grades: Optional[dict[str, SeverityGrade]] = None
    """Per-class severity grade, e.g. {'cocci_small': SeverityGrade.MODERATE}.
    None in AI-disabled mode (no model loaded)."""
    overall_severity: Optional[SeverityGrade] = None
    """Highest severity grade across all organism classes.
    None in AI-disabled mode."""
    flagged_field_ids: list[int] = field(default_factory=list)
    """Field IDs flagged for high density or rare findings."""
    density_map: Optional[np.ndarray] = None
    """2D array of detection density per slide region, shape (grid_y, grid_x)."""
    model_version: str = ""
    """Model version that produced these results."""
    plain_english_summary: Optional[str] = None
    """Auto-generated summary, e.g. 'Moderate yeast infection (2+) with occasional cocci (1+).'
    None in AI-disabled mode."""


# ---------------------------------------------------------------------------
# Layer 2 → Layer 6: Scan progress for UI updates
# ---------------------------------------------------------------------------

@dataclass
class ScanProgress:
    """
    Progress snapshot emitted by the CaptureSequencer during scanning.
    The UI thread receives this via a Qt signal to update the progress
    bar, ETA, and stage position map.

    Passed: Layer 2 (CaptureSequencer) → Layer 6 (UI, via signal).
    """
    fields_completed: int
    fields_total: int
    current_x: int
    """Current motor X position in microsteps."""
    current_y: int
    """Current motor Y position in microsteps."""
    eta_seconds: float
    """Estimated time remaining in seconds."""
    current_field_status: str
    """Human-readable status of the current field, e.g. 'capturing Z-depth 3/6'."""


# ---------------------------------------------------------------------------
# Layer 1: Scan region definition from the polygon drawing tool
# ---------------------------------------------------------------------------

@dataclass
class ScanRegion:
    """
    Defines the technician-drawn scan area. The polygon vertices are in
    motor step coordinates after coordinate mapping from the UI.

    Created: Layer 6 (UI polygon tool) → Layer 1 (ScanRegionManager).
    """
    polygon_vertices: list[tuple[int, int]]
    """Ordered list of (motor_x, motor_y) vertices defining the scan polygon."""
    field_positions: list[tuple[int, int]] = field(default_factory=list)
    """Generated list of (motor_x, motor_y) field centers within the polygon."""
    field_count: int = 0
    """Number of fields within the polygon (set after grid generation)."""
    estimated_scan_time_sec: float = 0.0
    """Estimated scan duration based on field count and fields_per_second."""
    estimated_disk_usage_mb: float = 0.0
    """Estimated disk usage based on field count and image sizes."""
    preset_name: Optional[str] = None
    """Name of preset used, if any (e.g. 'full_slide', 'center_half')."""
