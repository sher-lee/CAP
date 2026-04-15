"""
CAP Phase 7 Test Suite
=======================
Tests Layer 4 (AI Inference) in isolation and integrated with
Layers 1–7. Designed to run on Windows in simulation mode with
no trained model file (AI-disabled mode).

Run from the project root (the folder containing the 'cap' package):
    python -m pytest tests/test_phase7.py -v

Or run directly:
    python tests/test_phase7.py

Requirements:
    pip install pytest numpy pyyaml
    (ultralytics/torch NOT required — tests cover AI-disabled mode)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so 'cap' is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="cap_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def config():
    """Load the real CAPConfig from cap_config.yaml."""
    from cap.config.config_loader import load_config
    return load_config()


@pytest.fixture
def config_ai_disabled(config):
    """A config with inference.enabled = False."""
    config.inference.enabled = False
    return config


@pytest.fixture
def db(temp_dir):
    """A fresh in-memory-like database in a temp directory."""
    from cap.layer5_data.db_manager import DatabaseManager
    db_path = os.path.join(temp_dir, "test_cap.db")
    db = DatabaseManager(db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def sample_processed_frames():
    """Create a list of ProcessedFrame objects with synthetic image data."""
    from cap.common.dataclasses import ProcessedFrame

    frames = []
    for x in range(3):
        for y in range(3):
            frame = ProcessedFrame(
                slide_id=1,
                field_x=x,
                field_y=y,
                rgb_data=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                stacked=True,
                focus_score=0.85 + np.random.random() * 0.1,
            )
            frames.append(frame)
    return frames


@pytest.fixture
def sample_detections():
    """Create a list of Detection objects for testing aggregation."""
    from cap.common.dataclasses import Detection

    detections = []
    # 7 cocci_small across 3 fields → should be 2+ (moderate)
    for i in range(7):
        detections.append(Detection(
            field_id=(i % 3) + 1,
            class_name="cocci_small",
            confidence=0.85 + (i * 0.01),
            bbox=(50.0 + i * 10, 50.0, 20.0, 20.0),
            model_version="test_v1",
        ))
    # 2 yeast → should be 1+ (rare)
    for i in range(2):
        detections.append(Detection(
            field_id=1,
            class_name="yeast",
            confidence=0.75,
            bbox=(100.0, 100.0 + i * 30, 15.0, 15.0),
            model_version="test_v1",
        ))
    # 1 ear_mite → should be 1+ and flagged
    detections.append(Detection(
        field_id=2,
        class_name="ear_mites",
        confidence=0.92,
        bbox=(200.0, 200.0, 30.0, 30.0),
        model_version="test_v1",
    ))
    return detections


# ===========================================================================
# Test 1: Model Loader
# ===========================================================================

class TestModelLoader:
    """Tests for cap.layer4_inference.model_loader."""

    def test_load_model_disabled(self, config_ai_disabled):
        """When inference.enabled is False, load_model returns None."""
        from cap.layer4_inference.model_loader import load_model
        model = load_model(config_ai_disabled)
        assert model is None

    def test_load_model_missing_file(self, config):
        """When model .pt file doesn't exist, load_model returns None."""
        from cap.layer4_inference.model_loader import load_model
        config.inference.enabled = True
        config.inference.model_path = "/nonexistent/path/model.pt"
        model = load_model(config)
        assert model is None

    def test_load_model_no_ultralytics(self, config, temp_dir):
        """When ultralytics is not importable, load_model returns None."""
        from cap.layer4_inference.model_loader import load_model

        # Create a dummy .pt file so the file-exists check passes
        dummy_pt = os.path.join(temp_dir, "dummy.pt")
        with open(dummy_pt, "wb") as f:
            f.write(b"fake model data")
        config.inference.enabled = True
        config.inference.model_path = dummy_pt

        # Mock the import to fail
        with patch.dict("sys.modules", {"ultralytics": None}):
            with patch("builtins.__import__", side_effect=_block_ultralytics_import):
                model = load_model(config)
                assert model is None

    def test_get_model_version_none(self):
        """get_model_version(None) returns 'none'."""
        from cap.layer4_inference.model_loader import get_model_version
        assert get_model_version(None) == "none"

    def test_get_model_version_with_ckpt_path(self):
        """get_model_version extracts name from ckpt_path."""
        from cap.layer4_inference.model_loader import get_model_version
        mock_model = MagicMock()
        mock_model.ckpt = {}
        mock_model.ckpt_path = "/models/ear_cytology_v3.pt"
        assert get_model_version(mock_model) == "ear_cytology_v3"


def _block_ultralytics_import(name, *args, **kwargs):
    """Import hook that blocks ultralytics but allows everything else."""
    if name == "ultralytics" or (isinstance(name, str) and name.startswith("ultralytics")):
        raise ImportError("Mocked: ultralytics not available")
    return original_import(name, *args, **kwargs)


original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


# ===========================================================================
# Test 2: Inference Runner
# ===========================================================================

class TestInferenceRunner:
    """Tests for cap.layer4_inference.inference."""

    def test_run_inference_no_model(self, config, sample_processed_frames):
        """With model=None, run_inference returns (frame, None) pairs."""
        from cap.layer4_inference.inference import run_inference
        results = run_inference(None, sample_processed_frames, config)

        assert len(results) == len(sample_processed_frames)
        for frame, yolo_result in results:
            assert yolo_result is None
            assert frame.slide_id == 1

    def test_run_inference_empty_frames(self, config):
        """With no frames, run_inference returns empty list."""
        from cap.layer4_inference.inference import run_inference
        results = run_inference(None, [], config)
        assert results == []

    def test_run_inference_batching(self, config, sample_processed_frames):
        """Verify batching works with a mock model."""
        from cap.layer4_inference.inference import run_inference

        config.inference.batch_size = 4  # 9 frames → 3 batches

        mock_model = MagicMock()
        # model.predict() should return one result per image
        mock_model.predict.side_effect = lambda source, **kw: [
            MagicMock() for _ in source
        ]

        results = run_inference(mock_model, sample_processed_frames, config)

        assert len(results) == 9
        assert mock_model.predict.call_count == 3  # ceil(9/4) = 3 batches

    def test_run_single_inference_no_model(self, config):
        """run_single_inference(None, ...) returns None."""
        from cap.layer4_inference.inference import run_single_inference
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = run_single_inference(None, image, config)
        assert result is None


# ===========================================================================
# Test 3: Post-Processing
# ===========================================================================

class TestPostProcessing:
    """Tests for cap.layer4_inference.postprocess."""

    def test_extract_detections_none_result(self, config):
        """None YOLO result → empty detection list."""
        from cap.layer4_inference.postprocess import extract_detections
        from cap.common.dataclasses import ProcessedFrame

        frame = ProcessedFrame(
            slide_id=1, field_x=0, field_y=0,
            rgb_data=np.zeros((640, 640, 3), dtype=np.uint8),
            stacked=True, focus_score=0.9,
        )
        detections = extract_detections(frame, None, field_id=1, config=config)
        assert detections == []

    def test_extract_detections_with_mock_results(self, config):
        """Mock YOLO boxes are correctly converted to Detection objects."""
        from cap.layer4_inference.postprocess import extract_detections
        from cap.common.dataclasses import ProcessedFrame

        frame = ProcessedFrame(
            slide_id=1, field_x=0, field_y=0,
            rgb_data=np.zeros((640, 640, 3), dtype=np.uint8),
            stacked=True, focus_score=0.9,
        )

        # Build a mock YOLO result with 2 detections
        mock_result = MagicMock()
        mock_result.names = {0: "cocci_small", 1: "yeast"}
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 2
        mock_boxes.conf = [0.95, 0.72]
        mock_boxes.cls = [0, 1]
        # xywh: center_x, center_y, width, height
        mock_boxes.xywh = [
            [320.0, 240.0, 40.0, 40.0],  # cocci_small
            [100.0, 100.0, 20.0, 20.0],  # yeast
        ]
        mock_result.boxes = mock_boxes

        detections = extract_detections(
            frame, mock_result, field_id=42, config=config, model_version="test_v1",
        )

        assert len(detections) == 2
        assert detections[0].class_name == "cocci_small"
        assert detections[0].confidence == 0.95
        assert detections[0].field_id == 42
        assert detections[0].model_version == "test_v1"
        # Check bbox conversion: center (320,240) size (40,40) → top-left (300,220)
        assert abs(detections[0].bbox[0] - 300.0) < 0.01
        assert abs(detections[0].bbox[1] - 220.0) < 0.01

        assert detections[1].class_name == "yeast"

    def test_extract_detections_filters_low_confidence(self, config):
        """Detections below confidence_threshold are filtered out."""
        from cap.layer4_inference.postprocess import extract_detections
        from cap.common.dataclasses import ProcessedFrame

        config.inference.confidence_threshold = 0.8  # raise threshold

        frame = ProcessedFrame(
            slide_id=1, field_x=0, field_y=0,
            rgb_data=np.zeros((640, 640, 3), dtype=np.uint8),
            stacked=True, focus_score=0.9,
        )

        mock_result = MagicMock()
        mock_result.names = {0: "cocci_small"}
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 2
        mock_boxes.conf = [0.9, 0.6]  # second one below threshold
        mock_boxes.cls = [0, 0]
        mock_boxes.xywh = [
            [100.0, 100.0, 20.0, 20.0],
            [200.0, 200.0, 20.0, 20.0],
        ]
        mock_result.boxes = mock_boxes

        detections = extract_detections(frame, mock_result, field_id=1, config=config)
        assert len(detections) == 1
        assert detections[0].confidence == 0.9

    def test_extract_detections_skips_unknown_classes(self, config):
        """Classes not in config.inference.classes are skipped."""
        from cap.layer4_inference.postprocess import extract_detections
        from cap.common.dataclasses import ProcessedFrame

        frame = ProcessedFrame(
            slide_id=1, field_x=0, field_y=0,
            rgb_data=np.zeros((640, 640, 3), dtype=np.uint8),
            stacked=True, focus_score=0.9,
        )

        mock_result = MagicMock()
        mock_result.names = {0: "cocci_small", 1: "alien_bacteria"}
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 2
        mock_boxes.conf = [0.9, 0.9]
        mock_boxes.cls = [0, 1]
        mock_boxes.xywh = [
            [100.0, 100.0, 20.0, 20.0],
            [200.0, 200.0, 20.0, 20.0],
        ]
        mock_result.boxes = mock_boxes

        detections = extract_detections(frame, mock_result, field_id=1, config=config)
        assert len(detections) == 1
        assert detections[0].class_name == "cocci_small"

    def test_extract_all_detections_batch(self, config, sample_processed_frames):
        """extract_all_detections maps field positions to field_ids."""
        from cap.layer4_inference.postprocess import extract_all_detections

        # AI-disabled mode: all None results
        results_paired = [(f, None) for f in sample_processed_frames]
        field_id_map = {(f.field_x, f.field_y): idx + 1 for idx, f in enumerate(sample_processed_frames)}

        detections = extract_all_detections(results_paired, field_id_map, config)
        assert detections == []  # no results → no detections


# ===========================================================================
# Test 4: Aggregator
# ===========================================================================

class TestAggregator:
    """Tests for cap.layer4_inference.aggregator."""

    def test_aggregate_empty_detections(self, config):
        """Empty detection list → all None results (AI-disabled mode)."""
        from cap.layer4_inference.aggregator import aggregate_slide_results

        results = aggregate_slide_results(
            slide_id=1, detections=[], config=config,
        )

        assert results.slide_id == 1
        assert results.organism_counts == {}
        assert results.severity_grades is None
        assert results.overall_severity is None
        assert results.plain_english_summary is None
        assert results.flagged_field_ids == []
        assert results.density_map is None

    def test_aggregate_with_detections(self, config, sample_detections):
        """Detections are correctly counted and graded."""
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.common.dataclasses import SeverityGrade

        results = aggregate_slide_results(
            slide_id=1,
            detections=sample_detections,
            config=config,
            field_grid_size=(3, 3),
            model_version="test_v1",
        )

        # Counts
        assert results.organism_counts["cocci_small"] == 7
        assert results.organism_counts["yeast"] == 2
        assert results.organism_counts["ear_mites"] == 1

        # Severity grades (not None since we have detections)
        assert results.severity_grades is not None
        assert results.severity_grades["cocci_small"] == SeverityGrade.MODERATE  # 7 ≥ 5
        assert results.severity_grades["yeast"] == SeverityGrade.RARE           # 2 ≥ 1
        assert results.severity_grades["ear_mites"] == SeverityGrade.RARE       # 1 ≥ 1

        # Overall = highest = moderate
        assert results.overall_severity == SeverityGrade.MODERATE

        # Summary is a non-empty string
        assert results.plain_english_summary is not None
        assert len(results.plain_english_summary) > 10

        # Ear mites mentioned in summary
        assert "mite" in results.plain_english_summary.lower()

        # Density map shape
        assert results.density_map is not None
        assert results.density_map.shape == (3, 3)

        # Model version
        assert results.model_version == "test_v1"

    def test_flagging_ear_mites(self, config, sample_detections):
        """Fields with ear mites are always flagged."""
        from cap.layer4_inference.aggregator import aggregate_slide_results

        results = aggregate_slide_results(
            slide_id=1, detections=sample_detections, config=config,
        )

        # field_id=2 had the ear mite detection
        assert 2 in results.flagged_field_ids

    def test_aggregate_ensures_all_classes_present(self, config):
        """All configured classes appear in organism_counts, even at 0."""
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.common.dataclasses import Detection

        # Only one class detected
        detections = [Detection(
            field_id=1, class_name="yeast", confidence=0.9,
            bbox=(10, 10, 20, 20), model_version="test",
        )]

        results = aggregate_slide_results(
            slide_id=1, detections=detections, config=config,
        )

        for cls in config.inference.classes:
            assert cls in results.organism_counts


# ===========================================================================
# Test 5: AI-Disabled Mode
# ===========================================================================

class TestAIDisabledMode:
    """Tests for cap.layer4_inference.ai_disabled_mode."""

    def test_get_disabled_results(self):
        """get_disabled_results returns SlideResults with all None fields."""
        from cap.layer4_inference.ai_disabled_mode import get_disabled_results

        results = get_disabled_results(slide_id=42)

        assert results.slide_id == 42
        assert results.organism_counts == {}
        assert results.severity_grades is None
        assert results.overall_severity is None
        assert results.plain_english_summary is None
        assert results.density_map is None
        assert results.model_version == "none"
        assert results.flagged_field_ids == []

    def test_is_ai_available_none(self):
        """is_ai_available(None) returns False."""
        from cap.layer4_inference.ai_disabled_mode import is_ai_available
        assert is_ai_available(None) is False

    def test_is_ai_available_config_disabled(self, config_ai_disabled):
        """is_ai_available(config) returns False when disabled."""
        from cap.layer4_inference.ai_disabled_mode import is_ai_available
        assert is_ai_available(config_ai_disabled) is False

    def test_is_ai_available_mock_model(self):
        """is_ai_available(model_object) returns True for non-None."""
        from cap.layer4_inference.ai_disabled_mode import is_ai_available
        mock_model = MagicMock()
        assert is_ai_available(mock_model) is True


# ===========================================================================
# Test 6: SlideResults Dataclass (Optional fields)
# ===========================================================================

class TestSlideResultsDataclass:
    """Verify the dataclass changes support None values correctly."""

    def test_slide_results_all_none(self):
        """SlideResults can be created with all Optional fields as None."""
        from cap.common.dataclasses import SlideResults

        results = SlideResults(
            slide_id=1,
            organism_counts={},
        )

        assert results.severity_grades is None
        assert results.overall_severity is None
        assert results.plain_english_summary is None
        assert results.density_map is None
        assert results.flagged_field_ids == []
        assert results.model_version == ""

    def test_slide_results_with_values(self):
        """SlideResults works normally when severity fields are populated."""
        from cap.common.dataclasses import SlideResults, SeverityGrade

        results = SlideResults(
            slide_id=1,
            organism_counts={"yeast": 10},
            severity_grades={"yeast": SeverityGrade.MODERATE},
            overall_severity=SeverityGrade.MODERATE,
            flagged_field_ids=[1, 2],
            model_version="v1",
            plain_english_summary="Moderate yeast detected.",
        )

        assert results.severity_grades["yeast"] == SeverityGrade.MODERATE
        assert results.overall_severity == SeverityGrade.MODERATE
        assert results.plain_english_summary == "Moderate yeast detected."

    def test_slide_results_none_check_pattern(self):
        """Demonstrate the Java-like None check pattern works."""
        from cap.common.dataclasses import SlideResults

        results = SlideResults(slide_id=1, organism_counts={})

        # This is the pattern downstream code should use:
        if results.severity_grades is not None:
            pytest.fail("severity_grades should be None")

        if results.overall_severity is not None:
            pytest.fail("overall_severity should be None")

        if results.plain_english_summary is not None:
            pytest.fail("plain_english_summary should be None")


# ===========================================================================
# Test 7: Cross-Layer Integration (Layer 4 ↔ Layer 5 Database)
# ===========================================================================

class TestLayer4Layer5Integration:
    """Test that Layer 4 output writes correctly to Layer 5 database."""

    def test_store_detections_in_db(self, db, config, sample_detections):
        """Detection objects can be stored via crud.insert_detections_batch."""
        from cap.layer5_data import crud

        # Set up prerequisite records
        patient_id = crud.insert_patient(db, name="Buddy", species="canine")
        slide_id = crud.insert_slide(db, patient_id=patient_id)
        field_ids = crud.insert_fields_batch(db, slide_id, [(0, 0), (1, 0), (2, 0)])

        # Convert Detection objects to dicts for batch insert
        detection_dicts = []
        for det in sample_detections:
            detection_dicts.append({
                "field_id": field_ids[det.field_id - 1],  # map 1-indexed to actual IDs
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox_x": det.bbox[0],
                "bbox_y": det.bbox[1],
                "bbox_w": det.bbox[2],
                "bbox_h": det.bbox[3],
                "model_version": det.model_version,
            })

        count = crud.insert_detections_batch(db, detection_dicts)
        assert count == len(sample_detections)

        # Verify we can read them back
        all_dets = crud.get_detections_for_slide(db, slide_id)
        assert len(all_dets) == len(sample_detections)

        # Verify organism counts
        counts = crud.get_organism_counts(db, slide_id)
        assert counts.get("cocci_small", 0) == 7
        assert counts.get("yeast", 0) == 2
        assert counts.get("ear_mites", 0) == 1

    def test_store_results_in_db(self, db, config, sample_detections):
        """SlideResults can be stored via crud.insert_results."""
        from cap.layer5_data import crud
        from cap.layer4_inference.aggregator import aggregate_slide_results

        # Set up prerequisite records
        patient_id = crud.insert_patient(db, name="Mittens", species="feline")
        slide_id = crud.insert_slide(db, patient_id=patient_id)

        # Aggregate
        results = aggregate_slide_results(
            slide_id=slide_id,
            detections=sample_detections,
            config=config,
            model_version="test_v1",
        )

        # Store in DB (convert SeverityGrade enums to their string values)
        severity_grades_serializable = {
            cls: grade.value for cls, grade in results.severity_grades.items()
        } if results.severity_grades else None

        result_id = crud.insert_results(
            db,
            slide_id=slide_id,
            organism_counts=results.organism_counts,
            severity_score=results.overall_severity.value if results.overall_severity else "0",
            severity_grades=severity_grades_serializable,
            flagged_field_ids=results.flagged_field_ids,
            model_version=results.model_version,
            plain_english_summary=results.plain_english_summary,
        )

        assert result_id is not None

        # Read back and verify
        stored = crud.get_results(db, slide_id)
        assert stored is not None
        assert stored["severity_score"] == "2+"  # moderate
        assert stored["organism_counts"]["cocci_small"] == 7
        assert "mite" in stored["plain_english_summary"].lower()

    def test_store_disabled_results_in_db(self, db, config):
        """AI-disabled results (all None) can be stored without error."""
        from cap.layer5_data import crud
        from cap.layer4_inference.ai_disabled_mode import get_disabled_results

        patient_id = crud.insert_patient(db, name="Rex", species="canine")
        slide_id = crud.insert_slide(db, patient_id=patient_id)

        results = get_disabled_results(slide_id)

        # Store — None values should be handled gracefully
        result_id = crud.insert_results(
            db,
            slide_id=slide_id,
            organism_counts=results.organism_counts,
            severity_score=results.overall_severity.value if results.overall_severity else "0",
            severity_grades=None,
            flagged_field_ids=results.flagged_field_ids,
            model_version=results.model_version,
            plain_english_summary=results.plain_english_summary,
        )

        assert result_id is not None

        stored = crud.get_results(db, slide_id)
        assert stored is not None
        assert stored["severity_score"] == "0"
        assert stored["plain_english_summary"] is None


# ===========================================================================
# Test 8: Cross-Layer Integration (Layer 4 ↔ Layer 7 Severity)
# ===========================================================================

class TestLayer4Layer7Integration:
    """Test that Layer 4 aggregator works correctly with Layer 7 severity."""

    def test_severity_thresholds_from_config(self, config):
        """Verify config severity thresholds produce correct grades."""
        from cap.layer7_visualization.severity import compute_severity
        from cap.common.dataclasses import SeverityGrade

        # Use ear_mites thresholds: [1, 2, 5, 10]
        thresholds = config.inference.severity_thresholds["ear_mites"]

        assert compute_severity(0, thresholds) == SeverityGrade.NONE
        assert compute_severity(1, thresholds) == SeverityGrade.RARE
        assert compute_severity(2, thresholds) == SeverityGrade.MODERATE
        assert compute_severity(5, thresholds) == SeverityGrade.MANY
        assert compute_severity(10, thresholds) == SeverityGrade.PACKED

    def test_summary_generation_matches_grades(self, config, sample_detections):
        """The plain-English summary reflects the computed severity grades."""
        from cap.layer4_inference.aggregator import aggregate_slide_results

        results = aggregate_slide_results(
            slide_id=1,
            detections=sample_detections,
            config=config,
        )

        summary = results.plain_english_summary
        assert summary is not None

        # Summary should mention the highest-severity finding
        assert "moderate" in summary.lower() or "cocci" in summary.lower()


# ===========================================================================
# Test 9: End-to-End AI-Disabled Pipeline
# ===========================================================================

class TestEndToEndAIDisabled:
    """
    Simulates the full scan-to-results pipeline in AI-disabled mode.
    No model file, no ultralytics required.
    """

    def test_full_pipeline_no_model(self, db, config, temp_dir):
        """Run the complete pipeline with no model file."""
        from cap.layer5_data import crud
        from cap.layer4_inference.model_loader import load_model
        from cap.layer4_inference.inference import run_inference
        from cap.layer4_inference.postprocess import extract_all_detections
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.common.dataclasses import ProcessedFrame

        # Step 1: Config points to nonexistent model
        config.inference.enabled = True
        config.inference.model_path = os.path.join(temp_dir, "nonexistent.pt")

        # Step 2: Load model (should return None)
        model = load_model(config)
        assert model is None

        # Step 3: Create DB records (simulating what the scan pipeline does)
        patient_id = crud.insert_patient(db, name="Luna", species="feline")
        slide_id = crud.insert_slide(db, patient_id=patient_id)
        field_positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        field_ids = crud.insert_fields_batch(db, slide_id, field_positions)

        # Step 4: Create processed frames (simulating Layer 3 output)
        frames = []
        for (x, y) in field_positions:
            frames.append(ProcessedFrame(
                slide_id=slide_id,
                field_x=x,
                field_y=y,
                rgb_data=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                stacked=True,
                focus_score=0.88,
            ))

        # Step 5: Run inference (model is None → all results None)
        results_paired = run_inference(model, frames, config)
        assert all(r is None for _, r in results_paired)

        # Step 6: Post-process (all None → empty detections)
        field_id_map = {pos: fid for pos, fid in zip(field_positions, field_ids)}
        detections = extract_all_detections(
            results_paired, field_id_map, config,
        )
        assert detections == []

        # Step 7: Aggregate (empty → all None results)
        slide_results = aggregate_slide_results(
            slide_id=slide_id,
            detections=detections,
            config=config,
            field_grid_size=(2, 3),
        )

        assert slide_results.organism_counts == {}
        assert slide_results.severity_grades is None
        assert slide_results.overall_severity is None
        assert slide_results.plain_english_summary is None

        # Step 8: Store in DB
        result_id = crud.insert_results(
            db,
            slide_id=slide_id,
            organism_counts=slide_results.organism_counts,
            severity_score=(
                slide_results.overall_severity.value
                if slide_results.overall_severity
                else "0"
            ),
            severity_grades=None,
            model_version=slide_results.model_version,
            plain_english_summary=slide_results.plain_english_summary,
        )

        # Step 9: Verify stored results
        stored = crud.get_results(db, slide_id)
        assert stored is not None
        assert stored["severity_score"] == "0"
        assert stored["organism_counts"] == {}
        assert stored["plain_english_summary"] is None

        print("\n✓ Full AI-disabled pipeline completed successfully")

    def test_full_pipeline_with_mock_model(self, db, config, temp_dir):
        """
        Run the complete pipeline with a mock model that returns
        detections, verifying the full data flow end-to-end.
        """
        from cap.layer5_data import crud
        from cap.layer4_inference.inference import run_inference
        from cap.layer4_inference.postprocess import extract_all_detections
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.common.dataclasses import ProcessedFrame, SeverityGrade

        # Create DB records
        patient_id = crud.insert_patient(
            db, name="Max", species="canine", owner_name="Smith",
        )
        slide_id = crud.insert_slide(db, patient_id=patient_id)
        field_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        field_ids = crud.insert_fields_batch(db, slide_id, field_positions)

        # Create processed frames
        frames = []
        for x, y in field_positions:
            frames.append(ProcessedFrame(
                slide_id=slide_id, field_x=x, field_y=y,
                rgb_data=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                stacked=True, focus_score=0.9,
            ))

        # Build a mock model that returns fake detections
        mock_model = MagicMock()

        def fake_predict(source, **kwargs):
            """Return 3 cocci_small detections per image."""
            results = []
            for _ in source:
                mock_r = MagicMock()
                mock_r.names = {0: "cocci_small", 1: "yeast", 2: "rods"}
                mock_boxes = MagicMock()
                mock_boxes.__len__ = lambda self: 3
                mock_boxes.conf = [0.92, 0.87, 0.65]
                mock_boxes.cls = [0, 0, 1]  # 2 cocci, 1 yeast
                mock_boxes.xywh = [
                    [100, 100, 20, 20],
                    [200, 200, 25, 25],
                    [300, 300, 15, 15],
                ]
                mock_r.boxes = mock_boxes
                results.append(mock_r)
            return results

        mock_model.predict.side_effect = fake_predict

        # Run inference
        results_paired = run_inference(mock_model, frames, config)
        assert len(results_paired) == 4

        # Post-process
        field_id_map = {pos: fid for pos, fid in zip(field_positions, field_ids)}
        detections = extract_all_detections(
            results_paired, field_id_map, config, model_version="mock_v1",
        )

        # 4 fields × 3 detections each = 12 total
        assert len(detections) == 12
        cocci_count = sum(1 for d in detections if d.class_name == "cocci_small")
        yeast_count = sum(1 for d in detections if d.class_name == "yeast")
        assert cocci_count == 8   # 2 per field × 4 fields
        assert yeast_count == 4   # 1 per field × 4 fields

        # Aggregate
        slide_results = aggregate_slide_results(
            slide_id=slide_id,
            detections=detections,
            config=config,
            field_grid_size=(2, 2),
            model_version="mock_v1",
        )

        assert slide_results.severity_grades is not None
        assert slide_results.overall_severity is not None
        assert slide_results.plain_english_summary is not None

        # 8 cocci_small → 2+ moderate (threshold [1,5,15,30] → 8 ≥ 5)
        assert slide_results.severity_grades["cocci_small"] == SeverityGrade.MODERATE
        # 4 yeast → 1+ rare (threshold [1,5,15,30] → 4 ≥ 1 but < 5)
        assert slide_results.severity_grades["yeast"] == SeverityGrade.RARE

        assert slide_results.overall_severity == SeverityGrade.MODERATE

        # Store in DB
        severity_grades_serializable = {
            cls: grade.value for cls, grade in slide_results.severity_grades.items()
        }
        crud.insert_results(
            db,
            slide_id=slide_id,
            organism_counts=slide_results.organism_counts,
            severity_score=slide_results.overall_severity.value,
            severity_grades=severity_grades_serializable,
            flagged_field_ids=slide_results.flagged_field_ids,
            model_version=slide_results.model_version,
            plain_english_summary=slide_results.plain_english_summary,
        )

        # Store individual detections
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
        crud.insert_detections_batch(db, det_dicts)

        # Read back and verify full round-trip
        stored = crud.get_results(db, slide_id)
        assert stored["severity_score"] == "2+"
        assert stored["organism_counts"]["cocci_small"] == 8

        stored_dets = crud.get_detections_for_slide(db, slide_id)
        assert len(stored_dets) == 12

        counts = crud.get_organism_counts(db, slide_id)
        assert counts["cocci_small"] == 8
        assert counts["yeast"] == 4

        print("\n✓ Full mock-model pipeline completed successfully")


# ===========================================================================
# Test 10: Package Imports
# ===========================================================================

class TestPackageImports:
    """Verify all Layer 4 public API is importable."""

    def test_import_all_from_init(self):
        """All __all__ exports are importable."""
        from cap.layer4_inference import (
            load_model,
            get_model_version,
            run_inference,
            run_single_inference,
            extract_detections,
            extract_all_detections,
            aggregate_slide_results,
            get_disabled_results,
            is_ai_available,
        )

        # Just verify they're callable
        assert callable(load_model)
        assert callable(get_model_version)
        assert callable(run_inference)
        assert callable(run_single_inference)
        assert callable(extract_detections)
        assert callable(extract_all_detections)
        assert callable(aggregate_slide_results)
        assert callable(get_disabled_results)
        assert callable(is_ai_available)

    def test_import_dataclasses(self):
        """Verify modified SlideResults imports correctly."""
        from cap.common.dataclasses import (
            SlideResults, SeverityGrade, Detection,
            ProcessedFrame, FieldStatus, SlideStatus,
        )
        # Quick sanity: SlideResults can be instantiated with minimal args
        sr = SlideResults(slide_id=1, organism_counts={})
        assert sr.severity_grades is None


# ===========================================================================
# Run directly
# ===========================================================================

if __name__ == "__main__":
    # Allow running with: python tests/test_phase7.py
    pytest.main([__file__, "-v", "--tb=short"])
