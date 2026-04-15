"""
CAP Phase 8 Test Suite
=======================
Tests the integration wiring between workers, UI screens,
and backend layers. Runs without a Qt event loop or real
hardware — uses mocks for QThread and signal connections.

Run from the project root:
    python -m pytest tests/test_phase8.py -v

Or run directly:
    python tests/test_phase8.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="cap_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def config():
    from cap.config.config_loader import load_config
    return load_config()


@pytest.fixture
def db(temp_dir):
    from cap.layer5_data.db_manager import DatabaseManager
    db_path = os.path.join(temp_dir, "test_cap.db")
    db = DatabaseManager(db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def app_context(config, db, temp_dir):
    """Build a minimal AppContext with mocked hardware."""
    from cap.app import AppContext

    config.storage.image_root = os.path.join(temp_dir, "slides")
    os.makedirs(config.storage.image_root, exist_ok=True)

    ctx = AppContext(
        config=config,
        backend=MagicMock(),
        db=db,
        motor=MagicMock(),
        safety=MagicMock(),
        oil_monitor=MagicMock(),
        camera=MagicMock(),
        focus=MagicMock(),
        autofocus=MagicMock(),
        ai_model=None,
    )
    return ctx


# ===========================================================================
# Test 1: AppContext
# ===========================================================================

class TestAppContext:
    """Tests for the updated AppContext."""

    def test_all_slots_initialized_to_none(self):
        """All slots start as None if not provided."""
        from cap.app import AppContext
        ctx = AppContext()
        assert ctx.config is None
        assert ctx.current_slide_id is None
        assert ctx.field_id_map is None
        assert ctx.last_pipeline_result is None
        assert ctx.last_slide_results is None
        assert ctx.ai_model is None

    def test_kwargs_set_values(self, config, db):
        """Provided kwargs are set correctly."""
        from cap.app import AppContext
        ctx = AppContext(config=config, db=db, ai_model="fake_model")
        assert ctx.config is config
        assert ctx.db is db
        assert ctx.ai_model == "fake_model"

    def test_reset_workflow_state(self, app_context):
        """reset_workflow_state clears all dynamic attributes."""
        app_context.current_slide_id = 42
        app_context.field_id_map = {(0, 0): 1}
        app_context.last_slide_results = MagicMock()

        app_context.reset_workflow_state()

        assert app_context.current_slide_id is None
        assert app_context.field_id_map is None
        assert app_context.last_slide_results is None
        # Static attributes should be untouched
        assert app_context.config is not None
        assert app_context.db is not None

    def test_shutdown_handles_none_components(self):
        """shutdown() doesn't crash when components are None."""
        from cap.app import AppContext
        ctx = AppContext()
        ctx.shutdown()  # should not raise


# ===========================================================================
# Test 2: ScanWorker
# ===========================================================================

class TestScanWorker:
    """Tests for the ScanWorker thread."""

    def test_scan_worker_creation(self, app_context):
        """ScanWorker can be instantiated with region vertices."""
        from cap.workers.scan_worker import ScanWorker

        vertices = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        worker = ScanWorker(app_context, vertices)

        assert worker.scan_signals is not None
        assert worker.focus_signals is not None

    def test_scan_worker_has_control_methods(self, app_context):
        """ScanWorker exposes pause/resume/stop control methods."""
        from cap.workers.scan_worker import ScanWorker

        worker = ScanWorker(app_context, [(0, 0)])
        assert callable(worker.request_pause)
        assert callable(worker.request_resume)
        assert callable(worker.request_stop)

    def test_scan_worker_stop_before_start(self, app_context):
        """request_stop before run doesn't crash."""
        from cap.workers.scan_worker import ScanWorker

        worker = ScanWorker(app_context, [(0, 0)])
        worker.request_stop()  # no pipeline yet, should be safe


# ===========================================================================
# Test 3: InferenceWorker
# ===========================================================================

class TestInferenceWorker:
    """Tests for the InferenceWorker thread."""

    def test_inference_worker_creation(self, app_context):
        """InferenceWorker can be instantiated."""
        from cap.workers.inference_worker import InferenceWorker

        worker = InferenceWorker(app_context)
        assert worker.signals is not None

    def test_inference_worker_ai_disabled_run(self, app_context):
        """InferenceWorker emits inference_skipped when AI is disabled."""
        from cap.workers.inference_worker import InferenceWorker
        from cap.layer5_data import crud

        # Set up DB records
        patient_id = crud.insert_patient(app_context.db, name="Test", species="canine")
        slide_id = crud.insert_slide(app_context.db, patient_id=patient_id)
        app_context.current_slide_id = slide_id
        app_context.config.inference.enabled = False

        worker = InferenceWorker(app_context)

        # Track emitted signals
        skipped_reasons = []
        worker.signals.inference_skipped.connect(lambda r: skipped_reasons.append(r))

        # Run directly (not in a thread, for testing)
        worker.run()

        assert len(skipped_reasons) == 1
        assert "disabled" in skipped_reasons[0].lower()

        # Verify results stored in DB
        results = crud.get_results(app_context.db, slide_id)
        assert results is not None
        assert results["severity_score"] == "0"


# ===========================================================================
# Test 4: ReportWorker
# ===========================================================================

class TestReportWorker:
    """Tests for the ReportWorker thread."""

    def test_report_worker_creation(self, app_context):
        """ReportWorker can be instantiated."""
        from cap.workers.report_worker import ReportWorker

        worker = ReportWorker(app_context, technician_notes="Test notes")
        assert worker is not None


# ===========================================================================
# Test 5: Signal wiring patterns
# ===========================================================================

class TestSignalWiring:
    """Test that signal objects are properly structured."""

    def test_scan_signals_exist(self):
        """All required scan signals are defined."""
        from cap.layer6_ui.signals import ScanSignals

        signals = ScanSignals()
        assert hasattr(signals, "scan_start_requested")
        assert hasattr(signals, "scan_pause_requested")
        assert hasattr(signals, "scan_resume_requested")
        assert hasattr(signals, "scan_stop_requested")
        assert hasattr(signals, "progress")
        assert hasattr(signals, "field_captured")
        assert hasattr(signals, "field_stacked")
        assert hasattr(signals, "scan_complete")
        assert hasattr(signals, "scan_failed")

    def test_inference_signals_exist(self):
        """All required inference signals are defined."""
        from cap.layer6_ui.signals import InferenceSignals

        signals = InferenceSignals()
        assert hasattr(signals, "inference_started")
        assert hasattr(signals, "inference_progress")
        assert hasattr(signals, "inference_complete")
        assert hasattr(signals, "inference_failed")
        assert hasattr(signals, "inference_skipped")


# ===========================================================================
# Test 6: Worker → DB integration
# ===========================================================================

class TestWorkerDBIntegration:
    """Test that workers interact with the database correctly."""

    def test_inference_worker_stores_disabled_results(self, app_context):
        """InferenceWorker stores None results in DB when AI is disabled."""
        from cap.workers.inference_worker import InferenceWorker
        from cap.layer5_data import crud

        patient_id = crud.insert_patient(app_context.db, name="Luna", species="feline")
        slide_id = crud.insert_slide(app_context.db, patient_id=patient_id)
        app_context.current_slide_id = slide_id
        app_context.config.inference.enabled = False

        worker = InferenceWorker(app_context)
        worker.run()

        # Verify DB state
        results = crud.get_results(app_context.db, slide_id)
        assert results is not None
        assert results["severity_score"] == "0"
        assert results["plain_english_summary"] is None
        assert results["model_version"] == "none"

        # Slide status should be 'complete'
        slide = crud.get_slide(app_context.db, slide_id)
        assert slide["status"] == "complete"

    def test_inference_worker_with_mock_model(self, app_context):
        """InferenceWorker stores detections and results with a mock model."""
        from cap.workers.inference_worker import InferenceWorker
        from cap.layer5_data import crud
        from cap.common.dataclasses import StackedField

        # Create DB records
        patient_id = crud.insert_patient(app_context.db, name="Max", species="canine")
        slide_id = crud.insert_slide(app_context.db, patient_id=patient_id)
        app_context.current_slide_id = slide_id
        app_context.config.inference.enabled = True
        app_context.config.inference.model_path = "/fake/model.pt"

        # Create fake pipeline result with stacked fields
        stacked_fields = []
        field_positions = [(0, 0), (1, 0), (0, 1)]
        field_ids = crud.insert_fields_batch(app_context.db, slide_id, field_positions)
        app_context.field_id_map = {
            pos: fid for pos, fid in zip(field_positions, field_ids)
        }

        for x, y in field_positions:
            sf = StackedField(
                slide_id=slide_id,
                field_x=x,
                field_y=y,
                composite=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                sharpness_map=np.ones((40, 40), dtype=np.float32) * 0.9,
                z_distribution={0: 500, 1: 300, 2: 200},
                stacking_duration_ms=45.0,
                registration_shifts=[(0.1, 0.2), (0.3, -0.1)],
                block_size=16,
            )
            stacked_fields.append(sf)

        app_context.last_pipeline_result = {
            "fields_completed": 3,
            "stacked_fields": stacked_fields,
            "slide_dir": os.path.join(app_context.config.storage.image_root, str(slide_id)),
        }

        # Mock load_model to return a fake model
        def fake_predict(source, **kwargs):
            results = []
            for _ in source:
                r = MagicMock()
                r.names = {0: "cocci_small", 1: "yeast"}
                boxes = MagicMock()
                boxes.__len__ = lambda self: 2
                boxes.conf = [0.9, 0.8]
                boxes.cls = [0, 1]
                boxes.xywh = [[100, 100, 20, 20], [200, 200, 15, 15]]
                r.boxes = boxes
                results.append(r)
            return results

        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict
        mock_model.names = {0: "cocci_small", 1: "yeast"}
        mock_model.ckpt_path = "/fake/model.pt"

        # Patch load_model to return our mock
        with patch("cap.layer4_inference.model_loader.load_model", return_value=mock_model):
            with patch("cap.layer4_inference.model_loader.get_model_version", return_value="mock_v1"):
                worker = InferenceWorker(app_context)

                completed = []
                worker.signals.inference_complete.connect(
                    lambda sid, res: completed.append((sid, res))
                )

                worker.run()

        # Verify inference completed
        assert len(completed) == 1
        result_slide_id, slide_results = completed[0]
        assert result_slide_id == slide_id
        assert slide_results.severity_grades is not None

        # Verify detections in DB
        detections = crud.get_detections_for_slide(app_context.db, slide_id)
        assert len(detections) == 6  # 2 per field × 3 fields

        # Verify results in DB
        db_results = crud.get_results(app_context.db, slide_id)
        assert db_results is not None
        assert db_results["organism_counts"]["cocci_small"] == 3
        assert db_results["organism_counts"]["yeast"] == 3

        # Verify context updated
        assert app_context.last_slide_results is not None

        print("\n✓ InferenceWorker mock model test passed")


# ===========================================================================
# Test 7: Full workflow sequence (no Qt)
# ===========================================================================

class TestFullWorkflowSequence:
    """
    Test the complete workflow state progression through AppContext,
    simulating the sequence: session → region → scan → inference → results.
    """

    def test_workflow_state_progression(self, app_context):
        """Simulate the full state flow through AppContext."""
        from cap.layer5_data import crud

        ctx = app_context

        # Step 1: Session start (simulates SessionStartScreen._on_proceed)
        tech_id = crud.insert_technician(ctx.db, name="Test Tech", login="test")
        session_id = crud.start_session(ctx.db, tech_id)
        patient_id = crud.insert_patient(ctx.db, name="Buddy", species="canine")
        slide_id = crud.insert_slide(ctx.db, patient_id=patient_id, session_id=session_id)

        ctx.current_session_id = session_id
        ctx.current_patient_id = patient_id
        ctx.current_slide_id = slide_id
        ctx.current_tech_id = tech_id

        assert ctx.current_slide_id == slide_id

        # Step 2: Scan region (simulates ScanRegionScreen._on_confirm)
        motor_vertices = [(0, 0), (10000, 0), (10000, 5000), (0, 5000)]
        ctx.current_scan_region_vertices = motor_vertices

        assert len(ctx.current_scan_region_vertices) == 4

        # Step 3: After scan (simulates ScanWorker completion)
        field_positions = [(0, 0), (1, 0), (2, 0)]
        field_ids = crud.insert_fields_batch(ctx.db, slide_id, field_positions)
        ctx.field_id_map = {pos: fid for pos, fid in zip(field_positions, field_ids)}
        ctx.last_pipeline_result = {
            "fields_completed": 3,
            "stacked_fields": [],
            "slide_dir": "/fake/dir",
        }

        assert len(ctx.field_id_map) == 3

        # Step 4: After inference (simulates InferenceWorker)
        from cap.layer4_inference.ai_disabled_mode import get_disabled_results

        ctx.last_slide_results = get_disabled_results(slide_id)

        assert ctx.last_slide_results is not None
        assert ctx.last_slide_results.severity_grades is None

        # Step 5: Reset for new scan
        ctx.reset_workflow_state()

        assert ctx.current_slide_id is None
        assert ctx.field_id_map is None
        assert ctx.last_slide_results is None
        # Config and DB still intact
        assert ctx.config is not None
        assert ctx.db is not None

        print("\n✓ Full workflow state progression test passed")


# ===========================================================================
# Test 8: Package imports
# ===========================================================================

class TestPhase8Imports:
    """Verify all Phase 8 modules are importable."""

    def test_import_workers(self):
        from cap.workers.scan_worker import ScanWorker
        from cap.workers.inference_worker import InferenceWorker
        from cap.workers.report_worker import ReportWorker
        assert callable(ScanWorker)
        assert callable(InferenceWorker)
        assert callable(ReportWorker)

    def test_import_app_context(self):
        from cap.app import AppContext
        ctx = AppContext()
        assert ctx is not None


# ===========================================================================
# Run directly
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
