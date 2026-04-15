"""
CAP Phase 9 Test Suite
=======================
Tests Layer 8 (Metrics) and Layer 9 (Retraining) modules.

Run from the project root:
    python -m pytest tests/test_phase9.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

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
def populated_db(db):
    """A database with realistic test data for metrics."""
    from cap.layer5_data import crud

    # Technicians
    tech1 = crud.insert_technician(db, name="Alice", login="alice")
    tech2 = crud.insert_technician(db, name="Bob", login="bob")

    # Patients
    p1 = crud.insert_patient(db, name="Buddy", species="canine", owner_name="Smith")
    p2 = crud.insert_patient(db, name="Whiskers", species="feline", owner_name="Jones")
    p3 = crud.insert_patient(db, name="Rex", species="canine", owner_name="Brown")

    # Slides with results
    slides = []
    for i, (patient, tech) in enumerate([
        (p1, tech1), (p2, tech1), (p3, tech2), (p1, tech2),
    ]):
        sid = crud.insert_slide(db, patient_id=patient, technician_id=tech)
        crud.update_slide_status(db, sid, "complete")

        # Fields
        field_ids = crud.insert_fields_batch(db, sid, [(0, 0), (1, 0), (0, 1)])

        # Detections
        det_dicts = []
        for fid in field_ids:
            det_dicts.append({
                "field_id": fid, "class_name": "cocci_small",
                "confidence": 0.85 + i * 0.03, "bbox_x": 100, "bbox_y": 100,
                "bbox_w": 20, "bbox_h": 20, "model_version": "v1",
            })
            det_dicts.append({
                "field_id": fid, "class_name": "yeast",
                "confidence": 0.72 + i * 0.02, "bbox_x": 200, "bbox_y": 200,
                "bbox_w": 15, "bbox_h": 15, "model_version": "v1",
            })
        crud.insert_detections_batch(db, det_dicts)

        # Results
        severity = ["0", "1+", "2+", "1+"][i]
        crud.insert_results(
            db, slide_id=sid,
            organism_counts={"cocci_small": 3, "yeast": 3},
            severity_score=severity,
            severity_grades={"cocci_small": severity, "yeast": "1+"},
            model_version="v1",
            plain_english_summary=f"Test summary for slide {sid}",
        )

        slides.append(sid)

    # Corrections (Alice corrects some detections)
    all_dets = db.fetchall("SELECT detection_id, class FROM detections LIMIT 5")
    for i, det in enumerate(all_dets):
        if i < 3:
            # Class change
            crud.insert_correction(
                db, detection_id=det["detection_id"], tech_id=tech1,
                original_class=det["class"], corrected_class="rods",
            )
        elif i == 3:
            # False positive
            crud.insert_correction(
                db, detection_id=det["detection_id"], tech_id=tech1,
                original_class=det["class"], corrected_class="false_positive",
            )
        else:
            # Reviewed correction
            corr_id = crud.insert_correction(
                db, detection_id=det["detection_id"], tech_id=tech2,
                original_class=det["class"], corrected_class="cocci_large",
            )
            crud.mark_correction_reviewed(db, corr_id, reviewer_notes="Looks correct")

    return {"db": db, "slides": slides, "techs": [tech1, tech2]}


# ===========================================================================
# Test 1: Clinic Dashboard
# ===========================================================================

class TestClinicDashboard:

    def test_get_overview(self, populated_db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dashboard = ClinicDashboard(populated_db["db"])
        overview = dashboard.get_overview()

        assert overview["total_scans"] == 4
        assert overview["total_patients"] == 3
        assert overview["total_detections"] == 24  # 4 slides × 3 fields × 2 dets
        assert overview["total_corrections"] == 5
        assert overview["scans_with_findings"] == 3  # slides with severity != '0'
        assert overview["scans_clean"] == 1

    def test_severity_distribution(self, populated_db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dashboard = ClinicDashboard(populated_db["db"])
        dist = dashboard.get_severity_distribution()

        assert "0" in dist
        assert "1+" in dist
        assert "2+" in dist
        assert dist["0"] == 1
        assert dist["1+"] == 2
        assert dist["2+"] == 1

    def test_organism_frequency(self, populated_db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dashboard = ClinicDashboard(populated_db["db"])
        freq = dashboard.get_organism_frequency()

        assert "cocci_small" in freq
        assert "yeast" in freq
        assert freq["cocci_small"] == 12  # 4 slides × 3 fields
        assert freq["yeast"] == 12

    def test_technician_stats(self, populated_db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dashboard = ClinicDashboard(populated_db["db"])
        stats = dashboard.get_technician_stats()

        assert len(stats) == 2
        names = {s["name"] for s in stats}
        assert "Alice" in names
        assert "Bob" in names

    def test_species_breakdown(self, populated_db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dashboard = ClinicDashboard(populated_db["db"])
        species = dashboard.get_species_breakdown()

        assert "canine" in species
        assert "feline" in species
        assert species["canine"] == 3  # 3 canine slides
        assert species["feline"] == 1

    def test_empty_db(self, db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dashboard = ClinicDashboard(db)
        overview = dashboard.get_overview()
        assert overview["total_scans"] == 0
        assert overview["total_detections"] == 0


# ===========================================================================
# Test 2: AI Metrics
# ===========================================================================

class TestAIMetrics:

    def test_accuracy_summary(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        metrics = AIMetrics(populated_db["db"])
        summary = metrics.get_accuracy_summary()

        assert summary["total_detections"] == 24
        assert summary["total_corrections"] == 5
        assert summary["false_positive_count"] == 1
        assert summary["class_change_count"] == 4
        assert 0 < summary["estimated_accuracy"] < 1
        assert abs(summary["correction_rate"] - 5 / 24) < 0.001

    def test_per_class_accuracy(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        metrics = AIMetrics(populated_db["db"])
        per_class = metrics.get_per_class_accuracy()

        class_names = {c["class_name"] for c in per_class}
        assert "cocci_small" in class_names
        assert "yeast" in class_names

    def test_confusion_matrix(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        metrics = AIMetrics(populated_db["db"])
        confusion = metrics.get_confusion_matrix()

        assert confusion["total_corrections"] == 5
        assert "rods" in confusion["classes"]
        assert "false_positive" in confusion["classes"]
        assert len(confusion["matrix"]) > 0

    def test_confidence_analysis(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        metrics = AIMetrics(populated_db["db"])
        analysis = metrics.get_confidence_analysis()

        assert analysis["avg_confidence"] > 0
        assert len(analysis["confidence_buckets"]) == 5

    def test_model_version_comparison(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        metrics = AIMetrics(populated_db["db"])
        versions = metrics.get_model_version_comparison()

        assert len(versions) >= 1
        assert versions[0]["model_version"] == "v1"
        assert versions[0]["total_detections"] == 24

    def test_empty_db(self, db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        metrics = AIMetrics(db)
        summary = metrics.get_accuracy_summary()
        assert summary["total_detections"] == 0
        assert summary["estimated_accuracy"] == 1.0


# ===========================================================================
# Test 3: Metrics Export
# ===========================================================================

class TestMetricsExport:

    def test_generate_summary(self, populated_db):
        from cap.layer8_metrics.export import MetricsExporter
        exporter = MetricsExporter(populated_db["db"])
        summary = exporter.generate_summary()

        assert "generated_at" in summary
        assert "clinic" in summary
        assert "ai" in summary
        assert summary["clinic"]["overview"]["total_scans"] == 4
        assert summary["ai"]["accuracy_summary"]["total_corrections"] == 5

    def test_export_csv(self, populated_db, temp_dir):
        from cap.layer8_metrics.export import MetricsExporter
        exporter = MetricsExporter(populated_db["db"])
        output_dir = os.path.join(temp_dir, "metrics_csv")
        paths = exporter.export_csv(output_dir)

        assert len(paths) == 7
        for path in paths:
            assert os.path.isfile(path)
            assert path.endswith(".csv")

    def test_export_json(self, populated_db, temp_dir):
        from cap.layer8_metrics.export import MetricsExporter
        exporter = MetricsExporter(populated_db["db"])
        json_path = os.path.join(temp_dir, "metrics.json")
        exporter.export_json(json_path)

        assert os.path.isfile(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data["clinic"]["overview"]["total_scans"] == 4


# ===========================================================================
# Test 4: Correction Manager
# ===========================================================================

class TestCorrectionManager:

    def test_get_pending_review(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])
        pending = mgr.get_pending_review()

        # 4 unreviewed out of 5 total (1 was marked reviewed in fixture)
        assert len(pending) == 4

    def test_get_correction_stats(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])
        stats = mgr.get_correction_stats()

        assert stats["total"] == 5
        assert stats["reviewed"] == 1
        assert stats["pending"] == 4
        assert stats["false_positive_count"] == 1
        assert len(stats["most_confused_pairs"]) > 0

    def test_prepare_retraining_batch_not_enough(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])

        # Need 50 corrections, only have 5
        batch = mgr.prepare_retraining_batch(min_corrections=50)
        assert batch is None

    def test_prepare_retraining_batch_enough(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])

        # Lower threshold to trigger
        batch = mgr.prepare_retraining_batch(min_corrections=3, only_reviewed=False)
        assert batch is not None
        assert batch["correction_count"] >= 3
        assert len(batch["slide_ids"]) > 0

    def test_get_corrected_annotations(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])

        slide_id = populated_db["slides"][0]
        corrected = mgr.get_corrected_annotations(slide_id)

        # False positives should be excluded
        for det in corrected:
            assert det["class"] != "false_positive"

        # Some should be marked as corrected
        corrected_flags = [d.get("was_corrected", False) for d in corrected]
        assert any(corrected_flags) or len(corrected) > 0

    def test_mark_batch_reviewed(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])

        count = mgr.mark_batch_reviewed(reviewer_notes="Batch approved")
        assert count == 4  # 4 were pending

        # Now all should be reviewed
        pending = mgr.get_pending_review()
        assert len(pending) == 0


# ===========================================================================
# Test 5: CVAT Exporter
# ===========================================================================

class TestCVATExporter:

    def test_export_cvat_raw(self, populated_db, temp_dir):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exporter = CVATExporter(populated_db["db"])

        slide_id = populated_db["slides"][0]
        output_path = os.path.join(temp_dir, "raw.xml")
        exporter.export_cvat(slide_id, output_path, apply_corrections=False)

        assert os.path.isfile(output_path)
        with open(output_path) as f:
            content = f.read()
        assert "<annotations>" in content
        assert "box" in content

    def test_export_cvat_corrected(self, populated_db, temp_dir):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exporter = CVATExporter(populated_db["db"])

        slide_id = populated_db["slides"][0]
        output_path = os.path.join(temp_dir, "corrected.xml")
        exporter.export_cvat(slide_id, output_path, apply_corrections=True)

        assert os.path.isfile(output_path)

    def test_export_yolo_format(self, populated_db, temp_dir, config):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exporter = CVATExporter(populated_db["db"], config)

        slide_id = populated_db["slides"][0]
        output_dir = os.path.join(temp_dir, "yolo_single")
        exporter.export_yolo_format(slide_id, output_dir)

        assert os.path.isdir(os.path.join(output_dir, "labels"))
        assert os.path.isfile(os.path.join(output_dir, "classes.txt"))

        # Check classes.txt has correct content
        with open(os.path.join(output_dir, "classes.txt")) as f:
            classes = f.read().strip().split("\n")
        assert "cocci_small" in classes
        assert "yeast" in classes

    def test_export_yolo_batch(self, populated_db, temp_dir, config):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exporter = CVATExporter(populated_db["db"], config)

        slide_ids = populated_db["slides"][:2]
        output_dir = os.path.join(temp_dir, "yolo_batch")
        exporter.export_yolo_batch(slide_ids, output_dir)

        assert os.path.isfile(os.path.join(output_dir, "dataset.yaml"))
        assert os.path.isfile(os.path.join(output_dir, "classes.txt"))
        assert os.path.isdir(os.path.join(output_dir, "labels"))

        # Check dataset.yaml
        with open(os.path.join(output_dir, "dataset.yaml")) as f:
            content = f.read()
        assert "names:" in content
        assert "cocci_small" in content


# ===========================================================================
# Test 6: Package imports
# ===========================================================================

class TestPhase9Imports:

    def test_import_layer8(self):
        from cap.layer8_metrics import ClinicDashboard, AIMetrics, MetricsExporter
        assert callable(ClinicDashboard)
        assert callable(AIMetrics)
        assert callable(MetricsExporter)

    def test_import_layer9(self):
        from cap.layer9_retraining import CorrectionManager, CVATExporter
        assert callable(CorrectionManager)
        assert callable(CVATExporter)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
