"""
CAP Exhaustive System Test
============================
Tests the entire CAP system across all 9 layers plus configuration,
dataclasses, simulation backends, and end-to-end workflows.

Designed to run on Windows in simulation mode with no hardware
and no trained AI model.

Run from the project root:
    python -m pytest tests/test_system.py -v

Or run directly:
    python tests/test_system.py

Sections:
    1.  Configuration & Dataclasses
    2.  Layer 1: Hardware Abstraction (sim)
    3.  Layer 2: Image Acquisition (sim)
    4.  Layer 3: Image Processing
    5.  Layer 4: AI Inference
    6.  Layer 5: Data Layer
    7.  Layer 6: UI Signals
    8.  Layer 7: Client Visualization
    9.  Layer 8: Metrics
    10. Layer 9: Retraining
    11. Workers
    12. Cross-layer integration
    13. End-to-end pipeline simulation
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import Counter

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
    d = tempfile.mkdtemp(prefix="cap_sys_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def config():
    from cap.config.config_loader import load_config
    return load_config()


@pytest.fixture
def db(temp_dir):
    from cap.layer5_data.db_manager import DatabaseManager
    db_path = os.path.join(temp_dir, "test.db")
    db = DatabaseManager(db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def populated_db(db):
    """Database with full realistic test data spanning all tables."""
    from cap.layer5_data import crud

    tech1 = crud.insert_technician(db, name="Alice", login="alice")
    tech2 = crud.insert_technician(db, name="Bob", login="bob")
    p1 = crud.insert_patient(db, name="Buddy", species="canine", owner_name="Smith")
    p2 = crud.insert_patient(db, name="Whiskers", species="feline", owner_name="Jones")
    p3 = crud.insert_patient(db, name="Rex", species="canine", owner_name="Brown")
    session1 = crud.start_session(db, tech1)

    slides = []
    all_field_ids = []
    for i, (patient, tech) in enumerate([
        (p1, tech1), (p2, tech1), (p3, tech2), (p1, tech2),
    ]):
        sid = crud.insert_slide(db, patient_id=patient, session_id=session1, technician_id=tech)
        crud.update_slide_status(db, sid, "complete")
        fids = crud.insert_fields_batch(db, sid, [(0, 0), (1, 0), (0, 1), (1, 1)])
        all_field_ids.extend(fids)

        dets = []
        for fid in fids:
            dets.append({"field_id": fid, "class_name": "cocci_small", "confidence": 0.88,
                         "bbox_x": 100, "bbox_y": 100, "bbox_w": 20, "bbox_h": 20, "model_version": "v1"})
            dets.append({"field_id": fid, "class_name": "yeast", "confidence": 0.75,
                         "bbox_x": 300, "bbox_y": 200, "bbox_w": 15, "bbox_h": 15, "model_version": "v1"})
            if i == 2:  # slide 3 gets ear mites
                dets.append({"field_id": fid, "class_name": "ear_mites", "confidence": 0.92,
                             "bbox_x": 400, "bbox_y": 300, "bbox_w": 25, "bbox_h": 25, "model_version": "v1"})
        crud.insert_detections_batch(db, dets)

        sev = ["0", "1+", "3+", "2+"][i]
        crud.insert_results(db, slide_id=sid, organism_counts={"cocci_small": 4, "yeast": 4},
                            severity_score=sev, severity_grades={"cocci_small": sev, "yeast": "1+"},
                            model_version="v1", plain_english_summary=f"Test summary slide {sid}")
        slides.append(sid)

    # Corrections
    first_dets = db.fetchall("SELECT detection_id, class FROM detections LIMIT 6")
    for i, det in enumerate(first_dets):
        if i < 3:
            crud.insert_correction(db, det["detection_id"], tech1, det["class"], "rods")
        elif i == 3:
            crud.insert_correction(db, det["detection_id"], tech1, det["class"], "false_positive")
        else:
            cid = crud.insert_correction(db, det["detection_id"], tech2, det["class"], "cocci_large")
            crud.mark_correction_reviewed(db, cid, "Approved")

    crud.end_session(db, session1)

    return {
        "db": db, "slides": slides, "techs": [tech1, tech2],
        "patients": [p1, p2, p3], "session": session1,
        "field_ids": all_field_ids,
    }


# ###########################################################################
# 1. CONFIGURATION & DATACLASSES
# ###########################################################################

class TestConfiguration:

    def test_load_config_defaults(self, config):
        assert config.hardware_mode == "simulation"
        assert config.inference.enabled is True
        assert config.focus.z_depths_per_field == 6
        assert config.scan.fields_per_second == 2
        assert config.motor.microsteps == 256

    def test_config_inference_section(self, config):
        assert config.inference.confidence_threshold == 0.5
        assert config.inference.nms_iou_threshold == 0.45
        assert config.inference.batch_size == 16
        assert "cocci_small" in config.inference.classes
        assert "ear_mites" in config.inference.classes
        assert len(config.inference.classes) == 6

    def test_config_severity_thresholds(self, config):
        thresholds = config.inference.severity_thresholds
        assert "default" in thresholds
        assert "ear_mites" in thresholds
        assert thresholds["ear_mites"] == [1, 2, 5, 10]
        assert thresholds["rods"] == [1, 3, 10, 20]

    def test_config_to_dict_roundtrip(self, config):
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["hardware_mode"] == "simulation"
        assert d["inference"]["enabled"] is True

    def test_config_visualization_colors(self, config):
        colors = config.visualization.annotation_colors
        assert "cocci_small" in colors
        assert colors["yeast"].startswith("#")


class TestDataclasses:

    def test_all_dataclasses_importable(self):
        from cap.common.dataclasses import (
            FieldStatus, SlideStatus, SeverityGrade,
            FocusMapResult, RawFrame, ProcessedFrame, StackedField,
            Detection, SlideResults, ScanProgress, ScanRegion,
        )

    def test_severity_grade_values(self):
        from cap.common.dataclasses import SeverityGrade
        assert SeverityGrade.NONE.value == "0"
        assert SeverityGrade.RARE.value == "1+"
        assert SeverityGrade.MODERATE.value == "2+"
        assert SeverityGrade.MANY.value == "3+"
        assert SeverityGrade.PACKED.value == "4+"

    def test_slide_results_optional_fields(self):
        from cap.common.dataclasses import SlideResults
        sr = SlideResults(slide_id=1, organism_counts={})
        assert sr.severity_grades is None
        assert sr.overall_severity is None
        assert sr.plain_english_summary is None
        assert sr.flagged_field_ids == []

    def test_detection_creation(self):
        from cap.common.dataclasses import Detection
        d = Detection(field_id=1, class_name="yeast", confidence=0.95,
                      bbox=(10.0, 20.0, 30.0, 40.0), model_version="v1")
        assert d.class_name == "yeast"
        assert d.bbox == (10.0, 20.0, 30.0, 40.0)

    def test_scan_region_defaults(self):
        from cap.common.dataclasses import ScanRegion
        region = ScanRegion(polygon_vertices=[(0, 0), (100, 0), (100, 100)])
        assert region.field_count == 0
        assert region.field_positions == []

    def test_processed_frame(self):
        from cap.common.dataclasses import ProcessedFrame
        pf = ProcessedFrame(slide_id=1, field_x=0, field_y=0,
                            rgb_data=np.zeros((640, 640, 3), dtype=np.uint8),
                            stacked=True, focus_score=0.9)
        assert pf.stacked is True
        assert pf.rgb_data.shape == (640, 640, 3)


# ###########################################################################
# 2. LAYER 1: HARDWARE ABSTRACTION (simulation)
# ###########################################################################

class TestLayer1CoordinateMapper:

    def test_mm_to_motor_roundtrip(self, config):
        from cap.layer1_hardware.coordinate_mapper import CoordinateMapper
        mapper = CoordinateMapper(config)
        mx, my = mapper.mm_to_motor(10.0, 5.0)
        rx, ry = mapper.motor_to_mm(mx, my)
        assert abs(rx - 10.0) < 0.01
        assert abs(ry - 5.0) < 0.01

    def test_fractional_to_motor(self, config):
        from cap.layer1_hardware.coordinate_mapper import CoordinateMapper
        mapper = CoordinateMapper(config)
        mx, my = mapper.fractional_to_motor(0.5, 0.5)
        assert mx > 0
        assert my > 0

    def test_fractional_polygon_conversion(self, config):
        from cap.layer1_hardware.coordinate_mapper import CoordinateMapper
        mapper = CoordinateMapper(config)
        frac_poly = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        motor_poly = mapper.fractional_polygon_to_motor(frac_poly)
        assert len(motor_poly) == 4
        back = mapper.motor_polygon_to_fractional(motor_poly)
        assert len(back) == 4
        assert abs(back[0][0] - 0.1) < 0.01


class TestLayer1ScanRegion:

    def test_set_polygon_generates_fields(self, config):
        from cap.layer1_hardware.scan_region import ScanRegionManager
        mgr = ScanRegionManager(config)
        # Use small polygon scaled to FOV: ~10x10 fields at most
        fw = mgr.field_width_steps
        fh = mgr.field_height_steps
        region = mgr.set_polygon([
            (0, 0), (fw * 10, 0), (fw * 10, fh * 10), (0, fh * 10),
        ])
        assert region.field_count > 0
        assert len(region.field_positions) == region.field_count

    def test_set_preset_full_slide(self, config):
        from cap.layer1_hardware.scan_region import ScanRegionManager
        mgr = ScanRegionManager(config)
        region = mgr.set_preset("full_slide")
        assert region.field_count > 0

    def test_estimates(self, config):
        from cap.layer1_hardware.scan_region import ScanRegionManager
        mgr = ScanRegionManager(config)
        fw = mgr.field_width_steps
        fh = mgr.field_height_steps
        mgr.set_polygon([(0, 0), (fw * 5, 0), (fw * 5, fh * 5), (0, fh * 5)])
        est = mgr.get_estimates()
        assert "field_count" in est
        assert "scan_time_sec" in est
        assert "disk_mb" in est

    def test_to_json(self, config):
        from cap.layer1_hardware.scan_region import ScanRegionManager
        mgr = ScanRegionManager(config)
        fw = mgr.field_width_steps
        fh = mgr.field_height_steps
        mgr.set_polygon([(0, 0), (fw * 5, 0), (fw * 5, fh * 5), (0, fh * 5)])
        j = mgr.to_json()
        data = json.loads(j)
        assert "polygon_vertices" in data


class TestLayer1SimMotor:

    def test_motor_movement(self, config):
        from cap.layer1_hardware.sim.sim_motor import SimMotorController
        motor = SimMotorController(config)
        motor.move_to("x", 5000)
        motor.move_to("y", 3000)
        assert motor.get_position("x") == 5000
        assert motor.get_position("y") == 3000

    def test_motor_homing(self, config):
        from cap.layer1_hardware.sim.sim_motor import SimMotorController
        motor = SimMotorController(config)
        motor.move_to("x", 5000)
        motor.home_all()
        assert motor.is_homed  # property, not method call

    def test_emergency_stop(self, config):
        from cap.layer1_hardware.sim.sim_motor import SimMotorController
        motor = SimMotorController(config)
        motor.emergency_stop()
        assert motor.is_emergency_stopped  # property, not method call
        motor.clear_estop()
        assert not motor.is_emergency_stopped


class TestLayer1OilSafety:

    def test_brightness_check(self, config):
        from cap.layer1_hardware.oil_safety import OilSafetyMonitor
        monitor = OilSafetyMonitor(config)
        bright_frame = np.ones((100, 100), dtype=np.uint8) * 200
        result = monitor.check_frame_brightness(bright_frame)
        assert isinstance(result, bool)

    def test_brightness_stats(self, config):
        from cap.layer1_hardware.oil_safety import OilSafetyMonitor
        monitor = OilSafetyMonitor(config)
        monitor.check_frame_brightness(np.ones((100, 100), dtype=np.uint8) * 128)
        stats = monitor.get_brightness_stats()
        assert "history_len" in stats
        assert "warning_active" in stats


# ###########################################################################
# 3. LAYER 2: IMAGE ACQUISITION (simulation)
# ###########################################################################

class TestLayer2SimCamera:

    def test_camera_capture(self, config):
        from cap.layer2_acquisition.sim.sim_camera import SimCameraInterface
        cam = SimCameraInterface(config)
        cam.initialize()
        frame = cam.trigger_capture()
        assert frame is not None
        assert frame.ndim == 2  # Raw Bayer
        cam.release()

    def test_camera_frame_count(self, config):
        from cap.layer2_acquisition.sim.sim_camera import SimCameraInterface
        cam = SimCameraInterface(config)
        cam.initialize()
        cam.trigger_capture()
        cam.trigger_capture()
        assert cam.frame_count >= 2
        cam.release()


class TestLayer2FocusStacker:

    def test_stack_synthetic(self, config):
        from cap.layer2_acquisition.focus_stacker import FocusStacker
        stacker = FocusStacker(config)

        # Create 6 slightly different frames
        frames = []
        for z in range(6):
            f = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            frames.append(f)

        result = stacker.stack(frames=frames, slide_id=1, field_x=0, field_y=0)
        assert result.composite.shape == (256, 256, 3)
        assert result.composite.dtype == np.uint8
        assert result.stacking_duration_ms > 0
        assert len(result.z_distribution) > 0


# ###########################################################################
# 4. LAYER 3: IMAGE PROCESSING
# ###########################################################################

class TestLayer3Processing:

    def test_debayer(self):
        from cap.layer3_processing.debayer import debayer
        bayer = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        rgb = debayer(bayer, pattern="RG", bit_depth=8)
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8

    def test_denoise(self):
        from cap.layer3_processing.denoise import denoise
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = denoise(img, filter_type="gaussian", kernel_size=3)
        assert result.shape == img.shape

    def test_normalize_brightness(self):
        from cap.layer3_processing.normalize import normalize_brightness
        img = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
        result = normalize_brightness(img)
        assert result.shape == img.shape

    def test_resize_for_inference(self):
        from cap.layer3_processing.resize import resize_for_inference
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = resize_for_inference(img, target_width=640, target_height=640)
        assert result.shape == (640, 640, 3)

    def test_processing_pipeline(self, config):
        from cap.layer3_processing.pipeline import ProcessingPipeline
        pipeline = ProcessingPipeline(config)
        composite = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = pipeline.process_for_inference(composite)
        assert result.shape[0] == config.processing.model_input_height
        assert result.shape[1] == config.processing.model_input_width


# ###########################################################################
# 5. LAYER 4: AI INFERENCE
# ###########################################################################

class TestLayer4:

    def test_load_model_disabled(self, config):
        from cap.layer4_inference.model_loader import load_model
        config.inference.enabled = False
        assert load_model(config) is None

    def test_load_model_missing_file(self, config):
        from cap.layer4_inference.model_loader import load_model
        config.inference.model_path = "/nonexistent.pt"
        assert load_model(config) is None

    def test_run_inference_no_model(self, config):
        from cap.layer4_inference.inference import run_inference
        from cap.common.dataclasses import ProcessedFrame
        frames = [ProcessedFrame(slide_id=1, field_x=0, field_y=0,
                                 rgb_data=np.zeros((640, 640, 3), dtype=np.uint8),
                                 stacked=True, focus_score=0.9)]
        results = run_inference(None, frames, config)
        assert len(results) == 1
        assert results[0][1] is None

    def test_extract_detections_none(self, config):
        from cap.layer4_inference.postprocess import extract_detections
        from cap.common.dataclasses import ProcessedFrame
        frame = ProcessedFrame(slide_id=1, field_x=0, field_y=0,
                               rgb_data=np.zeros((10, 10, 3), dtype=np.uint8),
                               stacked=True, focus_score=0.9)
        assert extract_detections(frame, None, field_id=1, config=config) == []

    def test_aggregate_empty(self, config):
        from cap.layer4_inference.aggregator import aggregate_slide_results
        sr = aggregate_slide_results(slide_id=1, detections=[], config=config)
        assert sr.organism_counts == {}
        assert sr.severity_grades is None
        assert sr.overall_severity is None

    def test_aggregate_with_detections(self, config):
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.common.dataclasses import Detection, SeverityGrade
        dets = [Detection(field_id=1, class_name="yeast", confidence=0.9,
                          bbox=(10, 10, 20, 20), model_version="t") for _ in range(6)]
        sr = aggregate_slide_results(slide_id=1, detections=dets, config=config)
        assert sr.organism_counts["yeast"] == 6
        assert sr.severity_grades["yeast"] == SeverityGrade.MODERATE

    def test_disabled_results(self):
        from cap.layer4_inference.ai_disabled_mode import get_disabled_results
        r = get_disabled_results(42)
        assert r.slide_id == 42
        assert r.severity_grades is None

    def test_is_ai_available(self):
        from cap.layer4_inference.ai_disabled_mode import is_ai_available
        assert is_ai_available(None) is False
        assert is_ai_available(MagicMock()) is True


# ###########################################################################
# 6. LAYER 5: DATA LAYER
# ###########################################################################

class TestLayer5Database:

    def test_db_initialize_creates_tables(self, db):
        assert db.table_exists("patients")
        assert db.table_exists("slides")
        assert db.table_exists("fields")
        assert db.table_exists("detections")
        assert db.table_exists("results")
        assert db.table_exists("corrections")
        assert db.table_exists("audit_log")
        assert db.table_exists("model_versions")
        assert db.table_exists("technicians")
        assert db.table_exists("sessions")
        assert db.table_exists("focus_stacking_meta")

    def test_crud_patient_lifecycle(self, db):
        from cap.layer5_data import crud
        pid = crud.insert_patient(db, name="Luna", species="feline", owner_name="Doe")
        patient = crud.get_patient(db, pid)
        assert patient["name"] == "Luna"
        assert patient["species"] == "feline"
        results = crud.search_patients(db, "luna")
        assert len(results) == 1

    def test_crud_slide_lifecycle(self, db):
        from cap.layer5_data import crud
        pid = crud.insert_patient(db, name="Test", species="canine")
        sid = crud.insert_slide(db, patient_id=pid)
        slide = crud.get_slide(db, sid)
        assert slide["status"] == "pending"
        crud.update_slide_status(db, sid, "scanning")
        slide = crud.get_slide(db, sid)
        assert slide["status"] == "scanning"

    def test_crud_detection_batch(self, db):
        from cap.layer5_data import crud
        pid = crud.insert_patient(db, name="T", species="canine")
        sid = crud.insert_slide(db, patient_id=pid)
        fids = crud.insert_fields_batch(db, sid, [(0, 0), (1, 0)])
        dets = [{"field_id": fids[0], "class_name": "yeast", "confidence": 0.9,
                 "bbox_x": 10, "bbox_y": 10, "bbox_w": 20, "bbox_h": 20, "model_version": "v1"}]
        count = crud.insert_detections_batch(db, dets)
        assert count == 1
        counts = crud.get_organism_counts(db, sid)
        assert counts["yeast"] == 1

    def test_crud_results_with_none_severity(self, db):
        from cap.layer5_data import crud
        pid = crud.insert_patient(db, name="T", species="canine")
        sid = crud.insert_slide(db, patient_id=pid)
        crud.insert_results(db, slide_id=sid, organism_counts={}, severity_score="0",
                            severity_grades=None, plain_english_summary=None)
        r = crud.get_results(db, sid)
        assert r["severity_score"] == "0"
        assert r["plain_english_summary"] is None

    def test_crud_corrections(self, db):
        from cap.layer5_data import crud
        pid = crud.insert_patient(db, name="T", species="canine")
        sid = crud.insert_slide(db, patient_id=pid)
        fids = crud.insert_fields_batch(db, sid, [(0, 0)])
        dets = [{"field_id": fids[0], "class_name": "yeast", "confidence": 0.9,
                 "bbox_x": 10, "bbox_y": 10, "bbox_w": 20, "bbox_h": 20, "model_version": "v1"}]
        crud.insert_detections_batch(db, dets)
        det = crud.get_detections_for_field(db, fids[0])[0]
        tid = crud.insert_technician(db, name="T", login="t")
        cid = crud.insert_correction(db, det["detection_id"], tid, "yeast", "cocci_small")
        pending = crud.get_unreviewed_corrections(db)
        assert len(pending) == 1
        crud.mark_correction_reviewed(db, cid, "OK")
        assert len(crud.get_unreviewed_corrections(db)) == 0


class TestLayer5Audit:

    def test_audit_log(self, db):
        from cap.layer5_data.audit import AuditLogger, EventType
        audit = AuditLogger(db)
        audit.log(EventType.SYSTEM_STARTUP, details="Test startup")
        recent = audit.get_recent(limit=5)
        assert len(recent) == 1
        assert recent[0]["event_type"] == "system_startup"

    def test_audit_by_event_type(self, db):
        from cap.layer5_data.audit import AuditLogger, EventType
        audit = AuditLogger(db)
        audit.log(EventType.SCAN_STARTED, details="Scan 1")
        audit.log(EventType.SCAN_COMPLETED, details="Scan 1 done")
        audit.log(EventType.SCAN_STARTED, details="Scan 2")
        starts = audit.get_by_event_type(EventType.SCAN_STARTED)
        assert len(starts) == 2


class TestLayer5Export:

    def test_export_slide_csv(self, populated_db, temp_dir):
        from cap.layer5_data.export import export_slide_csv
        path = os.path.join(temp_dir, "dets.csv")
        export_slide_csv(populated_db["db"], populated_db["slides"][0], path)
        assert os.path.isfile(path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "class" in header
            rows = list(reader)
            assert len(rows) > 0

    def test_export_cvat_xml(self, populated_db, temp_dir):
        from cap.layer5_data.export import export_cvat_xml
        path = os.path.join(temp_dir, "anno.xml")
        export_cvat_xml(populated_db["db"], populated_db["slides"][0], path)
        assert os.path.isfile(path)
        with open(path) as f:
            assert "<annotations>" in f.read()

    def test_export_summary_csv(self, populated_db, temp_dir):
        from cap.layer5_data.export import export_summary_csv
        path = os.path.join(temp_dir, "summary.csv")
        export_summary_csv(populated_db["db"], populated_db["slides"][0], path)
        assert os.path.isfile(path)


class TestLayer5Backup:

    def test_disk_usage(self, config, db, temp_dir):
        from cap.layer5_data.backup import BackupManager
        config.storage.image_root = temp_dir
        mgr = BackupManager(config, db)
        usage = mgr.get_disk_usage()
        assert "total_bytes" in usage
        assert "total_gb" in usage
        assert "over_limit" in usage


# ###########################################################################
# 7. LAYER 6: UI SIGNALS
# ###########################################################################

class TestLayer6Signals:

    def test_all_signal_classes(self):
        from cap.layer6_ui.signals import (
            ScanSignals, InferenceSignals, MotorSignals,
            FocusSignals, SystemSignals, NavigationSignals,
        )
        for cls in [ScanSignals, InferenceSignals, MotorSignals,
                    FocusSignals, SystemSignals, NavigationSignals]:
            obj = cls()
            assert obj is not None


# ###########################################################################
# 8. LAYER 7: VISUALIZATION
# ###########################################################################

class TestLayer7Severity:

    def test_compute_severity_all_grades(self, config):
        from cap.layer7_visualization.severity import compute_severity
        from cap.common.dataclasses import SeverityGrade
        t = config.inference.severity_thresholds["default"]
        assert compute_severity(0, t) == SeverityGrade.NONE
        assert compute_severity(1, t) == SeverityGrade.RARE
        assert compute_severity(5, t) == SeverityGrade.MODERATE
        assert compute_severity(15, t) == SeverityGrade.MANY
        assert compute_severity(30, t) == SeverityGrade.PACKED

    def test_compute_all_severities(self, config):
        from cap.layer7_visualization.severity import compute_all_severities
        from cap.common.dataclasses import SeverityGrade
        counts = {"cocci_small": 7, "yeast": 2, "rods": 0}
        grades = compute_all_severities(counts, config.inference.severity_thresholds)
        assert grades["cocci_small"] == SeverityGrade.MODERATE
        assert grades["yeast"] == SeverityGrade.RARE
        assert grades["rods"] == SeverityGrade.NONE

    def test_generate_summary(self):
        from cap.layer7_visualization.severity import generate_summary, compute_all_severities, get_overall_severity
        from cap.common.dataclasses import SeverityGrade
        counts = {"cocci_small": 10, "yeast": 3}
        grades = compute_all_severities(counts)
        overall = get_overall_severity(grades)
        summary = generate_summary(counts, grades, overall)
        assert len(summary) > 10
        assert "cocci" in summary.lower()


class TestLayer7Stitcher:

    def test_stitch_fields(self, config):
        from cap.layer7_visualization.stitcher import SlideStitcher
        stitcher = SlideStitcher(config)
        images = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(4)]
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        result = stitcher.stitch(images, positions)
        assert result.ndim == 3
        assert result.shape[2] == 3


class TestLayer7Annotations:

    def test_annotate_image(self, config):
        from cap.layer7_visualization.annotations import AnnotationRenderer
        renderer = AnnotationRenderer(config)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [{"class": "yeast", "confidence": 0.9,
                 "bbox_x": 10, "bbox_y": 10, "bbox_w": 50, "bbox_h": 50}]
        result = renderer.annotate_image(img, dets)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)  # Should have drawn something


class TestLayer7TileBuilder:

    def test_build_tiles(self, config, temp_dir):
        from cap.layer7_visualization.tile_builder import TilePyramidBuilder
        builder = TilePyramidBuilder(config)
        composite = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = builder.build(composite, temp_dir, name="test")
        assert isinstance(result, str)
        assert os.path.exists(result)


class TestLayer7Transfer:

    def test_local_transfer(self, config, temp_dir):
        from cap.layer7_visualization.transfer import ExamRoomTransfer
        config.transfer.protocol = "local_copy"
        config.transfer.fallback_local_path = os.path.join(temp_dir, "reports")
        transfer = ExamRoomTransfer(config)

        # Create a dummy PDF
        pdf_path = os.path.join(temp_dir, "test.pdf")
        with open(pdf_path, "w") as f:
            f.write("fake pdf")

        result = transfer.transfer(pdf_path)
        assert os.path.isfile(result)


# ###########################################################################
# 9. LAYER 8: METRICS
# ###########################################################################

class TestLayer8:

    def test_clinic_overview(self, populated_db):
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        dash = ClinicDashboard(populated_db["db"])
        o = dash.get_overview()
        assert o["total_scans"] == 4
        assert o["total_patients"] == 3
        assert o["total_corrections"] == 6

    def test_ai_accuracy(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        ai = AIMetrics(populated_db["db"])
        s = ai.get_accuracy_summary()
        assert s["total_corrections"] == 6
        assert 0 < s["estimated_accuracy"] < 1

    def test_confusion_matrix(self, populated_db):
        from cap.layer8_metrics.ai_metrics import AIMetrics
        ai = AIMetrics(populated_db["db"])
        cm = ai.get_confusion_matrix()
        assert cm["total_corrections"] == 6
        assert len(cm["classes"]) > 0

    def test_metrics_export_json(self, populated_db, temp_dir):
        from cap.layer8_metrics.export import MetricsExporter
        exp = MetricsExporter(populated_db["db"])
        path = os.path.join(temp_dir, "metrics.json")
        exp.export_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["clinic"]["overview"]["total_scans"] == 4

    def test_metrics_export_csv(self, populated_db, temp_dir):
        from cap.layer8_metrics.export import MetricsExporter
        exp = MetricsExporter(populated_db["db"])
        paths = exp.export_csv(os.path.join(temp_dir, "csv"))
        assert len(paths) == 7
        for p in paths:
            assert os.path.isfile(p)


# ###########################################################################
# 10. LAYER 9: RETRAINING
# ###########################################################################

class TestLayer9:

    def test_correction_stats(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])
        stats = mgr.get_correction_stats()
        assert stats["total"] == 6
        assert stats["false_positive_count"] == 1

    def test_corrected_annotations(self, populated_db):
        from cap.layer9_retraining.corrections import CorrectionManager
        mgr = CorrectionManager(populated_db["db"])
        corrected = mgr.get_corrected_annotations(populated_db["slides"][0])
        for d in corrected:
            assert d["class"] != "false_positive"

    def test_cvat_export_corrected(self, populated_db, temp_dir, config):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exp = CVATExporter(populated_db["db"], config)
        path = os.path.join(temp_dir, "corr.xml")
        exp.export_cvat(populated_db["slides"][0], path, apply_corrections=True)
        assert os.path.isfile(path)

    def test_yolo_export(self, populated_db, temp_dir, config):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exp = CVATExporter(populated_db["db"], config)
        out = os.path.join(temp_dir, "yolo")
        exp.export_yolo_format(populated_db["slides"][0], out)
        assert os.path.isfile(os.path.join(out, "classes.txt"))

    def test_yolo_batch_with_dataset_yaml(self, populated_db, temp_dir, config):
        from cap.layer9_retraining.cvat_export import CVATExporter
        exp = CVATExporter(populated_db["db"], config)
        out = os.path.join(temp_dir, "yolo_batch")
        exp.export_yolo_batch(populated_db["slides"][:2], out)
        yaml_path = os.path.join(out, "dataset.yaml")
        assert os.path.isfile(yaml_path)
        with open(yaml_path) as f:
            content = f.read()
        assert "cocci_small" in content
        assert "ear_mites" in content


# ###########################################################################
# 11. WORKERS
# ###########################################################################

class TestWorkers:

    def test_scan_worker_importable(self):
        from cap.workers.scan_worker import ScanWorker
        assert callable(ScanWorker)

    def test_inference_worker_importable(self):
        from cap.workers.inference_worker import InferenceWorker
        assert callable(InferenceWorker)

    def test_report_worker_importable(self):
        from cap.workers.report_worker import ReportWorker
        assert callable(ReportWorker)


# ###########################################################################
# 12. CROSS-LAYER INTEGRATION
# ###########################################################################

class TestCrossLayerIntegration:

    def test_layer4_to_layer5_detection_roundtrip(self, db, config):
        """Detection → DB → read back → verify."""
        from cap.layer5_data import crud
        from cap.common.dataclasses import Detection

        pid = crud.insert_patient(db, name="T", species="canine")
        sid = crud.insert_slide(db, patient_id=pid)
        fids = crud.insert_fields_batch(db, sid, [(0, 0), (1, 0)])

        detections = [
            Detection(field_id=fids[0], class_name="cocci_small", confidence=0.92,
                      bbox=(50.0, 60.0, 20.0, 25.0), model_version="v1"),
            Detection(field_id=fids[1], class_name="yeast", confidence=0.78,
                      bbox=(100.0, 110.0, 15.0, 18.0), model_version="v1"),
        ]
        dicts = [{"field_id": d.field_id, "class_name": d.class_name,
                  "confidence": d.confidence, "bbox_x": d.bbox[0], "bbox_y": d.bbox[1],
                  "bbox_w": d.bbox[2], "bbox_h": d.bbox[3], "model_version": d.model_version}
                 for d in detections]
        crud.insert_detections_batch(db, dicts)

        stored = crud.get_detections_for_slide(db, sid)
        assert len(stored) == 2
        counts = crud.get_organism_counts(db, sid)
        assert counts["cocci_small"] == 1
        assert counts["yeast"] == 1

    def test_layer4_aggregator_to_layer7_severity(self, config):
        """Aggregator uses Layer 7 severity calculator correctly."""
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.common.dataclasses import Detection, SeverityGrade

        dets = []
        for _ in range(12):
            dets.append(Detection(field_id=1, class_name="cocci_small",
                                  confidence=0.9, bbox=(10, 10, 20, 20), model_version="t"))
        for _ in range(3):
            dets.append(Detection(field_id=1, class_name="ear_mites",
                                  confidence=0.95, bbox=(50, 50, 30, 30), model_version="t"))

        sr = aggregate_slide_results(slide_id=1, detections=dets, config=config)
        # ear_mites thresholds [1,2,5,10]: 3 → 2+ moderate
        assert sr.severity_grades["ear_mites"] == SeverityGrade.MODERATE
        # cocci_small thresholds [1,5,15,30]: 12 → 2+ moderate (< 15)
        assert sr.severity_grades["cocci_small"] == SeverityGrade.MODERATE
        assert "mite" in sr.plain_english_summary.lower()

    def test_layer5_to_layer8_metrics(self, populated_db):
        """Metrics read from the same DB that CRUD writes to."""
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        from cap.layer8_metrics.ai_metrics import AIMetrics

        dash = ClinicDashboard(populated_db["db"])
        ai = AIMetrics(populated_db["db"])

        assert dash.get_overview()["total_scans"] == 4
        assert ai.get_accuracy_summary()["total_corrections"] == 6

    def test_layer5_to_layer9_corrections_roundtrip(self, populated_db, temp_dir, config):
        """Corrections → CVAT export → verify false positives excluded."""
        from cap.layer9_retraining.corrections import CorrectionManager
        from cap.layer9_retraining.cvat_export import CVATExporter

        mgr = CorrectionManager(populated_db["db"])
        corrected = mgr.get_corrected_annotations(populated_db["slides"][0])
        for d in corrected:
            assert d["class"] != "false_positive"

        exp = CVATExporter(populated_db["db"], config)
        path = os.path.join(temp_dir, "cross.xml")
        exp.export_cvat(populated_db["slides"][0], path)
        with open(path) as f:
            xml = f.read()
        assert "false_positive" not in xml


# ###########################################################################
# 13. END-TO-END PIPELINE SIMULATION
# ###########################################################################

class TestEndToEnd:

    def test_full_scan_to_report_simulation(self, db, config, temp_dir):
        """
        Simulates the entire workflow from session start to report,
        verifying data flows correctly through all layers.
        """
        from cap.layer5_data import crud
        from cap.layer5_data.audit import AuditLogger, EventType
        from cap.layer1_hardware.scan_region import ScanRegionManager
        from cap.layer3_processing.pipeline import ProcessingPipeline
        from cap.layer4_inference.aggregator import aggregate_slide_results
        from cap.layer4_inference.ai_disabled_mode import get_disabled_results
        from cap.layer7_visualization.severity import compute_all_severities, get_overall_severity
        from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
        from cap.common.dataclasses import Detection, SeverityGrade

        audit = AuditLogger(db)
        config.storage.image_root = os.path.join(temp_dir, "slides")
        os.makedirs(config.storage.image_root, exist_ok=True)

        # === Step 1: Session + Patient ===
        tech_id = crud.insert_technician(db, name="System Test", login="sys")
        session_id = crud.start_session(db, tech_id)
        patient_id = crud.insert_patient(db, name="E2E Dog", species="canine", owner_name="Test")
        slide_id = crud.insert_slide(db, patient_id=patient_id, session_id=session_id, technician_id=tech_id)
        audit.log(EventType.SCAN_STARTED, details=f"E2E slide {slide_id}")

        # === Step 2: Scan Region ===
        region_mgr = ScanRegionManager(config)
        fw = region_mgr.field_width_steps
        fh = region_mgr.field_height_steps
        scan_region = region_mgr.set_polygon([
            (0, 0), (fw * 5, 0), (fw * 5, fh * 5), (0, fh * 5)
        ])
        assert scan_region.field_count > 0

        # === Step 3: Insert fields ===
        field_positions = scan_region.field_positions[:6]  # Limit for speed
        field_ids = crud.insert_fields_batch(db, slide_id, field_positions)
        field_id_map = {pos: fid for pos, fid in zip(field_positions, field_ids)}

        # === Step 4: Simulate captured frames ===
        proc_pipeline = ProcessingPipeline(config)
        processed_images = []
        for pos in field_positions:
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            processed = proc_pipeline.process_for_inference(img)
            processed_images.append((pos, processed))

        # === Step 5: Simulate detections (mock AI) ===
        all_detections = []
        for (fx, fy), img in processed_images:
            fid = field_id_map[(fx, fy)]
            # Simulate finding 2 organisms per field
            all_detections.append(Detection(
                field_id=fid, class_name="cocci_small", confidence=0.88,
                bbox=(100.0, 100.0, 20.0, 20.0), model_version="e2e_test"))
            all_detections.append(Detection(
                field_id=fid, class_name="yeast", confidence=0.76,
                bbox=(200.0, 200.0, 15.0, 15.0), model_version="e2e_test"))

        # === Step 6: Store detections ===
        det_dicts = [{"field_id": d.field_id, "class_name": d.class_name,
                      "confidence": d.confidence, "bbox_x": d.bbox[0], "bbox_y": d.bbox[1],
                      "bbox_w": d.bbox[2], "bbox_h": d.bbox[3], "model_version": d.model_version}
                     for d in all_detections]
        crud.insert_detections_batch(db, det_dicts)

        # === Step 7: Aggregate ===
        slide_results = aggregate_slide_results(
            slide_id=slide_id, detections=all_detections, config=config,
            field_grid_size=(2, 3), model_version="e2e_test")

        assert slide_results.organism_counts["cocci_small"] == 6
        assert slide_results.organism_counts["yeast"] == 6
        assert slide_results.severity_grades is not None
        assert slide_results.overall_severity is not None
        assert slide_results.plain_english_summary is not None

        # === Step 8: Store results ===
        sev_serial = {c: g.value for c, g in slide_results.severity_grades.items()}
        crud.insert_results(
            db, slide_id=slide_id, organism_counts=slide_results.organism_counts,
            severity_score=slide_results.overall_severity.value,
            severity_grades=sev_serial, flagged_field_ids=slide_results.flagged_field_ids,
            model_version="e2e_test", plain_english_summary=slide_results.plain_english_summary)

        # === Step 9: Verify stored results ===
        stored = crud.get_results(db, slide_id)
        assert stored is not None
        assert stored["organism_counts"]["cocci_small"] == 6
        assert stored["plain_english_summary"] is not None

        # === Step 10: Simulate correction ===
        det_list = crud.get_detections_for_slide(db, slide_id)
        crud.insert_correction(db, det_list[0]["detection_id"], tech_id, det_list[0]["class"], "rods")

        # === Step 11: Mark slide complete and verify metrics ===
        crud.update_slide_status(db, slide_id, "complete")

        dash = ClinicDashboard(db)
        overview = dash.get_overview()
        assert overview["total_scans"] >= 1
        assert overview["total_corrections"] >= 1

        # === Step 12: Retraining export ===
        from cap.layer9_retraining.cvat_export import CVATExporter
        exporter = CVATExporter(db, config)
        yolo_dir = os.path.join(temp_dir, "yolo_e2e")
        exporter.export_yolo_format(slide_id, yolo_dir)
        assert os.path.isfile(os.path.join(yolo_dir, "classes.txt"))

        # === Step 13: End session ===
        crud.end_session(db, session_id)
        audit.log(EventType.SCAN_COMPLETED, details=f"E2E complete: slide {slide_id}")

        # Final verification
        slide = crud.get_slide(db, slide_id)
        assert slide["status"] == "complete"
        audit_entries = audit.get_by_event_type(EventType.SCAN_COMPLETED)
        assert len(audit_entries) >= 1

        print("\n✓ Full end-to-end simulation passed: "
              f"{len(field_positions)} fields, "
              f"{len(all_detections)} detections, "
              f"severity={slide_results.overall_severity.value}")


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])