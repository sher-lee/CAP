"""
Continuous-Scan End-to-End Integration Test
============================================
Drives a small continuous scan all the way through the sim:

    SimMotorController (constant-velocity X, oscillating Z)
        → SimCameraInterface (position-aware via WorldTexture)
            → FrameTagger → FrameGrouper
                → FocusStacker (with step-counter prior)
                    → MasterCanvas (direct placement)

Validates:
  - The motor's time-aware position model advances X while Z moves block
  - The controller orchestrates all three axes correctly
  - The grouper assembles complete 6-frame VirtualFields per virtual position
  - The stacker accepts the prior and produces composites
  - The canvas placer absorbs all composites without crashing

Run from the project root:
    python -m pytest cap/tests/test_continuous_e2e.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cap.common.dataclasses import FocusMapResult, ScanRegion  # noqa: E402
from cap.config.config_loader import load_config  # noqa: E402
from cap.layer1_hardware.sim.sim_motor import SimMotorController  # noqa: E402
from cap.layer2_acquisition.continuous_scan_controller import (  # noqa: E402
    ContinuousScanController,
    plan_scan_lines,
)
from cap.layer2_acquisition.focus_stacker import FocusStacker  # noqa: E402
from cap.layer2_acquisition.frame_grouper import VirtualField  # noqa: E402
from cap.layer2_acquisition.frame_tagger import x_steps_to_um, y_steps_to_um  # noqa: E402
from cap.layer2_acquisition.sim.sim_camera import SimCameraInterface  # noqa: E402
from cap.layer7_visualization.canvas_placer import canvas_for_slide  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a small, fast continuous-scan config
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_config():
    """Load real config and shrink it so the test runs in seconds, not minutes."""
    cfg = load_config()
    # Small synthetic frame for fast WorldTexture rendering.
    cfg.sim.synthetic_image_width = 384
    cfg.sim.synthetic_image_height = 216
    cfg.sim.motor_delay_ms = 0
    cfg.sim.generate_synthetic = True
    cfg.sim.camera_test_image_dir = "/nonexistent_so_world_texture_path_is_taken"

    # Continuous-mode settings
    cfg.continuous_scan.enabled = True
    cfg.continuous_scan.target_fields_per_second = 1.0  # Slow → X advances modestly per cycle
    cfg.continuous_scan.frames_per_field = 6
    cfg.continuous_scan.z_cycle_range_um = 2.5
    cfg.continuous_scan.z_cycle_waveform = "triangle"
    cfg.continuous_scan.z_tracking_enabled = True

    # Allow Z negative so the cycle around a 0-um center won't clip.
    # (Real hardware will set z_min based on physical limits.)
    cfg.motor.z_min = -10000

    return cfg


# ---------------------------------------------------------------------------
# Helper: build a small ScanRegion (2 lines × 2 fields)
# ---------------------------------------------------------------------------

def _make_2x2_region(cfg) -> ScanRegion:
    """Build a 2-line × 2-field region using the actual ScanRegionManager."""
    from cap.layer1_hardware.scan_region import ScanRegionManager
    mgr = ScanRegionManager(cfg)
    fw = mgr.field_width_steps
    fh = mgr.field_height_steps
    return mgr.set_polygon([
        (10, 10),
        (10 + fw * 2, 10),
        (10 + fw * 2, 10 + fh * 2),
        (10, 10 + fh * 2),
    ])


def _flat_focus_map(z_um: float = 2.0) -> FocusMapResult:
    """A flat focus surface at z = z_um — keeps Z cycle inside bounds."""
    return FocusMapResult(
        sample_points=[],
        surface_coefficients=np.array([z_um, 0.0, 0.0, 0.0, 0.0, 0.0]),
        grid_size=(1, 1),
        fit_residual=0.0,
    )


# ===========================================================================
# Tests
# ===========================================================================

class TestSimMotorContinuous:
    """Direct unit-style coverage of the new SimMotorController primitives."""

    def test_constant_velocity_advances_position_with_time(self, fast_config):
        m = SimMotorController(fast_config)
        m.home_all()

        m.start_constant_velocity("x", 1000.0)  # 1000 steps/sec
        time.sleep(0.05)
        pos1 = m.get_position("x")
        time.sleep(0.05)
        pos2 = m.get_position("x")
        m.stop_axis("x")

        # After ~50 ms at 1000 steps/sec, ~50 steps elapsed (allow generous slack)
        assert pos1 > 20
        assert pos2 > pos1

    def test_stop_axis_freezes_position(self, fast_config):
        m = SimMotorController(fast_config)
        m.home_all()

        m.start_constant_velocity("x", 1000.0)
        time.sleep(0.05)
        m.stop_axis("x")
        frozen = m.get_position("x")
        time.sleep(0.05)
        # No further advance after stop
        assert m.get_position("x") == frozen

    def test_position_snapshot_atomic(self, fast_config):
        """All three axes captured at one timestamp, with X mid-motion."""
        m = SimMotorController(fast_config)
        m.home_all()
        m.move_to("y", 500)
        m.move_to("z", 100)
        m.start_constant_velocity("x", 500.0)
        time.sleep(0.02)
        snap = m.read_position_snapshot()
        m.stop_axis("x")

        assert snap.y_steps == 500
        assert snap.z_steps == 100
        assert snap.x_steps > 0
        assert snap.timestamp > 0

    def test_z_move_does_not_disturb_x_velocity(self, fast_config):
        """The whole point of the continuous design — X keeps moving while Z is commanded."""
        m = SimMotorController(fast_config)
        m.home_all()

        m.start_constant_velocity("x", 1000.0)
        x_before = m.get_position("x")
        m.move_to_blocking("z", 50)
        time.sleep(0.05)
        x_after = m.get_position("x")
        m.stop_axis("x")

        assert m.get_position("z") == 50
        assert x_after > x_before


class TestSimCameraPositionAware:
    def test_world_texture_engages_when_provider_set(self, fast_config):
        cam = SimCameraInterface(fast_config)
        cam.initialize()

        pose = [0.0, 0.0, 0.0]
        cam.set_position_provider(lambda: tuple(pose))

        # Same position twice → identical frames (deterministic)
        f1 = cam.capture_one()
        f2 = cam.capture_one()
        assert np.array_equal(f1, f2)

        # Different position → different frames
        pose[0] = 5.0
        f3 = cam.capture_one()
        assert not np.array_equal(f1, f3)

        cam.release()

    def test_z_focus_affects_frame(self, fast_config):
        """A frame at z far from ideal should differ from one at z=ideal."""
        cam = SimCameraInterface(fast_config)
        cam.initialize()

        pose = [0.0, 0.0, 0.0]
        cam.set_position_provider(lambda: tuple(pose))
        # z=0 should be near ideal (at world (0,0), ideal_z ≈ 0)
        f_ideal = cam.capture_one()

        pose[2] = 5.0  # Way out of focus
        f_blurred = cam.capture_one()
        assert not np.array_equal(f_ideal, f_blurred)
        # Blurred frame should have lower variance (averaged out)
        assert np.var(f_blurred.astype(np.float32)) < np.var(f_ideal.astype(np.float32))

        cam.release()


# ---------------------------------------------------------------------------
# End-to-end continuous scan
# ---------------------------------------------------------------------------

class TestContinuousScanEndToEnd:

    def test_2x2_scan_produces_complete_virtual_fields(self, fast_config):
        cfg = fast_config
        region = _make_2x2_region(cfg)
        assert region.field_count == 4  # 2 lines × 2 fields

        motor = SimMotorController(cfg)
        motor.home_all()
        camera = SimCameraInterface(cfg)
        camera.initialize()

        # Wire the camera to the motor so frames render at the live pose.
        x_steps_per_mm = cfg.motor.x_steps_per_mm
        y_steps_per_mm = cfg.motor.y_steps_per_mm
        z_step_um = cfg.motor.z_step_size_microns

        def pose_provider():
            snap = motor.read_position_snapshot()
            return (
                x_steps_to_um(snap.x_steps, x_steps_per_mm),
                y_steps_to_um(snap.y_steps, y_steps_per_mm),
                snap.z_steps * z_step_um,
            )

        camera.set_position_provider(pose_provider)

        controller = ContinuousScanController(
            config=cfg,
            motor_controller=motor,
            camera_interface=camera,
            focus_map=_flat_focus_map(z_um=2.0),
            scan_region=region,
            slide_id=1,
        )

        emitted: list[VirtualField] = []
        result = controller.run(field_complete_callback=lambda vf: emitted.append(vf))

        camera.release()

        # Diagnostics print on failure
        complete = [vf for vf in emitted if vf.is_complete]
        assert result["fields_completed"] == len(complete)
        assert len(complete) >= 4, (
            f"Expected at least 4 complete virtual fields, got {len(complete)}. "
            f"Stats: {result}"
        )
        for vf in complete:
            assert vf.n_frames == 6
            assert len(vf.expected_shifts_px) == 6
            assert vf.expected_shifts_px[0] == (0.0, 0.0)

    def test_stacker_consumes_continuous_output(self, fast_config):
        """Pipe one virtual field through the focus stacker with the prior."""
        cfg = fast_config
        region = _make_2x2_region(cfg)

        motor = SimMotorController(cfg)
        motor.home_all()
        camera = SimCameraInterface(cfg)
        camera.initialize()

        def pose_provider():
            snap = motor.read_position_snapshot()
            return (
                x_steps_to_um(snap.x_steps, cfg.motor.x_steps_per_mm),
                y_steps_to_um(snap.y_steps, cfg.motor.y_steps_per_mm),
                snap.z_steps * cfg.motor.z_step_size_microns,
            )
        camera.set_position_provider(pose_provider)

        controller = ContinuousScanController(
            config=cfg,
            motor_controller=motor,
            camera_interface=camera,
            focus_map=_flat_focus_map(z_um=2.0),
            scan_region=region,
            slide_id=1,
        )

        captured: list[VirtualField] = []
        controller.run(field_complete_callback=lambda vf: captured.append(vf))
        camera.release()

        complete = [vf for vf in captured if vf.is_complete]
        assert complete, f"No complete fields captured. Got: {len(captured)} total"

        stacker = FocusStacker(cfg)
        vf = complete[0]
        # Frames are uint8 single-channel. Stacker handles 2D fine.
        stacked = stacker.stack(
            frames=[fr.bayer_data for fr in vf.frames],
            slide_id=1,
            field_x=vf.frames[0].field_x,
            field_y=vf.frames[0].field_y,
            expected_shifts_px=vf.expected_shifts_px,
        )

        assert stacked.composite.shape == vf.frames[0].bayer_data.shape
        assert len(stacked.registration_shifts) == 6
        assert stacked.registration_shifts[0] == (0.0, 0.0)

    def test_canvas_placement_round_trip(self, fast_config):
        """Stack every virtual field and place onto a master canvas."""
        cfg = fast_config
        region = _make_2x2_region(cfg)

        motor = SimMotorController(cfg)
        motor.home_all()
        camera = SimCameraInterface(cfg)
        camera.initialize()

        def pose_provider():
            snap = motor.read_position_snapshot()
            return (
                x_steps_to_um(snap.x_steps, cfg.motor.x_steps_per_mm),
                y_steps_to_um(snap.y_steps, cfg.motor.y_steps_per_mm),
                snap.z_steps * cfg.motor.z_step_size_microns,
            )
        camera.set_position_provider(pose_provider)

        controller = ContinuousScanController(
            config=cfg,
            motor_controller=motor,
            camera_interface=camera,
            focus_map=_flat_focus_map(z_um=2.0),
            scan_region=region,
            slide_id=1,
        )

        all_fields: list[VirtualField] = []
        controller.run(field_complete_callback=lambda vf: all_fields.append(vf))
        camera.release()

        complete = [vf for vf in all_fields if vf.is_complete]
        assert len(complete) >= 4

        # Stack each, placing onto a small canvas sized to the region's extent
        stacker = FocusStacker(cfg)
        # Canvas at the camera's native px/μm so placement is 1:1 with sources
        px_per_um = cfg.sim.synthetic_image_width / (cfg.camera.fov_width_mm * 1000.0)
        canvas = canvas_for_slide(
            slide_width_mm=0.05,   # 50 μm — covers the small region with margin
            slide_height_mm=0.05,
            px_per_um=px_per_um,
            block_size=cfg.focus.block_size,
            channels=1,            # grayscale sim
        )

        placements = 0
        for vf in complete:
            stacked = stacker.stack(
                frames=[fr.bayer_data for fr in vf.frames],
                slide_id=1,
                field_x=vf.frames[0].field_x,
                field_y=vf.frames[0].field_y,
                expected_shifts_px=vf.expected_shifts_px,
            )
            res = canvas.place(stacked, x_um=vf.field_x_um, y_um=vf.field_y_um)
            if res.placed:
                placements += 1
                assert res.blocks_painted > 0

        assert placements == len(complete)
        # Some canvas coverage achieved
        assert canvas.coverage() > 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
