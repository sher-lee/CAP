"""
Continuous-Scan Unit Tests
==========================
Covers the Chunk 1 modules added for continuous-motion scanning:

  - cap.layer1_hardware.z_cycle_generator
  - cap.layer2_acquisition.frame_grouper
  - cap.layer2_acquisition.focus_stacker (the new expected_shifts_px prior)

These tests are pure-Python — no motors, no camera, no real hardware.
They validate the data-flow modules in isolation. Sim end-to-end runs
land in Chunk 2.

Run from the project root:
    python -m pytest tests/test_continuous_scan.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cap.common.dataclasses import RawFrame  # noqa: E402
from cap.layer1_hardware.z_cycle_generator import (  # noqa: E402
    depth_index_sequence,
    evaluate_focus_surface,
    generate_cycle_targets,
    generate_depth_offsets,
)
from cap.layer2_acquisition.focus_stacker import FocusStacker  # noqa: E402
from cap.layer2_acquisition.frame_grouper import (  # noqa: E402
    FrameGrouper,
    VirtualField,
    group_frames_batch,
)


# ===========================================================================
# z_cycle_generator
# ===========================================================================

class TestFocusSurface:
    def test_constant_surface(self):
        """f(x,y) = 5 + 0*x + 0*y + ... should evaluate to 5 everywhere."""
        z = evaluate_focus_surface(100.0, 200.0, [5.0, 0, 0, 0, 0, 0])
        assert z == pytest.approx(5.0)

    def test_linear_x_surface(self):
        """f(x,y) = 0 + 0.1*x — slope only in X."""
        z = evaluate_focus_surface(50.0, 999.0, [0, 0.1, 0, 0, 0, 0])
        assert z == pytest.approx(5.0)

    def test_full_polynomial(self):
        """Quadratic terms exercised."""
        # f(x,y) = 1 + 2x + 3y + 4x² + 5y² + 6xy
        # at (1, 1) = 1+2+3+4+5+6 = 21
        z = evaluate_focus_surface(1.0, 1.0, [1, 2, 3, 4, 5, 6])
        assert z == pytest.approx(21.0)

    def test_wrong_coefficient_count_raises(self):
        with pytest.raises(ValueError):
            evaluate_focus_surface(0.0, 0.0, [1, 2, 3])


class TestDepthOffsets:
    def test_six_depths_explicit(self):
        offs = generate_depth_offsets(6, 2.5)
        # step = 2.5/6 ≈ 0.4167; first offset = -1.25 + 0.4167/2 = -1.0417
        assert len(offs) == 6
        assert offs[0] == pytest.approx(-2.5 / 2 + 2.5 / 12)
        assert offs[-1] == pytest.approx(2.5 / 2 - 2.5 / 12)
        # symmetric around 0
        assert sum(offs) == pytest.approx(0.0, abs=1e-9)

    def test_evenly_spaced(self):
        offs = generate_depth_offsets(6, 3.0)
        diffs = np.diff(offs)
        assert all(d == pytest.approx(diffs[0]) for d in diffs)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            generate_depth_offsets(1, 2.5)
        with pytest.raises(ValueError):
            generate_depth_offsets(6, 0.0)


class TestDepthIndexSequence:
    def test_sawtooth_always_forward(self):
        for cyc in (0, 1, 5, 999):
            assert depth_index_sequence(6, cyc, "sawtooth") == [0, 1, 2, 3, 4, 5]

    def test_triangle_alternates(self):
        assert depth_index_sequence(6, 0, "triangle") == [0, 1, 2, 3, 4, 5]
        assert depth_index_sequence(6, 1, "triangle") == [5, 4, 3, 2, 1, 0]
        assert depth_index_sequence(6, 2, "triangle") == [0, 1, 2, 3, 4, 5]
        assert depth_index_sequence(6, 3, "triangle") == [5, 4, 3, 2, 1, 0]

    def test_unknown_waveform_raises(self):
        with pytest.raises(ValueError):
            depth_index_sequence(6, 0, "square")


class TestGenerateCycleTargets:
    def test_no_surface_uses_fallback(self):
        targets = generate_cycle_targets(
            x_um=100, y_um=200,
            coefficients=None,
            n_depths=6, total_range_um=2.5,
            cycle_index=0, waveform="triangle",
            fallback_z_center_um=10.0,
        )
        assert len(targets) == 6
        # Average of all six z values should equal the fallback center
        avg_z = sum(z for _, z in targets) / 6
        assert avg_z == pytest.approx(10.0, abs=1e-9)

    def test_surface_tracks_position(self):
        # f(x,y) = 0 + 1*x + 0*y + ...; at x=50, center should be 50
        targets = generate_cycle_targets(
            x_um=50, y_um=0,
            coefficients=[0, 1.0, 0, 0, 0, 0],
            n_depths=6, total_range_um=2.5,
            cycle_index=0, waveform="triangle",
        )
        avg_z = sum(z for _, z in targets) / 6
        assert avg_z == pytest.approx(50.0, abs=1e-9)

    def test_triangle_reverses_capture_order(self):
        """Cycle 1 (triangle) should visit depths in reverse order."""
        cyc0 = generate_cycle_targets(0, 0, None, 6, 2.5, 0, "triangle", 0.0)
        cyc1 = generate_cycle_targets(0, 0, None, 6, 2.5, 1, "triangle", 0.0)
        idx0 = [d for d, _ in cyc0]
        idx1 = [d for d, _ in cyc1]
        assert idx0 == [0, 1, 2, 3, 4, 5]
        assert idx1 == [5, 4, 3, 2, 1, 0]


# ===========================================================================
# frame_grouper
# ===========================================================================

def _make_frame(
    x_steps: int,
    y_steps: int,
    z_steps: int,
    depth_idx: int,
    scan_line: int = 0,
    scan_dir: int = 1,
    frame_id: int = 0,
    img_width: int = 100,
    img_height: int = 60,
) -> RawFrame:
    """Build a minimal RawFrame for grouper tests."""
    return RawFrame(
        slide_id=0,
        field_x=-1,
        field_y=-1,
        z_depth=depth_idx,
        timestamp=0.0,
        bayer_data=np.zeros((img_height, img_width), dtype=np.uint8),
        motor_position=(x_steps, y_steps, z_steps),
        frame_id=frame_id,
        scan_line=scan_line,
        scan_direction=scan_dir,
    )


class TestFrameGrouper:
    """Calibration: 1000 steps/mm = 1 step/μm. FOV: 100 μm × 60 μm."""

    def _grouper(self, frames_per_field=6) -> FrameGrouper:
        return FrameGrouper(
            fov_width_um=100.0,
            fov_height_um=60.0,
            frames_per_field=frames_per_field,
            x_steps_per_mm=1000.0,
            y_steps_per_mm=1000.0,
        )

    def test_complete_field_emitted_on_sixth_frame(self):
        g = self._grouper()
        g.begin_scan_line(0, +1, line_origin_x_um=0.0, line_y_um=0.0)

        emitted = []
        for i in range(6):
            x_um = 10.0 + i * 1.0  # all within field 0 (0–100 μm)
            f = _make_frame(x_steps=int(x_um), y_steps=0, z_steps=0, depth_idx=i, frame_id=i)
            vf = g.feed(f, frame_x_um=x_um, frame_y_um=0.0)
            if vf is not None:
                emitted.append(vf)

        assert len(emitted) == 1
        vf = emitted[0]
        assert vf.is_complete is True
        assert vf.scan_line == 0
        assert vf.scan_direction == +1
        assert vf.field_index_in_line == 0
        assert vf.field_x_um == pytest.approx(50.0)  # center of field 0
        assert len(vf.frames) == 6

    def test_two_consecutive_fields_on_same_line(self):
        g = self._grouper()
        g.begin_scan_line(0, +1, 0.0, 0.0)

        emitted = []
        # field 0: x = 10..60 (within 0–100)
        # field 1: x = 110..160 (within 100–200)
        xs = [10, 20, 30, 40, 50, 60, 110, 120, 130, 140, 150, 160]
        for i, x_um in enumerate(xs):
            f = _make_frame(int(x_um), 0, 0, depth_idx=i % 6, frame_id=i)
            vf = g.feed(f, x_um, 0.0)
            if vf is not None:
                emitted.append(vf)

        assert len(emitted) == 2
        assert emitted[0].field_index_in_line == 0
        assert emitted[1].field_index_in_line == 1
        assert emitted[0].field_x_um == pytest.approx(50.0)
        assert emitted[1].field_x_um == pytest.approx(150.0)

    def test_reverse_direction_field_indexing(self):
        """On a -1 direction line, the line origin is at the high X end."""
        g = self._grouper()
        g.begin_scan_line(1, -1, line_origin_x_um=200.0, line_y_um=60.0)

        emitted = []
        # x decreasing: 190, 180, ..., 140 (all within field 0, distance 10..60)
        for i in range(6):
            x_um = 200.0 - 10.0 - i * 1.0  # 190, 189, ... 185
            f = _make_frame(int(x_um), 60, 0, depth_idx=i, frame_id=i, scan_line=1, scan_dir=-1)
            vf = g.feed(f, x_um, 60.0)
            if vf is not None:
                emitted.append(vf)

        assert len(emitted) == 1
        vf = emitted[0]
        assert vf.scan_direction == -1
        assert vf.field_index_in_line == 0
        # Field center is line_origin + dir * (idx + 0.5) * fov_w = 200 + (-1) * 50 = 150
        assert vf.field_x_um == pytest.approx(150.0)
        assert vf.field_y_um == pytest.approx(60.0)

    def test_partial_field_drained_by_flush(self):
        g = self._grouper()
        g.begin_scan_line(0, +1, 0.0, 0.0)
        for i in range(3):
            f = _make_frame(int(10 + i), 0, 0, depth_idx=i, frame_id=i)
            assert g.feed(f, 10.0 + i, 0.0) is None  # not yet complete

        partials = g.flush_line()
        assert len(partials) == 1
        assert partials[0].is_complete is False
        assert partials[0].n_frames == 3

    def test_partial_carried_over_when_new_line_begins(self):
        g = self._grouper()
        g.begin_scan_line(0, +1, 0.0, 0.0)
        # Capture only 4 frames before the line ends
        for i in range(4):
            g.feed(_make_frame(int(10 + i), 0, 0, depth_idx=i, frame_id=i), 10.0 + i, 0.0)

        leftover = g.begin_scan_line(1, -1, line_origin_x_um=200.0, line_y_um=60.0)
        assert len(leftover) == 1
        assert leftover[0].is_complete is False
        assert leftover[0].n_frames == 4

    def test_frame_field_xy_stamped_on_completion(self):
        """field_x and field_y on each frame should be set when the field is emitted."""
        g = self._grouper()
        g.begin_scan_line(0, +1, 0.0, 0.0)

        produced_frames = []
        for i in range(6):
            f = _make_frame(10, 0, 0, depth_idx=i, frame_id=i)
            produced_frames.append(f)
            vf = g.feed(f, 10.0, 0.0)
            if vf is not None:
                # All six frames should now have field_x/field_y stamped
                for fr in vf.frames:
                    assert fr.field_x != -1
                    assert fr.field_y != -1

    def test_prior_shifts_zero_for_zero_motion(self):
        g = self._grouper()
        g.begin_scan_line(0, +1, 0.0, 0.0)
        emitted = None
        for i in range(6):
            f = _make_frame(50, 0, 0, depth_idx=i, frame_id=i)  # all at same X
            vf = g.feed(f, 50.0, 0.0)
            if vf is not None:
                emitted = vf
        assert emitted is not None
        assert emitted.expected_shifts_px == [(0.0, 0.0)] * 6

    def test_prior_shift_sign_positive_for_positive_x_motion(self):
        """
        When stage moves +ΔX between frame 0 and frame i, the prior dx
        should be POSITIVE (matches cv2.warpAffine + phaseCorrelate sign
        convention used by FocusStacker._register_frames).
        """
        # FOV width 100 μm, frame width 100 px → 1 px/um
        g = self._grouper()
        g.begin_scan_line(0, +1, 0.0, 0.0)
        emitted = None
        # Stage advances 1 step (= 1 μm = 1 px) per frame
        for i in range(6):
            f = _make_frame(50 + i, 0, 0, depth_idx=i, frame_id=i)
            vf = g.feed(f, 50.0 + i, 0.0)
            if vf is not None:
                emitted = vf

        assert emitted is not None
        shifts = emitted.expected_shifts_px
        assert shifts[0] == (0.0, 0.0)
        # frame i moved +i μm = +i px; prior should be +i px
        for i in range(1, 6):
            assert shifts[i][0] == pytest.approx(float(i))
            assert shifts[i][1] == pytest.approx(0.0)


class TestBatchGroupHelper:
    def test_round_trip_two_lines(self):
        # 12 frames, 2 lines of 6 frames each, both forward
        frames = []
        for line in (0, 1):
            for i in range(6):
                x_um = 10 + i
                frames.append(_make_frame(
                    x_steps=int(x_um),
                    y_steps=line * 60,
                    z_steps=0,
                    depth_idx=i,
                    scan_line=line,
                    scan_dir=+1,
                    frame_id=line * 6 + i,
                ))

        origins = {0: (0.0, 0.0, +1), 1: (0.0, 60.0, +1)}
        out = list(group_frames_batch(
            frames,
            line_origins_um=origins,
            fov_width_um=100.0,
            fov_height_um=60.0,
            frames_per_field=6,
            x_steps_per_mm=1000.0,
            y_steps_per_mm=1000.0,
        ))
        complete = [vf for vf in out if vf.is_complete]
        assert len(complete) == 2
        assert {vf.scan_line for vf in complete} == {0, 1}


# ===========================================================================
# focus_stacker prior path
# ===========================================================================

class _StackerCfg:
    """Minimal config object satisfying FocusStacker's hasattr('focus') path."""
    class _F:
        block_size = 16
        blend_sigma = 4.0
        max_registration_shift = 50

    focus = _F()


def _synthetic_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """High-frequency pattern that phase correlation can lock onto."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    return img


def _simulate_stage_motion(ref: np.ndarray, stage_dx_px: int, stage_dy_px: int) -> np.ndarray:
    """
    Build the frame the camera would see AFTER the stage moved by
    (stage_dx_px, stage_dy_px) in world coordinates relative to where
    `ref` was captured.

    When the stage moves +ΔX in world, image content shifts -ΔX in the
    frame (the same scene is now imaged at a smaller pixel index).
    Implemented with np.roll so wrap-around content matches the
    pre-shifted reference well enough for phase correlation to lock on.
    """
    return np.roll(ref, shift=(-stage_dy_px, -stage_dx_px), axis=(0, 1))


class TestStackerWithPrior:
    def test_no_prior_rejects_large_shift(self):
        """Without a prior, a frame shifted far beyond max_registration_shift
        should be detected as out-of-range and recorded with a (0, 0)
        shift (the rejection sentinel), leaving the frame unregistered."""
        stacker = FocusStacker(_StackerCfg())
        ref = _synthetic_image(128, 256, seed=1)
        shifted = _simulate_stage_motion(ref, stage_dx_px=200, stage_dy_px=0)
        result = stacker.stack([ref, shifted])
        assert result.composite.shape == ref.shape
        # Rejection writes (0.0, 0.0) to the shifts list per _register_frames
        assert len(result.registration_shifts) == 1
        assert result.registration_shifts[0] == (0.0, 0.0)

    def test_prior_recovers_large_shift(self):
        """
        With a +200 px prior matching the simulated stage motion, the
        stacker pre-warps the shifted frame and finds only a sub-pixel
        residual. The recorded shift should match the prior closely.
        """
        stacker = FocusStacker(_StackerCfg())
        ref = _synthetic_image(128, 256, seed=2)
        shifted = _simulate_stage_motion(ref, stage_dx_px=200, stage_dy_px=0)

        # Prior says: stage moved +200 px in X between frame 0 and frame 1.
        # The grouper computes priors with this exact sign convention.
        prior = [(0.0, 0.0), (200.0, 0.0)]
        result = stacker.stack([ref, shifted], expected_shifts_px=prior)

        assert result.composite.shape == ref.shape
        # Continuous-mode path records shifts for every frame including frame 0
        assert len(result.registration_shifts) == 2
        assert result.registration_shifts[0] == (0.0, 0.0)
        # Frame 1's recorded shift should be close to (200, 0): prior + small residual
        dx, dy = result.registration_shifts[1]
        assert abs(dx - 200.0) < 5.0
        assert abs(dy) < 5.0

    def test_prior_length_mismatch_falls_back_to_no_prior(self):
        """A wrong-length prior should be ignored without raising."""
        stacker = FocusStacker(_StackerCfg())
        ref = _synthetic_image(128, 256, seed=3)
        f1 = _synthetic_image(128, 256, seed=4)
        result = stacker.stack([ref, f1], expected_shifts_px=[(0.0, 0.0)])
        assert result.composite.shape == ref.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
