"""
Continuous Scan Controller
==========================
Orchestrates continuous-motion scanning: drives X at constant velocity
along each serpentine line, oscillates Z through frames_per_field
discrete depths, fires the camera at each depth, position-tags every
frame, groups them into virtual fields, and hands off complete groups
for stacking.

This is the continuous-mode peer of CaptureSequencer. Both can coexist
under a single ScanPipeline by selecting at runtime via
`config.continuous_scan.enabled`.

Status: Chunk 1 skeleton. The orchestration logic is complete, but it
depends on motor and camera primitives that don't exist yet on the
existing sim or stub backends. Specifically, the motor needs:

    start_constant_velocity(axis, velocity_steps_per_sec)
    stop_axis(axis)
    move_to_blocking(axis, target_steps)        # already conceptually exists
    read_position_snapshot() -> (x_steps, y_steps, z_steps, timestamp)

…and the camera needs a synchronous `capture_one() -> np.ndarray` (or
equivalent triggered-capture API). Those land in Chunk 2 for sim and
Chunk 3 for real hardware.

The controller is written so that as soon as those primitives exist, it
runs end-to-end without modification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from cap.common.dataclasses import (
    FocusMapResult, ScanProgress, ScanRegion,
)
from cap.common.logging_setup import get_logger
from cap.layer1_hardware.z_cycle_generator import (
    depth_index_sequence, generate_cycle_targets,
)
from cap.layer2_acquisition.frame_grouper import FrameGrouper, VirtualField
from cap.layer2_acquisition.frame_tagger import (
    FrameTagger, PositionSnapshot,
    x_steps_to_um, y_steps_to_um,
    x_um_to_steps, y_um_to_steps, z_um_to_steps,
)

logger = get_logger("cont_scan")


# ---------------------------------------------------------------------------
# Scan-line layout
# ---------------------------------------------------------------------------

@dataclass
class ScanLineSpec:
    """
    One serpentine scan line: where X starts, where it ends, what Y
    row it covers, and which direction it travels.
    """
    line_index: int
    y_um: float
    x_start_um: float
    """Starting X position for this line, in travel direction."""
    x_end_um: float
    """Ending X position. May be < x_start_um for reverse-direction lines."""
    direction: int
    """+1 if x_end_um > x_start_um, -1 otherwise."""

    @property
    def length_um(self) -> float:
        return abs(self.x_end_um - self.x_start_um)


def plan_scan_lines(
    region: ScanRegion,
    fov_width_um: float,
    fov_height_um: float,
    x_steps_per_mm: float,
    y_steps_per_mm: float,
    serpentine: bool = True,
) -> list[ScanLineSpec]:
    """
    Convert a ScanRegion into a sequence of scan lines.

    Uses the existing region.field_positions (which the polygon-grid
    generator already produced for the stop-and-go path) to determine
    the X and Y extent. Each unique Y becomes one scan line; the X
    range is the span of fields at that Y. Direction alternates if
    serpentine is enabled.
    """
    if not region.field_positions:
        return []

    by_y_steps: dict[int, list[int]] = {}
    for fx_steps, fy_steps in region.field_positions:
        by_y_steps.setdefault(fy_steps, []).append(fx_steps)

    lines: list[ScanLineSpec] = []
    direction = 1
    for line_idx, fy_steps in enumerate(sorted(by_y_steps.keys())):
        xs_sorted = sorted(by_y_steps[fy_steps])
        x_min_um = x_steps_to_um(xs_sorted[0], x_steps_per_mm) - fov_width_um / 2.0
        x_max_um = x_steps_to_um(xs_sorted[-1], x_steps_per_mm) + fov_width_um / 2.0
        y_um = y_steps_to_um(fy_steps, y_steps_per_mm)

        if direction == 1:
            x_start_um, x_end_um = x_min_um, x_max_um
        else:
            x_start_um, x_end_um = x_max_um, x_min_um

        lines.append(ScanLineSpec(
            line_index=line_idx,
            y_um=y_um,
            x_start_um=x_start_um,
            x_end_um=x_end_um,
            direction=direction,
        ))
        if serpentine:
            direction = -direction

    return lines


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ContinuousScanController:
    """
    Continuous-mode scan orchestrator.

    Required motor interface (duck-typed):
        start_constant_velocity(axis: str, velocity_steps_per_sec: float) -> None
        stop_axis(axis: str) -> None
        move_to_blocking(axis: str, target_steps: int) -> None
        read_position_snapshot() -> PositionSnapshot
        emergency_stop() -> None

    Required camera interface (duck-typed):
        capture_one() -> np.ndarray   # blocks until one frame is captured

    `motor` and `camera` may be sim implementations; the controller
    doesn't care.
    """

    def __init__(
        self,
        config: object,
        motor_controller: object,
        camera_interface: object,
        focus_map: Optional[FocusMapResult],
        scan_region: ScanRegion,
        slide_id: int = 0,
    ) -> None:
        self._config = config
        self._motor = motor_controller
        self._camera = camera_interface
        self._focus_map = focus_map
        self._region = scan_region
        self._slide_id = slide_id

        cs = config.continuous_scan
        cam = config.camera
        motor_cfg = config.motor

        self._frames_per_field = cs.frames_per_field
        self._target_fps_per_field = cs.target_fields_per_second
        self._z_range_um = cs.z_cycle_range_um
        self._z_waveform = cs.z_cycle_waveform
        self._z_tracking = cs.z_tracking_enabled
        self._brightness_check_every = max(1, cs.brightness_check_every_n_frames)

        self._fov_w_um = cam.fov_width_mm * 1000.0
        self._fov_h_um = cam.fov_height_mm * 1000.0

        self._x_steps_per_mm = motor_cfg.x_steps_per_mm
        self._y_steps_per_mm = motor_cfg.y_steps_per_mm
        self._z_step_size_um = motor_cfg.z_step_size_microns

        # Stage velocity in steps/sec. Stage moves one FOV-width per
        # (1/target_fields_per_second) seconds.
        x_velocity_um_per_sec = self._fov_w_um * self._target_fps_per_field
        self._x_velocity_steps_per_sec = (
            x_velocity_um_per_sec * self._x_steps_per_mm / 1000.0
        )

        # Frame period: how often a capture must fire to keep the per-FOV
        # frame budget on schedule. The controller sleeps between captures
        # to maintain this. On real hardware where the camera self-paces
        # at fps, the sleep collapses to ~0 — so the same code path works
        # for both sim and real.
        self._frame_period_sec = (
            1.0 / (self._target_fps_per_field * self._frames_per_field)
        )

        # Deadline state — set per scan line in _scan_one_line()
        self._line_start_time: Optional[float] = None
        self._frame_index_in_line: int = 0

        self._tagger = FrameTagger(
            slide_id=slide_id,
            x_steps_per_mm=self._x_steps_per_mm,
            y_steps_per_mm=self._y_steps_per_mm,
            z_step_size_microns=self._z_step_size_um,
        )
        self._grouper = FrameGrouper(
            fov_width_um=self._fov_w_um,
            fov_height_um=self._fov_h_um,
            frames_per_field=self._frames_per_field,
            x_steps_per_mm=self._x_steps_per_mm,
            y_steps_per_mm=self._y_steps_per_mm,
        )

        self._is_running = False
        self._is_paused = False
        self._stop_requested = False

        # Diagnostics
        self._fields_complete = 0
        self._fields_partial = 0
        self._start_time: Optional[float] = None

        logger.info(
            "ContinuousScanController initialized: %d fields planned, "
            "fov=%.2f×%.2f μm, frames/field=%d, target_fps_per_field=%.1f, "
            "x_velocity=%.1f steps/sec",
            len(scan_region.field_positions), self._fov_w_um, self._fov_h_um,
            self._frames_per_field, self._target_fps_per_field,
            self._x_velocity_steps_per_sec,
        )

    # ----- Public lifecycle -----

    @property
    def is_running(self) -> bool:
        return self._is_running

    def pause(self) -> None:
        self._is_paused = True
        logger.info("Continuous scan pause requested")

    def resume(self) -> None:
        self._is_paused = False
        logger.info("Continuous scan resume requested")

    def stop(self) -> None:
        self._stop_requested = True
        logger.info("Continuous scan stop requested")

    def run(
        self,
        progress_callback: Optional[Callable[[ScanProgress], None]] = None,
        field_complete_callback: Optional[Callable[[VirtualField], None]] = None,
    ) -> dict:
        """
        Execute the full continuous scan.

        progress_callback is invoked per scan line (not per frame, to
        avoid drowning the UI thread).

        field_complete_callback is invoked synchronously each time a
        VirtualField is complete; this is where the caller hooks the
        focus stacker (typically by pushing onto the existing
        stack_queue used by ScanPipeline).
        """
        self._is_running = True
        self._stop_requested = False
        self._is_paused = False
        self._start_time = time.monotonic()

        lines = plan_scan_lines(
            self._region,
            fov_width_um=self._fov_w_um,
            fov_height_um=self._fov_h_um,
            x_steps_per_mm=self._x_steps_per_mm,
            y_steps_per_mm=self._y_steps_per_mm,
            serpentine=self._config.scan.serpentine_enabled,
        )
        total_fields = len(self._region.field_positions)
        logger.info("Continuous scan starting: %d lines, %d fields", len(lines), total_fields)

        try:
            for spec in lines:
                if self._stop_requested:
                    break
                self._wait_while_paused()
                if self._stop_requested:
                    break

                self._scan_one_line(spec, field_complete_callback)

                if progress_callback:
                    elapsed = time.monotonic() - self._start_time
                    fields_done = self._fields_complete + self._fields_partial
                    fields_left = max(0, total_fields - fields_done)
                    eta = (elapsed / max(1, fields_done)) * fields_left if fields_done else 0.0
                    # current_x/current_y carry the last commanded line position
                    # for the UI; precise tracking is a Chunk 2 concern.
                    progress_callback(ScanProgress(
                        fields_completed=self._fields_complete,
                        fields_total=total_fields,
                        current_x=x_um_to_steps(spec.x_end_um, self._x_steps_per_mm),
                        current_y=y_um_to_steps(spec.y_um, self._y_steps_per_mm),
                        eta_seconds=eta,
                        current_field_status=f"line {spec.line_index} done",
                    ))

            # Drain any partials buffered at the end of the last line.
            for vf in self._grouper.flush_line():
                self._dispatch(vf, field_complete_callback)

        finally:
            self._safe_stop_axes()
            self._is_running = False

        duration = time.monotonic() - self._start_time
        logger.info(
            "Continuous scan complete: %d complete fields, %d partial, %.1f sec",
            self._fields_complete, self._fields_partial, duration,
        )

        return {
            "fields_completed": self._fields_complete,
            "fields_partial": self._fields_partial,
            "fields_total": total_fields,
            "duration_sec": duration,
            "grouper_stats": self._grouper.stats(),
            "frames_tagged": self._tagger.frames_tagged,
        }

    # ----- Per-line orchestration -----

    def _scan_one_line(
        self,
        spec: ScanLineSpec,
        field_complete_callback: Optional[Callable[[VirtualField], None]],
    ) -> None:
        """
        Execute one scan line: move to start, kick X to constant velocity,
        oscillate Z while X traverses, capture frames, group, dispatch.
        """
        # Carry over partials from previous line (pause/retry decisions
        # are the caller's; we just emit them so they aren't lost).
        for vf in self._grouper.begin_scan_line(
            scan_line=spec.line_index,
            scan_direction=spec.direction,
            line_origin_x_um=spec.x_start_um,
            line_y_um=spec.y_um,
        ):
            self._dispatch(vf, field_complete_callback)

        # Position to line start
        self._motor.move_to_blocking("y", y_um_to_steps(spec.y_um, self._y_steps_per_mm))
        self._motor.move_to_blocking("x", x_um_to_steps(spec.x_start_um, self._x_steps_per_mm))

        # Cycles per line: enough to cover the full line length, plus a small
        # safety margin in case the stage overshoots slightly at line end.
        n_cycles = max(1, int(spec.length_um / self._fov_w_um) + 1)

        # Coefficients used to track focus surface, if enabled
        coeffs = (
            list(self._focus_map.surface_coefficients)
            if (self._z_tracking and self._focus_map is not None)
            else None
        )

        # Begin constant-velocity X motion
        x_vel = spec.direction * self._x_velocity_steps_per_sec
        self._motor.start_constant_velocity("x", x_vel)
        self._line_start_time = time.monotonic()
        self._frame_index_in_line = 0
        try:
            for cycle_idx in range(n_cycles):
                if self._stop_requested:
                    break
                self._wait_while_paused()
                if self._stop_requested:
                    break

                # Use the latest motor X to center the cycle's Z surface evaluation
                snap = self._motor.read_position_snapshot()
                x_um_now = x_steps_to_um(snap.x_steps, self._x_steps_per_mm)
                y_um_now = y_steps_to_um(snap.y_steps, self._y_steps_per_mm)

                targets = generate_cycle_targets(
                    x_um=x_um_now,
                    y_um=y_um_now,
                    coefficients=coeffs,
                    n_depths=self._frames_per_field,
                    total_range_um=self._z_range_um,
                    cycle_index=cycle_idx,
                    waveform=self._z_waveform,
                    fallback_z_center_um=0.0,
                )

                for depth_idx, z_um_target in targets:
                    if self._stop_requested:
                        break
                    self._capture_and_route(
                        spec, depth_idx, z_um_target, field_complete_callback,
                    )

        finally:
            self._motor.stop_axis("x")

        # End-of-line cleanup: any stragglers buffered as partials get flushed
        # at the start of the next line (begin_scan_line returns them).

    def _capture_and_route(
        self,
        spec: ScanLineSpec,
        depth_idx: int,
        z_um_target: float,
        field_complete_callback: Optional[Callable[[VirtualField], None]],
    ) -> None:
        """One depth: drive Z, wait for the camera's frame slot, capture,
        tag, feed to grouper, dispatch."""
        z_steps_target = z_um_to_steps(z_um_target, self._z_step_size_um)
        # Z is fast-moved blocking inside the cycle. The stage X keeps
        # moving during this — that's the whole point.
        self._motor.move_to_blocking("z", z_steps_target)

        # Hold until the camera's next frame slot. On real hardware the
        # camera blocks on its own FPS; in sim, this is what enforces
        # the X-velocity-vs-frame-count calibration that keeps the
        # grouper producing one virtual field per FOV-width.
        if self._line_start_time is not None:
            deadline = (
                self._line_start_time
                + (self._frame_index_in_line + 1) * self._frame_period_sec
            )
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self._frame_index_in_line += 1

        bayer = self._camera.capture_one()
        snap = self._motor.read_position_snapshot()

        frame = self._tagger.tag(
            bayer_data=bayer,
            snapshot=snap,
            depth_index=depth_idx,
            scan_line=spec.line_index,
            scan_direction=spec.direction,
        )
        x_um = x_steps_to_um(snap.x_steps, self._x_steps_per_mm)
        y_um = y_steps_to_um(snap.y_steps, self._y_steps_per_mm)
        vf = self._grouper.feed(frame, x_um, y_um)
        if vf is not None:
            self._dispatch(vf, field_complete_callback)

    def _dispatch(
        self,
        vf: VirtualField,
        field_complete_callback: Optional[Callable[[VirtualField], None]],
    ) -> None:
        """Hand off a complete or partial VirtualField to the caller."""
        if vf.is_complete:
            self._fields_complete += 1
        else:
            self._fields_partial += 1
            logger.warning(
                "Partial virtual field at line %d index %d: %d/%d frames",
                vf.scan_line, vf.field_index_in_line, vf.n_frames, self._frames_per_field,
            )
        if field_complete_callback is not None:
            field_complete_callback(vf)

    # ----- Helpers -----

    def _wait_while_paused(self) -> None:
        while self._is_paused and not self._stop_requested:
            time.sleep(0.05)

    def _safe_stop_axes(self) -> None:
        """Best-effort stop on all axes, swallowing errors during shutdown."""
        for axis in ("x", "y", "z"):
            try:
                self._motor.stop_axis(axis)
            except Exception as e:
                logger.debug("stop_axis(%s) raised during shutdown: %s", axis, e)
