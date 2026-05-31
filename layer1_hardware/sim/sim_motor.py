"""
Simulated Motor Controller
============================
Drop-in replacement for the real MotorController. Tracks X, Y, Z
positions in memory. No GPIO, no real motors. Enforces software
boundary limits and optionally simulates movement delay.

Implements the same public interface as the real MotorController so
all upstream code (ScanRegionManager, CaptureSequencer, UI) works
identically in simulation and real modes.

Also implements the continuous-scan primitives that
ContinuousScanController requires:

    start_constant_velocity(axis, velocity_steps_per_sec)
    stop_axis(axis)
    move_to_blocking(axis, target_steps)
    read_position_snapshot()

Time-aware position model: each axis tracks (position_at_anchor,
velocity_steps_per_sec, anchor_time). When velocity != 0, reads
compute the live position as:

    pos_now = pos_anchor + velocity * (time.monotonic() - anchor_time)

so that X keeps advancing while a Z move_to_blocking sleeps, exactly
as it would on real hardware.
"""

from __future__ import annotations

import time
from typing import Optional

from cap.common.logging_setup import get_logger
from cap.layer2_acquisition.frame_tagger import PositionSnapshot

logger = get_logger("motor.sim")


class SimMotorController:
    """Simulated motor controller for Windows development."""

    def __init__(self, config: dict | object) -> None:
        """
        Initialize with motor config section.

        Parameters
        ----------
        config : dict or CAPConfig
            If a CAPConfig object, reads from config.motor.
            If a dict, reads motor keys directly.
        """
        if hasattr(config, "motor"):
            mc = config.motor
            self._x_min = mc.x_min
            self._x_max = mc.x_max
            self._y_min = mc.y_min
            self._y_max = mc.y_max
            self._z_min = mc.z_min
            self._z_max = mc.z_max
            self._microsteps = mc.microsteps
            self._speed = mc.speed
            self._settle_delay_ms = mc.settle_delay_ms
            self._sim_delay_ms = config.sim.motor_delay_ms if hasattr(config, "sim") else 10
        else:
            motor = config.get("motor", {})
            self._x_min = motor.get("x_min", 0)
            self._x_max = motor.get("x_max", 100000)
            self._y_min = motor.get("y_min", 0)
            self._y_max = motor.get("y_max", 100000)
            self._z_min = motor.get("z_min", 0)
            self._z_max = motor.get("z_max", 10000)
            self._microsteps = motor.get("microsteps", 256)
            self._speed = motor.get("speed", 1000)
            self._settle_delay_ms = motor.get("settle_delay_ms", 100)
            sim = config.get("sim", {})
            self._sim_delay_ms = sim.get("motor_delay_ms", 10)

        # Per-axis time-aware position. Each axis stores (anchor_pos,
        # anchor_time, velocity). When velocity is 0 the live position is
        # just anchor_pos. When velocity != 0, live position is
        # anchor_pos + velocity * (now - anchor_time). Discrete moves
        # update the anchor and zero the velocity.
        now = time.monotonic()
        self._anchor_pos: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._anchor_time: dict[str, float] = {"x": now, "y": now, "z": now}
        self._velocity: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}

        # State
        self._homed: bool = False
        self._emergency_stopped: bool = False

        logger.info(
            "SimMotorController initialized: bounds X[%d,%d] Y[%d,%d] Z[%d,%d], "
            "sim_delay=%dms",
            self._x_min, self._x_max,
            self._y_min, self._y_max,
            self._z_min, self._z_max,
            self._sim_delay_ms,
        )

    # ----- Movement -----

    def move_to(self, axis: str, position: int) -> None:
        """
        Move specified axis to absolute position in microsteps.

        Parameters
        ----------
        axis : str
            "x", "y", or "z" (case-insensitive).
        position : int
            Target position in microsteps.

        Raises
        ------
        ValueError
            If axis is invalid or position is outside boundary limits.
        RuntimeError
            If emergency stop is active.
        """
        self._check_estop()
        axis = axis.lower()
        self._validate_position(axis, position)

        old_pos = self._get_axis(axis)
        self._simulate_delay(abs(position - old_pos))
        self._set_axis(axis, position)

        logger.debug("move_to(%s, %d) — was %d", axis, position, old_pos)

    def move_relative(self, axis: str, delta: int) -> None:
        """
        Move specified axis by relative delta in microsteps.

        Parameters
        ----------
        axis : str
            "x", "y", or "z".
        delta : int
            Relative movement in microsteps (positive or negative).
        """
        current = self._get_axis(axis.lower())
        self.move_to(axis, current + delta)

    def move_xyz(self, x: int, y: int, z: int) -> None:
        """
        Move all three axes to absolute positions.
        Convenience method for the capture sequencer.
        """
        self.move_to("x", x)
        self.move_to("y", y)
        self.move_to("z", z)

    # ----- Position -----

    def get_position(self, axis: str) -> int:
        """Return current position of specified axis in microsteps."""
        return self._get_axis(axis.lower())

    def get_position_xyz(self) -> tuple[int, int, int]:
        """Return current position of all axes as (x, y, z)."""
        return (self._get_axis("x"), self._get_axis("y"), self._get_axis("z"))

    # ----- Homing -----

    def home(self, axis: str) -> None:
        """
        Home specified axis. In simulation, instantly moves to 0.

        Parameters
        ----------
        axis : str
            "x", "y", or "z".
        """
        self._check_estop()
        axis = axis.lower()
        self._set_axis(axis, 0)
        logger.info("Homed axis %s", axis)

    def home_all(self) -> None:
        """Home all three axes in safe sequence (Z first, then X, then Y)."""
        self._check_estop()
        self.home("z")
        self.home("x")
        self.home("y")
        self._homed = True
        logger.info("All axes homed")

    @property
    def is_homed(self) -> bool:
        """Whether the home_all() routine has completed."""
        return self._homed

    # ----- Speed -----

    def set_speed(self, axis: str, speed: int) -> None:
        """Set motor speed in steps/sec. In simulation, stored but not used."""
        self._speed = speed
        logger.debug("Speed set to %d steps/sec for axis %s", speed, axis)

    # ----- Emergency stop -----

    def emergency_stop(self) -> None:
        """Immediately halt all axes. Must call clear_estop() to resume."""
        self._emergency_stopped = True
        logger.warning("EMERGENCY STOP activated")

    def clear_estop(self) -> None:
        """Clear emergency stop state. Requires re-homing before scanning."""
        self._emergency_stopped = False
        self._homed = False
        logger.info("Emergency stop cleared — re-homing required")

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    # ----- Settle delay -----

    def wait_settle(self) -> None:
        """
        Wait for the configured settle delay. In simulation, uses the
        sim motor delay (shorter) instead of the real settle delay.
        """
        delay_sec = self._sim_delay_ms / 1000.0
        if delay_sec > 0:
            time.sleep(delay_sec)

    # ----- Continuous-mode primitives -----

    def start_constant_velocity(self, axis: str, velocity_steps_per_sec: float) -> None:
        """
        Begin moving an axis at constant velocity (microsteps/sec).
        Returns immediately — subsequent reads compute live position.

        velocity_steps_per_sec may be negative (reverse direction). A value
        of 0 stops the axis (equivalent to stop_axis).
        """
        self._check_estop()
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis: '{axis}'")

        # Freeze current live position before changing velocity, so the
        # transition is continuous (no jump).
        live = self._live_position(axis)
        self._anchor_pos[axis] = live
        self._anchor_time[axis] = time.monotonic()
        self._velocity[axis] = float(velocity_steps_per_sec)

        logger.debug(
            "start_constant_velocity(%s, %.2f steps/sec) — anchor=%.2f",
            axis, velocity_steps_per_sec, live,
        )

    def stop_axis(self, axis: str) -> None:
        """Stop an axis at its current live position. Idempotent."""
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis: '{axis}'")
        live = self._live_position(axis)
        self._anchor_pos[axis] = live
        self._anchor_time[axis] = time.monotonic()
        self._velocity[axis] = 0.0
        logger.debug("stop_axis(%s) — frozen at %.2f", axis, live)

    def move_to_blocking(self, axis: str, target_steps: int) -> None:
        """
        Discrete blocking move to target position. Stops any constant-
        velocity motion on this axis first. Other axes' motion is
        unaffected. Identical to move_to() — provided as an explicit
        name for the continuous-scan controller's API.
        """
        self.move_to(axis, target_steps)

    def read_position_snapshot(self) -> PositionSnapshot:
        """
        Atomic snapshot of all three axes at the current instant.
        Used by the FrameTagger to bind motor state to a captured frame.
        """
        now = time.monotonic()
        return PositionSnapshot(
            x_steps=int(round(self._live_position("x", at_time=now))),
            y_steps=int(round(self._live_position("y", at_time=now))),
            z_steps=int(round(self._live_position("z", at_time=now))),
            timestamp=now,
        )

    # ----- Internal helpers -----

    def _live_position(self, axis: str, at_time: Optional[float] = None) -> float:
        """
        Compute the live position of an axis at `at_time` (default: now),
        accounting for any active constant-velocity motion. Returns
        microsteps as a float; integer rounding is the caller's choice.

        Boundary clamp: if the live position would exceed the axis bounds,
        the axis is auto-stopped at the bound. This mimics StallGuard
        kicking in on real hardware.
        """
        v = self._velocity[axis]
        if v == 0.0:
            return self._anchor_pos[axis]

        if at_time is None:
            at_time = time.monotonic()
        live = self._anchor_pos[axis] + v * (at_time - self._anchor_time[axis])

        bounds = {
            "x": (self._x_min, self._x_max),
            "y": (self._y_min, self._y_max),
            "z": (self._z_min, self._z_max),
        }
        lo, hi = bounds[axis]
        if live < lo:
            self._anchor_pos[axis] = float(lo)
            self._anchor_time[axis] = at_time
            self._velocity[axis] = 0.0
            logger.warning("Axis %s hit lower bound %d during constant-velocity motion", axis, lo)
            return float(lo)
        if live > hi:
            self._anchor_pos[axis] = float(hi)
            self._anchor_time[axis] = at_time
            self._velocity[axis] = 0.0
            logger.warning("Axis %s hit upper bound %d during constant-velocity motion", axis, hi)
            return float(hi)

        return live

    def _get_axis(self, axis: str) -> int:
        """Public reads return integer microsteps from the live position."""
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis: '{axis}'. Must be 'x', 'y', or 'z'.")
        return int(round(self._live_position(axis)))

    def _set_axis(self, axis: str, value: int) -> None:
        """Discrete set: anchors the axis at value and zeros its velocity."""
        if axis not in ("x", "y", "z"):
            return
        self._anchor_pos[axis] = float(value)
        self._anchor_time[axis] = time.monotonic()
        self._velocity[axis] = 0.0

    def _validate_position(self, axis: str, position: int) -> None:
        bounds = {
            "x": (self._x_min, self._x_max),
            "y": (self._y_min, self._y_max),
            "z": (self._z_min, self._z_max),
        }
        if axis not in bounds:
            raise ValueError(f"Invalid axis: '{axis}'")
        lo, hi = bounds[axis]
        if not (lo <= position <= hi):
            raise ValueError(
                f"Position {position} out of bounds for axis {axis}: [{lo}, {hi}]"
            )

    def _check_estop(self) -> None:
        if self._emergency_stopped:
            raise RuntimeError(
                "Cannot move: emergency stop is active. Call clear_estop() first."
            )

    def _simulate_delay(self, steps: int) -> None:
        """Optionally sleep to simulate motor movement time."""
        if self._sim_delay_ms > 0:
            time.sleep(self._sim_delay_ms / 1000.0)
