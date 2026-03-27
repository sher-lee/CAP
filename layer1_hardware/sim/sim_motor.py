"""
Simulated Motor Controller
============================
Drop-in replacement for the real MotorController. Tracks X, Y, Z
positions in memory. No GPIO, no real motors. Enforces software
boundary limits and optionally simulates movement delay.

Implements the same public interface as the real MotorController
so all upstream code (ScanRegionManager, CaptureSequencer, UI)
works identically in simulation and real modes.
"""

from __future__ import annotations

import time
from typing import Optional

from cap.common.logging_setup import get_logger

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

        # Current position (microsteps)
        self._x: int = 0
        self._y: int = 0
        self._z: int = 0

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
        return (self._x, self._y, self._z)

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

    # ----- Internal helpers -----

    def _get_axis(self, axis: str) -> int:
        if axis == "x":
            return self._x
        elif axis == "y":
            return self._y
        elif axis == "z":
            return self._z
        else:
            raise ValueError(f"Invalid axis: '{axis}'. Must be 'x', 'y', or 'z'.")

    def _set_axis(self, axis: str, value: int) -> None:
        if axis == "x":
            self._x = value
        elif axis == "y":
            self._y = value
        elif axis == "z":
            self._z = value

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
