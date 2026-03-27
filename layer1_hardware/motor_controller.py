"""
Motor Controller (Real Hardware)
===================================
Interface for controlling NEMA 11 stepper motors via TMC2209
drivers on the Jetson Orin Nano.

THIS IS A STUB — the implementation will be filled in during
Phase 10 (hardware bring-up) when real GPIO and serial hardware
are available. The interface matches SimMotorController exactly.

For Windows development, use SimMotorController via the
backend selector (hardware_mode: simulation).
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("motor")


class MotorController:
    """
    Real motor controller for Jetson + TMC2209 + NEMA 11.

    Stub implementation — raises NotImplementedError for all methods.
    Will be implemented during Phase 10 hardware bring-up.
    """

    def __init__(self, config: object) -> None:
        raise NotImplementedError(
            "Real MotorController requires Jetson hardware. "
            "Set hardware_mode to 'simulation' for Windows development."
        )

    def move_to(self, axis: str, position: int) -> None:
        raise NotImplementedError

    def move_relative(self, axis: str, delta: int) -> None:
        raise NotImplementedError

    def move_xyz(self, x: int, y: int, z: int) -> None:
        raise NotImplementedError

    def get_position(self, axis: str) -> int:
        raise NotImplementedError

    def get_position_xyz(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def home(self, axis: str) -> None:
        raise NotImplementedError

    def home_all(self) -> None:
        raise NotImplementedError

    @property
    def is_homed(self) -> bool:
        raise NotImplementedError

    def set_speed(self, axis: str, speed: int) -> None:
        raise NotImplementedError

    def emergency_stop(self) -> None:
        raise NotImplementedError

    def clear_estop(self) -> None:
        raise NotImplementedError

    @property
    def is_emergency_stopped(self) -> bool:
        raise NotImplementedError

    def wait_settle(self) -> None:
        raise NotImplementedError
