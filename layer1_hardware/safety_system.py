"""
Safety System (Real Hardware)
================================
Monitors motor torque, enforces boundaries, and provides
emergency stop capability via real TMC2209 StallGuard readings.

THIS IS A STUB — the implementation will be filled in during
Phase 10 (hardware bring-up). The interface matches SimSafetySystem.

For Windows development, use SimSafetySystem via the
backend selector (hardware_mode: simulation).
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("safety")


class SafetySystem:
    """
    Real safety system for Jetson + TMC2209.

    Stub implementation — raises NotImplementedError.
    Will be implemented during Phase 10 hardware bring-up.
    """

    def __init__(self, config: object) -> None:
        raise NotImplementedError(
            "Real SafetySystem requires Jetson hardware. "
            "Set hardware_mode to 'simulation' for Windows development."
        )

    def start_watchdog(self) -> None:
        raise NotImplementedError

    def stop_watchdog(self) -> None:
        raise NotImplementedError

    def check_torque(self, axis: str) -> bool:
        raise NotImplementedError

    def check_all_axes(self) -> dict[str, bool]:
        raise NotImplementedError

    def is_safe(self) -> bool:
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
