"""
Simulated Safety System
========================
Drop-in replacement for the real SafetySystem. Enforces software
boundary limits but has no torque monitoring or physical watchdog.
Always reports safe conditions.
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("safety.sim")


class SimSafetySystem:
    """Simulated safety system for Windows development."""

    def __init__(self, config: dict | object) -> None:
        self._active = True
        logger.info("SimSafetySystem initialized (all checks pass in simulation)")

    def start_watchdog(self) -> None:
        """Start the safety watchdog thread. No-op in simulation."""
        logger.debug("Watchdog start requested (no-op in simulation)")

    def stop_watchdog(self) -> None:
        """Stop the safety watchdog thread. No-op in simulation."""
        logger.debug("Watchdog stop requested (no-op in simulation)")

    def check_torque(self, axis: str) -> bool:
        """
        Check torque/resistance on specified axis.
        Always returns True (safe) in simulation.
        """
        return True

    def check_all_axes(self) -> dict[str, bool]:
        """
        Check safety status of all axes.
        Returns dict of axis → safe boolean.
        """
        return {"x": True, "y": True, "z": True}

    def is_safe(self) -> bool:
        """Whether the system is in a safe state. Always True in simulation."""
        return True

    @property
    def is_active(self) -> bool:
        return self._active

    def shutdown(self) -> None:
        """Clean shutdown of the safety system."""
        self._active = False
        logger.info("SimSafetySystem shut down")
