"""
Simulated Oil Safety Monitor
==============================
Drop-in replacement for the real OilSafetyMonitor.
Always reports oil present. No brightness monitoring.
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("oil.sim")


class SimOilSafetyMonitor:
    """Simulated oil safety monitor for Windows development."""

    def __init__(self, config: dict | object) -> None:
        self._monitoring = False
        self._oil_present = True
        logger.info("SimOilSafetyMonitor initialized (always reports oil present)")

    def start_monitoring(self) -> None:
        """Start background brightness monitoring. No-op in simulation."""
        self._monitoring = True
        logger.debug("Oil monitoring started (no-op in simulation)")

    def stop_monitoring(self) -> None:
        """Stop background brightness monitoring. No-op in simulation."""
        self._monitoring = False
        logger.debug("Oil monitoring stopped")

    def check_oil_present(self) -> bool:
        """
        Check if the objective is in oil contact.
        Always returns True in simulation.
        """
        return self._oil_present

    def check_frame_brightness(self, frame_data) -> bool:
        """
        Check a captured frame for unexpected brightness deviation.
        Always returns True (brightness OK) in simulation.

        Parameters
        ----------
        frame_data : np.ndarray
            Raw or processed frame data (unused in sim).

        Returns
        -------
        bool
            True if brightness is within expected range.
        """
        return True

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring

    def shutdown(self) -> None:
        """Clean shutdown of the monitor."""
        self._monitoring = False
        logger.info("SimOilSafetyMonitor shut down")
