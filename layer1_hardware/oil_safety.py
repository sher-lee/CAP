"""
Oil Safety Monitor
====================
Monitors captured frames for unexpected brightness changes that
indicate the objective has left the oil-covered zone. This is a
safety-only mechanism — the technician-drawn polygon is the primary
scan boundary. The oil monitor watches for failures of that boundary.

Works with both real and simulated camera backends since it only
analyzes frame pixel data.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("oil")


class OilSafetyMonitor:
    """
    Monitors frame brightness for unexpected oil loss during scanning.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "scan_region"):
            self._threshold = config.scan_region.oil_safety_brightness_threshold
        else:
            sr = config.get("scan_region", {})
            self._threshold = sr.get("oil_safety_brightness_threshold", 0.3)

        self._monitoring = False
        self._baseline: Optional[float] = None
        self._history: deque[float] = deque(maxlen=20)
        self._warning_active = False

        logger.info(
            "OilSafetyMonitor initialized: brightness_threshold=%.2f",
            self._threshold,
        )

    def start_monitoring(self) -> None:
        """Begin monitoring. Resets baseline and history."""
        self._monitoring = True
        self._baseline = None
        self._history.clear()
        self._warning_active = False
        logger.debug("Oil monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
        logger.debug("Oil monitoring stopped")

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring

    @property
    def is_warning_active(self) -> bool:
        return self._warning_active

    def check_frame_brightness(self, frame_data: np.ndarray) -> bool:
        """
        Check a captured frame for unexpected brightness deviation.

        The first few frames establish a baseline mean brightness.
        Subsequent frames are compared against the baseline. A sharp
        increase in brightness (exceeding the threshold fraction)
        indicates the objective may have left the oil zone.

        Parameters
        ----------
        frame_data : np.ndarray
            Raw or processed frame data. Can be any shape — only the
            mean pixel value is used.

        Returns
        -------
        bool
            True if brightness is within expected range (OK).
            False if brightness deviation detected (WARNING).
        """
        if not self._monitoring:
            return True

        # Compute mean brightness
        mean_brightness = float(np.mean(frame_data))
        self._history.append(mean_brightness)

        # Establish baseline from first 5 frames
        if len(self._history) < 5:
            self._baseline = np.mean(list(self._history))
            return True

        if self._baseline is None or self._baseline == 0:
            self._baseline = mean_brightness
            return True

        # Check for deviation
        deviation = abs(mean_brightness - self._baseline) / self._baseline

        if deviation > self._threshold:
            self._warning_active = True
            logger.warning(
                "Oil safety WARNING: brightness deviation %.1f%% "
                "(current=%.1f, baseline=%.1f, threshold=%.0f%%)",
                deviation * 100, mean_brightness, self._baseline,
                self._threshold * 100,
            )
            return False

        # Update rolling baseline (slow adaptation to gradual changes)
        self._baseline = 0.95 * self._baseline + 0.05 * mean_brightness
        self._warning_active = False
        return True

    def clear_warning(self) -> None:
        """Clear the warning state (after technician acknowledges)."""
        self._warning_active = False
        # Reset baseline to current level
        if self._history:
            self._baseline = float(np.mean(list(self._history)[-5:]))
        logger.info("Oil warning cleared, baseline reset")

    def check_oil_present(self) -> bool:
        """
        Simple check: is oil likely present based on recent frames?
        Returns True if no warning is active.
        """
        return not self._warning_active

    def get_brightness_stats(self) -> dict:
        """Get current brightness statistics for debugging."""
        return {
            "baseline": self._baseline,
            "current": self._history[-1] if self._history else None,
            "history_len": len(self._history),
            "warning_active": self._warning_active,
        }

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._monitoring = False
        logger.info("OilSafetyMonitor shut down")
