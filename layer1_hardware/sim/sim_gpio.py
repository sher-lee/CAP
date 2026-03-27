"""
Simulated GPIO
===============
No-op replacement for Jetson.GPIO. Logs all pin state changes
for debugging without requiring any hardware.

Mimics the Jetson.GPIO API surface used by the motor controller
and camera trigger.
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("gpio.sim")

# Constants matching Jetson.GPIO
BCM = "BCM"
BOARD = "BOARD"
OUT = "OUT"
IN = "IN"
HIGH = 1
LOW = 0


class SimGPIO:
    """Simulated GPIO interface."""

    def __init__(self) -> None:
        self._mode: str | None = None
        self._pins: dict[int, dict] = {}
        logger.info("SimGPIO initialized")

    def setmode(self, mode: str) -> None:
        """Set pin numbering mode (BCM or BOARD)."""
        self._mode = mode
        logger.debug("GPIO mode set to %s", mode)

    def setup(self, pin: int, direction: str) -> None:
        """
        Configure a pin as input or output.

        Parameters
        ----------
        pin : int
            Pin number.
        direction : str
            OUT or IN.
        """
        self._pins[pin] = {"direction": direction, "state": LOW}
        logger.debug("Pin %d configured as %s", pin, direction)

    def output(self, pin: int, state: int) -> None:
        """
        Set output pin state (HIGH or LOW).

        Parameters
        ----------
        pin : int
            Pin number (must be configured as OUT).
        state : int
            HIGH (1) or LOW (0).
        """
        if pin in self._pins:
            self._pins[pin]["state"] = state
        logger.debug("Pin %d → %s", pin, "HIGH" if state else "LOW")

    def input(self, pin: int) -> int:
        """
        Read input pin state.

        Returns
        -------
        int
            HIGH or LOW. Always returns LOW in simulation.
        """
        state = self._pins.get(pin, {}).get("state", LOW)
        return state

    def cleanup(self) -> None:
        """Release all pins."""
        pin_count = len(self._pins)
        self._pins.clear()
        self._mode = None
        logger.info("GPIO cleanup: released %d pins", pin_count)

    def pulse(self, pin: int, duration_us: float = 10) -> None:
        """
        Send a single HIGH→LOW pulse. Used for motor step signals
        and camera hardware triggers.

        Parameters
        ----------
        pin : int
            Pin number.
        duration_us : float
            Pulse duration in microseconds (not enforced in simulation).
        """
        logger.debug("Pulse on pin %d (%.1f µs)", pin, duration_us)


# Module-level singleton so it can be used like the real Jetson.GPIO module
_instance = SimGPIO()
setmode = _instance.setmode
setup = _instance.setup
output = _instance.output
input = _instance.input
cleanup = _instance.cleanup
pulse = _instance.pulse
