"""
Simulated UART
===============
No-op replacement for pyserial. Logs TMC2209 configuration commands
for debugging without requiring any serial hardware.

Used only at startup to configure TMC2209 driver parameters
(microstep resolution, motor current, StallGuard threshold).
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("uart.sim")


class SimUART:
    """Simulated UART serial connection for TMC2209 communication."""

    def __init__(self, port: str = "/dev/ttyTHS1", baudrate: int = 115200) -> None:
        """
        Initialize simulated serial connection.

        Parameters
        ----------
        port : str
            Serial port path (logged but not opened).
        baudrate : int
            Baud rate (logged but not used).
        """
        self._port = port
        self._baudrate = baudrate
        self._is_open = False
        self._drivers: dict[int, dict] = {}
        logger.info("SimUART initialized: port=%s, baudrate=%d", port, baudrate)

    def open(self) -> None:
        """Open the serial connection. No-op in simulation."""
        self._is_open = True
        logger.debug("Serial port %s opened (simulated)", self._port)

    def close(self) -> None:
        """Close the serial connection. No-op in simulation."""
        self._is_open = False
        logger.debug("Serial port %s closed (simulated)", self._port)

    @property
    def is_open(self) -> bool:
        return self._is_open

    def configure_driver(
        self,
        address: int,
        microsteps: int = 256,
        current_ma: int = 700,
        stallguard_threshold: int = 50,
    ) -> None:
        """
        Configure a TMC2209 driver via UART.

        In simulation, stores the configuration and logs it.
        In real mode, this sends UART register writes to the driver.

        Parameters
        ----------
        address : int
            TMC2209 UART address (0x00, 0x01, or 0x02 for X, Y, Z).
        microsteps : int
            Microstep resolution (e.g. 256).
        current_ma : int
            Motor current in milliamps.
        stallguard_threshold : int
            StallGuard sensitivity value.
        """
        self._drivers[address] = {
            "microsteps": microsteps,
            "current_ma": current_ma,
            "stallguard_threshold": stallguard_threshold,
        }
        logger.info(
            "TMC2209 [addr=0x%02X] configured: microsteps=%d, current=%dmA, "
            "stallguard=%d",
            address, microsteps, current_ma, stallguard_threshold,
        )

    def read_stallguard(self, address: int) -> int:
        """
        Read StallGuard value from a TMC2209 driver.
        In simulation, always returns a safe (non-stalled) value.

        Parameters
        ----------
        address : int
            TMC2209 UART address.

        Returns
        -------
        int
            StallGuard load value. Higher = more load. 0 = no load.
        """
        logger.debug("StallGuard read from addr 0x%02X: returning 0 (simulated)", address)
        return 0

    def get_driver_config(self, address: int) -> dict | None:
        """Return stored configuration for a driver address, or None."""
        return self._drivers.get(address)

    def write_register(self, address: int, register: int, value: int) -> None:
        """
        Write a raw register value to a TMC2209.
        No-op in simulation; logged for debugging.
        """
        logger.debug(
            "UART write: addr=0x%02X, reg=0x%02X, value=0x%08X",
            address, register, value,
        )

    def read_register(self, address: int, register: int) -> int:
        """
        Read a raw register value from a TMC2209.
        Returns 0 in simulation.
        """
        logger.debug(
            "UART read: addr=0x%02X, reg=0x%02X → returning 0 (simulated)",
            address, register,
        )
        return 0
