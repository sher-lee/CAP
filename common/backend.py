"""
CAP Backend Selector
====================
Reads the hardware_mode from configuration and returns the appropriate
module references for Layer 1 (hardware) and Layer 2 (acquisition).

All other layers (3–9) are hardware-independent and always use
their real implementations regardless of mode.

Usage:
    from cap.common.backend import get_backend
    backend = get_backend(config)
    motor = backend.motor_controller_class(config)
    camera = backend.camera_interface_class(config)

The backend object provides class references, not instances —
instantiation is the caller's responsibility so that constructor
arguments and lifecycle are controlled by the layer that owns them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Type

from cap.common.logging_setup import get_logger

if TYPE_CHECKING:
    pass  # Type hints for IDE support without circular imports

logger = get_logger("backend")


class HardwareMode:
    """Valid hardware_mode values."""
    SIMULATION = "simulation"
    REAL = "real"


@dataclass
class Backend:
    """
    Container for all hardware-dependent class references.
    Each field holds a class (not an instance) that implements
    the interface defined in the spec for that module.
    """
    mode: str

    # Layer 1: Hardware abstraction
    motor_controller_class: Type[Any]
    safety_system_class: Type[Any]
    preliminary_focus_class: Type[Any]
    per_field_autofocus_class: Type[Any]
    oil_safety_class: Type[Any]

    # Layer 2: Image acquisition
    camera_interface_class: Type[Any]

    @property
    def is_simulation(self) -> bool:
        return self.mode == HardwareMode.SIMULATION

    @property
    def is_real(self) -> bool:
        return self.mode == HardwareMode.REAL


def get_backend(config: dict) -> Backend:
    """
    Build a Backend from the application configuration.

    Parameters
    ----------
    config : dict
        The loaded cap_config.yaml as a dictionary. Must contain
        a top-level 'hardware_mode' key with value 'simulation' or 'real'.

    Returns
    -------
    Backend
        A Backend instance with the appropriate class references loaded.

    Raises
    ------
    ValueError
        If hardware_mode is not 'simulation' or 'real'.
    ImportError
        If real-hardware modules are requested but Jetson-specific
        packages (Jetson.GPIO, pyserial, harvesters) are not installed.
    """
    mode = config.get("hardware_mode", HardwareMode.SIMULATION)

    if mode == HardwareMode.SIMULATION:
        return _load_simulation_backend()
    elif mode == HardwareMode.REAL:
        return _load_real_backend()
    else:
        raise ValueError(
            f"Invalid hardware_mode: '{mode}'. "
            f"Must be '{HardwareMode.SIMULATION}' or '{HardwareMode.REAL}'."
        )


def _load_simulation_backend() -> Backend:
    """Load all simulation module classes."""
    logger.info("Loading SIMULATION backend (no hardware required)")

    from cap.layer1_hardware.sim.sim_motor import SimMotorController
    from cap.layer1_hardware.sim.sim_safety import SimSafetySystem
    from cap.layer1_hardware.sim.sim_focus import SimPreliminaryFocus
    from cap.layer1_hardware.sim.sim_autofocus import SimPerFieldAutofocus
    from cap.layer1_hardware.sim.sim_oil_safety import SimOilSafetyMonitor
    from cap.layer2_acquisition.sim.sim_camera import SimCameraInterface

    return Backend(
        mode=HardwareMode.SIMULATION,
        motor_controller_class=SimMotorController,
        safety_system_class=SimSafetySystem,
        preliminary_focus_class=SimPreliminaryFocus,
        per_field_autofocus_class=SimPerFieldAutofocus,
        oil_safety_class=SimOilSafetyMonitor,
        camera_interface_class=SimCameraInterface,
    )


def _load_real_backend() -> Backend:
    """
    Load real hardware module classes.

    This will fail on Windows or any system without Jetson.GPIO,
    pyserial, and harvesters installed — which is expected.
    Run in simulation mode on non-Jetson systems.
    """
    logger.info("Loading REAL hardware backend (Jetson + peripherals required)")

    try:
        from cap.layer1_hardware.motor_controller import MotorController
        from cap.layer1_hardware.safety_system import SafetySystem
        from cap.layer1_hardware.preliminary_focus import PreliminaryFocus
        from cap.layer1_hardware.per_field_autofocus import PerFieldAutofocus
        from cap.layer1_hardware.oil_safety import OilSafetyMonitor
        from cap.layer2_acquisition.camera_interface import CameraInterface
    except ImportError as e:
        logger.critical(
            "Failed to import real hardware modules. "
            "Are Jetson.GPIO, pyserial, and harvesters installed? "
            "Set hardware_mode to 'simulation' for Windows development. "
            "Error: %s",
            e,
        )
        raise

    return Backend(
        mode=HardwareMode.REAL,
        motor_controller_class=MotorController,
        safety_system_class=SafetySystem,
        preliminary_focus_class=PreliminaryFocus,
        per_field_autofocus_class=PerFieldAutofocus,
        oil_safety_class=OilSafetyMonitor,
        camera_interface_class=CameraInterface,
    )
