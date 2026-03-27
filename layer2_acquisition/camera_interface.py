"""
Camera Interface (Real Hardware)
===================================
Interface for the VEN-830-22U3C-M01 camera via USB3 Vision /
Genicam using the harvesters Python SDK.

THIS IS A STUB — the implementation will be filled in during
Phase 10 (hardware bring-up). The interface matches SimCameraInterface.

For Windows development, use SimCameraInterface via the
backend selector (hardware_mode: simulation).
"""

from __future__ import annotations

import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("camera")


class CameraInterface:
    """
    Real camera interface for VEN-830 via harvesters/Genicam.

    Stub implementation — raises NotImplementedError.
    Will be implemented during Phase 10 hardware bring-up.
    """

    def __init__(self, config: object) -> None:
        raise NotImplementedError(
            "Real CameraInterface requires harvesters and a USB3 Vision camera. "
            "Set hardware_mode to 'simulation' for Windows development."
        )

    def initialize(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError

    def set_exposure(self, value: int) -> None:
        raise NotImplementedError

    def set_gain(self, value: float) -> None:
        raise NotImplementedError

    def set_white_balance(self, r: float, g: float, b: float) -> None:
        raise NotImplementedError

    def trigger_capture(self) -> np.ndarray:
        raise NotImplementedError

    def get_frame_buffer(self):
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        raise NotImplementedError

    @property
    def resolution(self) -> tuple[int, int]:
        raise NotImplementedError
