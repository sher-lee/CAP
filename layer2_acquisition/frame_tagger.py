"""
Frame Tagger
============
Wraps a captured sensor frame plus a motor-position snapshot into a
RawFrame. In stop-and-go mode the CaptureSequencer already produces
RawFrames with field_x/field_y/z_depth set at capture time; in
continuous mode we instead capture *first* (at whatever the motor
positions happen to be) and assign virtual field coordinates later via
the frame grouper.

This module is intentionally narrow: it does not know about scan
strategy, focus surfaces, or grouping. It just bundles raw input
together with calibration-converted positions.

Also exposes microsteps↔microns conversion helpers used by the grouper
and the controller, so unit handling lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cap.common.dataclasses import RawFrame
from cap.common.logging_setup import get_logger

logger = get_logger("tagger")


# ---------------------------------------------------------------------------
# Unit conversion (microsteps ↔ microns)
# ---------------------------------------------------------------------------

def x_steps_to_um(steps: int, x_steps_per_mm: float) -> float:
    """X microstep count → microns."""
    return (steps / x_steps_per_mm) * 1000.0


def y_steps_to_um(steps: int, y_steps_per_mm: float) -> float:
    """Y microstep count → microns."""
    return (steps / y_steps_per_mm) * 1000.0


def z_steps_to_um(steps: int, z_step_size_microns: float) -> float:
    """Z microstep count → microns."""
    return steps * z_step_size_microns


def x_um_to_steps(x_um: float, x_steps_per_mm: float) -> int:
    """Microns → X microstep count (rounded)."""
    return int(round(x_um * x_steps_per_mm / 1000.0))


def y_um_to_steps(y_um: float, y_steps_per_mm: float) -> int:
    """Microns → Y microstep count (rounded)."""
    return int(round(y_um * y_steps_per_mm / 1000.0))


def z_um_to_steps(z_um: float, z_step_size_microns: float) -> int:
    """Microns → Z microstep count (rounded)."""
    if z_step_size_microns <= 0:
        raise ValueError(f"z_step_size_microns must be > 0; got {z_step_size_microns}")
    return int(round(z_um / z_step_size_microns))


# ---------------------------------------------------------------------------
# Position snapshot (what the motor controller hands to the tagger)
# ---------------------------------------------------------------------------

@dataclass
class PositionSnapshot:
    """
    Motor positions sampled at the moment a camera trigger fires.

    On real hardware this is built from latched step counts inside the
    GPIO interrupt that fires the camera; in sim it's read directly from
    the SimMotorController. Either way it must be self-consistent — all
    three axes sampled at the same instant.
    """
    x_steps: int
    y_steps: int
    z_steps: int
    timestamp: float
    """time.monotonic() at the trigger instant."""


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------

class FrameTagger:
    """
    Builds RawFrame records from sensor data + position snapshots.

    Holds calibration constants so callers don't have to thread them
    through. Stateless except for the frame_id counter, which assigns
    sequential ids across the whole scan.
    """

    def __init__(
        self,
        slide_id: int,
        x_steps_per_mm: float,
        y_steps_per_mm: float,
        z_step_size_microns: float,
    ) -> None:
        self._slide_id = slide_id
        self._x_steps_per_mm = x_steps_per_mm
        self._y_steps_per_mm = y_steps_per_mm
        self._z_step_size_microns = z_step_size_microns
        self._next_frame_id = 0

        logger.debug(
            "FrameTagger initialized: slide_id=%d, calibration "
            "(x=%.2f steps/mm, y=%.2f steps/mm, z=%.4f μm/step)",
            slide_id, x_steps_per_mm, y_steps_per_mm, z_step_size_microns,
        )

    def tag(
        self,
        bayer_data: np.ndarray,
        snapshot: PositionSnapshot,
        depth_index: int,
        scan_line: int,
        scan_direction: int,
    ) -> RawFrame:
        """
        Build a RawFrame for one captured sensor frame.

        Parameters
        ----------
        bayer_data : np.ndarray
            Raw sensor frame. Shape (H, W), dtype uint8 or uint16.
        snapshot : PositionSnapshot
            Motor positions at trigger time.
        depth_index : int
            Which Z bucket (0..frames_per_field-1) this frame belongs to,
            as decided by the z_cycle_generator's depth_index_sequence.
        scan_line : int
            Serpentine scan-line index (Y row), 0-based.
        scan_direction : int
            +1 for left-to-right travel on this line, -1 for right-to-left.

        Returns
        -------
        RawFrame
            field_x and field_y are left as -1 — the frame_grouper will
            assign them when it binds this frame to a virtual field.
        """
        frame = RawFrame(
            slide_id=self._slide_id,
            field_x=-1,
            field_y=-1,
            z_depth=depth_index,
            timestamp=snapshot.timestamp,
            bayer_data=bayer_data,
            motor_position=(snapshot.x_steps, snapshot.y_steps, snapshot.z_steps),
            frame_id=self._next_frame_id,
            scan_line=scan_line,
            scan_direction=scan_direction,
        )
        self._next_frame_id += 1
        return frame

    @property
    def frames_tagged(self) -> int:
        return self._next_frame_id

    def position_um(self, frame: RawFrame) -> tuple[float, float, float]:
        """Return (x_um, y_um, z_um) for a tagged frame using this tagger's calibration."""
        xs, ys, zs = frame.motor_position
        return (
            x_steps_to_um(xs, self._x_steps_per_mm),
            y_steps_to_um(ys, self._y_steps_per_mm),
            z_steps_to_um(zs, self._z_step_size_microns),
        )
