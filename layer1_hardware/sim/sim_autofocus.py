"""
Simulated Per-Field Autofocus
===============================
Drop-in replacement for the real PerFieldAutofocus module.
Returns the predicted Z from the focus map for every field
without performing any actual Z movement or sharpness evaluation.
"""

from __future__ import annotations

from cap.common.dataclasses import FocusMapResult
from cap.common.logging_setup import get_logger

logger = get_logger("autofocus.sim")


class SimPerFieldAutofocus:
    """Simulated per-field autofocus for Windows development."""

    def __init__(self, config: dict | object) -> None:
        if hasattr(config, "focus"):
            self._z_depths = config.focus.z_depths_per_field
            self._drift_threshold = config.focus.drift_threshold
        else:
            focus = config.get("focus", {})
            self._z_depths = focus.get("z_depths_per_field", 6)
            self._drift_threshold = focus.get("drift_threshold", 200)

        logger.info(
            "SimPerFieldAutofocus initialized: z_depths=%d, drift_threshold=%d",
            self._z_depths, self._drift_threshold,
        )

    def find_best_z(
        self,
        motor_controller: object,
        camera_interface: object,
        focus_map: FocusMapResult,
        field_x: int,
        field_y: int,
    ) -> AutofocusResult:
        """
        Find the best Z position for a specific field.

        In simulation, returns the predicted Z from the focus map
        with no actual Z movement or image capture.

        Parameters
        ----------
        motor_controller : SimMotorController or MotorController
            Motor controller (unused in sim).
        camera_interface : SimCameraInterface or CameraInterface
            Camera (unused in sim).
        focus_map : FocusMapResult
            Focus surface from preliminary focus.
        field_x : int
            Motor X position of this field.
        field_y : int
            Motor Y position of this field.

        Returns
        -------
        AutofocusResult
            The focus result for this field.
        """
        # Predict Z from the focus surface
        c = focus_map.surface_coefficients
        predicted_z = (
            c[0]
            + c[1] * field_x
            + c[2] * field_y
            + c[3] * field_x**2
            + c[4] * field_y**2
            + c[5] * field_x * field_y
        )

        # In simulation, actual Z equals predicted Z (no drift)
        actual_z = predicted_z
        drift = abs(actual_z - predicted_z)
        drift_flagged = drift > self._drift_threshold

        # Generate the Z positions for the Z-stack capture
        z_step = 50  # Simulated Z step size in microsteps
        z_center = int(actual_z)
        z_positions = [
            z_center + (i - self._z_depths // 2) * z_step
            for i in range(self._z_depths)
        ]

        logger.debug(
            "Autofocus at (%d, %d): predicted_z=%.1f, actual_z=%.1f, drift=%.1f%s",
            field_x, field_y, predicted_z, actual_z, drift,
            " [FLAGGED]" if drift_flagged else "",
        )

        return AutofocusResult(
            predicted_z=predicted_z,
            actual_z=actual_z,
            drift=drift,
            drift_flagged=drift_flagged,
            z_positions=z_positions,
            sharpness_scores=[1.0] * self._z_depths,  # All equally sharp in sim
        )


class AutofocusResult:
    """Result of per-field autofocus for one field."""

    __slots__ = (
        "predicted_z",
        "actual_z",
        "drift",
        "drift_flagged",
        "z_positions",
        "sharpness_scores",
    )

    def __init__(
        self,
        predicted_z: float,
        actual_z: float,
        drift: float,
        drift_flagged: bool,
        z_positions: list[int],
        sharpness_scores: list[float],
    ) -> None:
        self.predicted_z = predicted_z
        self.actual_z = actual_z
        self.drift = drift
        self.drift_flagged = drift_flagged
        self.z_positions = z_positions
        self.sharpness_scores = sharpness_scores
