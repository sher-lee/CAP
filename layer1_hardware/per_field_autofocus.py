"""
Per-Field Autofocus
=====================
At each X/Y field position, performs fine Z adjustment around the
predicted Z from the focus map. Captures images at N Z-step positions
and returns both the best Z and the full Z-stack for focus stacking.

Works with any motor/camera backend (real or simulated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cap.common.dataclasses import FocusMapResult
from cap.common.logging_setup import get_logger

logger = get_logger("autofocus")


@dataclass
class AutofocusResult:
    """Result of per-field autofocus."""
    predicted_z: float
    actual_z: float
    drift: float
    drift_flagged: bool
    z_positions: list[int]
    sharpness_scores: list[float]
    frames: list[np.ndarray]      # captured frames at each Z depth


class PerFieldAutofocus:
    """
    Per-field autofocus with Z-stack capture for focus stacking.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "focus"):
            self._z_depths = config.focus.z_depths_per_field
            self._drift_threshold = config.focus.drift_threshold
            self._sharpness_metric = config.focus.sharpness_metric
        else:
            focus = config.get("focus", {})
            self._z_depths = focus.get("z_depths_per_field", 6)
            self._drift_threshold = focus.get("drift_threshold", 200)
            self._sharpness_metric = focus.get("sharpness_metric", "laplacian")

        if hasattr(config, "motor"):
            self._z_min = config.motor.z_min
            self._z_max = config.motor.z_max
            self._z_step_microns = config.motor.z_step_size_microns
        else:
            motor = config.get("motor", {})
            self._z_min = motor.get("z_min", 0)
            self._z_max = motor.get("z_max", 10000)
            self._z_step_microns = motor.get("z_step_size_microns", 0.5)

        # Calculate Z step in motor steps
        # The step size determines how far apart the Z-depth captures are
        # Must span ~4 focal planes in z_depths captures
        z_range = self._z_max - self._z_min
        self._z_step = max(1, z_range // (self._z_depths * 10))

        logger.info(
            "PerFieldAutofocus initialized: z_depths=%d, z_step=%d, "
            "drift_threshold=%d, metric=%s",
            self._z_depths, self._z_step,
            self._drift_threshold, self._sharpness_metric,
        )

    def find_best_z(
        self,
        motor_controller,
        camera_interface,
        focus_map: FocusMapResult,
        field_x: int,
        field_y: int,
    ) -> AutofocusResult:
        """
        Find the best Z and capture a full Z-stack at a field position.

        Parameters
        ----------
        motor_controller : MotorController or SimMotorController
        camera_interface : CameraInterface or SimCameraInterface
        focus_map : FocusMapResult
            From preliminary focus.
        field_x : int
            Motor X position of this field.
        field_y : int
            Motor Y position of this field.

        Returns
        -------
        AutofocusResult
            Contains best Z, drift info, and all captured frames.
        """
        # Predict Z from focus surface
        c = focus_map.surface_coefficients
        predicted_z = float(
            c[0]
            + c[1] * field_x
            + c[2] * field_y
            + c[3] * field_x ** 2
            + c[4] * field_y ** 2
            + c[5] * field_x * field_y
        )

        # Generate Z positions centered on predicted Z
        z_center = int(predicted_z)
        half_range = (self._z_depths // 2) * self._z_step
        z_positions = [
            max(self._z_min, min(self._z_max,
                z_center + (i - self._z_depths // 2) * self._z_step
            ))
            for i in range(self._z_depths)
        ]

        # Capture frames at each Z position and score sharpness
        frames = []
        sharpness_scores = []

        for z in z_positions:
            motor_controller.move_to("z", z)
            motor_controller.wait_settle()
            frame = camera_interface.trigger_capture()
            frames.append(frame)

            sharpness = self._compute_sharpness(frame)
            sharpness_scores.append(sharpness)

        # Find best Z
        best_idx = int(np.argmax(sharpness_scores))
        actual_z = float(z_positions[best_idx])

        # Check for drift
        drift = abs(actual_z - predicted_z)
        drift_flagged = drift > self._drift_threshold

        if drift_flagged:
            logger.warning(
                "Z-drift at field (%d, %d): predicted=%.0f, actual=%.0f, "
                "drift=%.0f > threshold=%d",
                field_x, field_y, predicted_z, actual_z,
                drift, self._drift_threshold,
            )
        else:
            logger.debug(
                "Autofocus at (%d, %d): predicted=%.0f, actual=%.0f, drift=%.0f",
                field_x, field_y, predicted_z, actual_z, drift,
            )

        return AutofocusResult(
            predicted_z=predicted_z,
            actual_z=actual_z,
            drift=drift,
            drift_flagged=drift_flagged,
            z_positions=z_positions,
            sharpness_scores=sharpness_scores,
            frames=frames,
        )

    def _compute_sharpness(self, frame: np.ndarray) -> float:
        """Compute sharpness score for a frame."""
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.astype(np.float64)

        if self._sharpness_metric == "brenner":
            return self._brenner_gradient(gray)
        else:
            return self._laplacian_variance(gray)

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Laplacian variance sharpness metric."""
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0
        laplacian = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        return float(np.var(laplacian))

    @staticmethod
    def _brenner_gradient(gray: np.ndarray) -> float:
        """Brenner gradient sharpness metric."""
        if gray.shape[1] < 3:
            return 0.0
        diff = gray[:, 2:] - gray[:, :-2]
        return float(np.sum(diff ** 2))
