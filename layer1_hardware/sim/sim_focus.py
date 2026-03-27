"""
Simulated Preliminary Focus
=============================
Drop-in replacement for the real PreliminaryFocus module.
Returns a flat focus surface (all Z values identical) since
there is no real slide to focus on in simulation.
"""

from __future__ import annotations

import numpy as np

from cap.common.dataclasses import FocusMapResult
from cap.common.logging_setup import get_logger

logger = get_logger("focus.sim")


class SimPreliminaryFocus:
    """Simulated preliminary focus for Windows development."""

    def __init__(self, config: dict | object) -> None:
        if hasattr(config, "focus"):
            self._grid_rows = config.focus.focus_map_grid.rows
            self._grid_cols = config.focus.focus_map_grid.cols
            self._z_center = (config.motor.z_max - config.motor.z_min) // 2
        else:
            focus = config.get("focus", {})
            grid = focus.get("focus_map_grid", {})
            self._grid_rows = grid.get("rows", 3)
            self._grid_cols = grid.get("cols", 3)
            motor = config.get("motor", {})
            self._z_center = (motor.get("z_max", 10000) - motor.get("z_min", 0)) // 2

        logger.info(
            "SimPreliminaryFocus initialized: grid=%dx%d, z_center=%d",
            self._grid_rows, self._grid_cols, self._z_center,
        )

    def run(
        self,
        motor_controller: object,
        camera_interface: object,
    ) -> FocusMapResult:
        """
        Run the preliminary focus routine.

        In simulation, skips all motor movement and camera capture.
        Returns a flat focus surface at the Z midpoint.

        Parameters
        ----------
        motor_controller : SimMotorController or MotorController
            Motor controller for Z movement (unused in sim).
        camera_interface : SimCameraInterface or CameraInterface
            Camera for sharpness evaluation (unused in sim).

        Returns
        -------
        FocusMapResult
            A flat focus surface with zero residual.
        """
        logger.info("Running simulated preliminary focus...")

        # Generate sample points on a grid, all at the same Z
        sample_points = []
        for row in range(self._grid_rows):
            for col in range(self._grid_cols):
                # Spread points across the motor range
                x = int(col * 10000 / max(self._grid_cols - 1, 1))
                y = int(row * 10000 / max(self._grid_rows - 1, 1))
                sample_points.append((x, y, float(self._z_center)))

        # Flat surface: z = a (constant), all other coefficients zero
        # Coefficients: [a, b, c, d, e, f] for z = a + bx + cy + dx² + ey² + fxy
        surface_coefficients = np.array(
            [float(self._z_center), 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        result = FocusMapResult(
            sample_points=sample_points,
            surface_coefficients=surface_coefficients,
            grid_size=(self._grid_rows, self._grid_cols),
            fit_residual=0.0,
        )

        logger.info(
            "Preliminary focus complete: %d sample points, flat surface at Z=%d, "
            "residual=%.4f",
            len(sample_points), self._z_center, result.fit_residual,
        )

        return result

    def predict_z(self, result: FocusMapResult, motor_x: int, motor_y: int) -> float:
        """
        Predict the best Z position for a given X/Y using the fitted surface.

        Parameters
        ----------
        result : FocusMapResult
            The focus map from run().
        motor_x : int
            X position in microsteps.
        motor_y : int
            Y position in microsteps.

        Returns
        -------
        float
            Predicted Z position in microsteps.
        """
        c = result.surface_coefficients
        z = (
            c[0]
            + c[1] * motor_x
            + c[2] * motor_y
            + c[3] * motor_x**2
            + c[4] * motor_y**2
            + c[5] * motor_x * motor_y
        )
        return float(z)
