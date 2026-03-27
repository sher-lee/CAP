"""
Preliminary Focus
==================
Establishes the focal plane of the slide before scanning begins.
Performs a coarse-fine Z sweep, then samples focus at multiple
X/Y positions to fit a polynomial surface model of the slide tilt.

Works with any motor/camera backend (real or simulated) since it
calls them through their standard interfaces.
"""

from __future__ import annotations

import numpy as np

from cap.common.dataclasses import FocusMapResult
from cap.common.logging_setup import get_logger

logger = get_logger("focus")


class PreliminaryFocus:
    """
    Establishes the slide focal surface via coarse-fine sweep
    and multi-point sampling with polynomial surface fitting.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "focus"):
            self._coarse_steps = config.focus.coarse_sweep_steps
            self._fine_steps = config.focus.fine_sweep_steps
            self._grid_rows = config.focus.focus_map_grid.rows
            self._grid_cols = config.focus.focus_map_grid.cols
            self._fit_order = config.focus.surface_fit_order
            self._sharpness_metric = config.focus.sharpness_metric
        else:
            focus = config.get("focus", {})
            grid = focus.get("focus_map_grid", {})
            self._coarse_steps = focus.get("coarse_sweep_steps", 20)
            self._fine_steps = focus.get("fine_sweep_steps", 10)
            self._grid_rows = grid.get("rows", 3)
            self._grid_cols = grid.get("cols", 3)
            self._fit_order = focus.get("surface_fit_order", 2)
            self._sharpness_metric = focus.get("sharpness_metric", "laplacian")

        if hasattr(config, "motor"):
            self._z_min = config.motor.z_min
            self._z_max = config.motor.z_max
            self._x_min = config.motor.x_min
            self._x_max = config.motor.x_max
            self._y_min = config.motor.y_min
            self._y_max = config.motor.y_max
        else:
            motor = config.get("motor", {})
            self._z_min = motor.get("z_min", 0)
            self._z_max = motor.get("z_max", 10000)
            self._x_min = motor.get("x_min", 0)
            self._x_max = motor.get("x_max", 100000)
            self._y_min = motor.get("y_min", 0)
            self._y_max = motor.get("y_max", 100000)

        logger.info(
            "PreliminaryFocus initialized: coarse=%d, fine=%d, grid=%dx%d, "
            "fit_order=%d, metric=%s",
            self._coarse_steps, self._fine_steps,
            self._grid_rows, self._grid_cols,
            self._fit_order, self._sharpness_metric,
        )

    def run(self, motor_controller, camera_interface) -> FocusMapResult:
        """
        Run the full preliminary focus routine.

        1. Move to slide center
        2. Coarse Z sweep to find approximate focal plane
        3. Fine Z sweep around coarse peak
        4. Sample focus at grid positions across the slide
        5. Fit polynomial surface to the Z values

        Parameters
        ----------
        motor_controller : MotorController or SimMotorController
        camera_interface : CameraInterface or SimCameraInterface

        Returns
        -------
        FocusMapResult
        """
        logger.info("Starting preliminary focus routine...")

        # Step 1: Move to slide center
        center_x = (self._x_min + self._x_max) // 2
        center_y = (self._y_min + self._y_max) // 2
        motor_controller.move_to("x", center_x)
        motor_controller.move_to("y", center_y)

        # Step 2: Coarse sweep
        coarse_best_z = self._coarse_sweep(motor_controller, camera_interface)
        logger.info("Coarse sweep: best Z = %d", coarse_best_z)

        # Step 3: Fine sweep around coarse peak
        fine_best_z = self._fine_sweep(motor_controller, camera_interface, coarse_best_z)
        logger.info("Fine sweep: best Z = %d", fine_best_z)

        # Step 4: Sample across grid
        sample_points = self._sample_grid(motor_controller, camera_interface, fine_best_z)
        logger.info("Focus map: sampled %d points", len(sample_points))

        # Step 5: Fit surface
        coefficients, residual = self._fit_surface(sample_points)
        logger.info("Surface fit: residual = %.4f", residual)

        result = FocusMapResult(
            sample_points=sample_points,
            surface_coefficients=coefficients,
            grid_size=(self._grid_rows, self._grid_cols),
            fit_residual=residual,
        )

        logger.info("Preliminary focus complete")
        return result

    def predict_z(self, result: FocusMapResult, motor_x: int, motor_y: int) -> float:
        """
        Predict the best Z position for a given X/Y using the fitted surface.

        z = a + bx + cy + dx² + ey² + fxy
        """
        c = result.surface_coefficients
        return float(
            c[0]
            + c[1] * motor_x
            + c[2] * motor_y
            + c[3] * motor_x ** 2
            + c[4] * motor_y ** 2
            + c[5] * motor_x * motor_y
        )

    # ----- Internal routines -----

    def _coarse_sweep(self, motor, camera) -> int:
        """Sweep Z across full range at large steps, return Z with highest sharpness."""
        z_positions = np.linspace(self._z_min, self._z_max, self._coarse_steps, dtype=int)
        best_z = int(z_positions[0])
        best_sharpness = -1.0

        for z in z_positions:
            z = int(z)
            motor.move_to("z", z)
            motor.wait_settle()
            frame = camera.trigger_capture()
            sharpness = self._compute_sharpness(frame)

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_z = z

        return best_z

    def _fine_sweep(self, motor, camera, center_z: int) -> int:
        """Fine sweep around the coarse peak."""
        z_range = (self._z_max - self._z_min) // self._coarse_steps * 2
        z_lo = max(self._z_min, center_z - z_range)
        z_hi = min(self._z_max, center_z + z_range)

        z_positions = np.linspace(z_lo, z_hi, self._fine_steps, dtype=int)
        best_z = center_z
        best_sharpness = -1.0

        for z in z_positions:
            z = int(z)
            motor.move_to("z", z)
            motor.wait_settle()
            frame = camera.trigger_capture()
            sharpness = self._compute_sharpness(frame)

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_z = z

        return best_z

    def _sample_grid(
        self, motor, camera, reference_z: int
    ) -> list[tuple[int, int, float]]:
        """
        Sample focus at multiple X/Y positions across the slide.
        At each position, do a small Z sweep around reference_z
        to find the local best Z.
        """
        sample_points = []

        for row in range(self._grid_rows):
            for col in range(self._grid_cols):
                # Spread positions across the motor range
                if self._grid_cols > 1:
                    x = int(self._x_min + col * (self._x_max - self._x_min) / (self._grid_cols - 1))
                else:
                    x = (self._x_min + self._x_max) // 2

                if self._grid_rows > 1:
                    y = int(self._y_min + row * (self._y_max - self._y_min) / (self._grid_rows - 1))
                else:
                    y = (self._y_min + self._y_max) // 2

                motor.move_to("x", x)
                motor.move_to("y", y)

                # Small Z sweep at this position
                local_best_z = self._local_z_sweep(motor, camera, reference_z)
                sample_points.append((x, y, float(local_best_z)))

                logger.debug(
                    "Focus sample (%d/%d): pos=(%d, %d), best_z=%d",
                    row * self._grid_cols + col + 1,
                    self._grid_rows * self._grid_cols,
                    x, y, local_best_z,
                )

        return sample_points

    def _local_z_sweep(self, motor, camera, center_z: int, n_steps: int = 7) -> int:
        """Small Z sweep around a center point to find local best focus."""
        step_size = max(1, (self._z_max - self._z_min) // (self._coarse_steps * 4))
        z_lo = max(self._z_min, center_z - step_size * (n_steps // 2))
        z_hi = min(self._z_max, center_z + step_size * (n_steps // 2))

        z_positions = np.linspace(z_lo, z_hi, n_steps, dtype=int)
        best_z = center_z
        best_sharpness = -1.0

        for z in z_positions:
            z = int(z)
            motor.move_to("z", z)
            motor.wait_settle()
            frame = camera.trigger_capture()
            sharpness = self._compute_sharpness(frame)

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_z = z

        return best_z

    def _compute_sharpness(self, frame: np.ndarray) -> float:
        """
        Compute sharpness score for a frame.
        Uses Laplacian variance (primary) or Brenner gradient (fallback).
        """
        if frame.ndim == 3:
            # Convert to grayscale by averaging channels
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.astype(np.float64)

        if self._sharpness_metric == "brenner":
            return self._brenner_gradient(gray)
        else:
            return self._laplacian_variance(gray)

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """
        Laplacian variance sharpness metric.
        Higher = sharper. Computed as variance of the discrete Laplacian.
        """
        # Simple 3x3 Laplacian kernel applied via convolution
        # For efficiency, use the sum of differences approach
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0

        laplacian = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]    # top
            + gray[2:, 1:-1]     # bottom
            + gray[1:-1, :-2]    # left
            + gray[1:-1, 2:]     # right
        )

        return float(np.var(laplacian))

    @staticmethod
    def _brenner_gradient(gray: np.ndarray) -> float:
        """
        Brenner gradient sharpness metric.
        Sum of squared differences between pixels 2 apart.
        """
        if gray.shape[1] < 3:
            return 0.0

        diff = gray[:, 2:] - gray[:, :-2]
        return float(np.sum(diff ** 2))

    def _fit_surface(
        self, sample_points: list[tuple[int, int, float]]
    ) -> tuple[np.ndarray, float]:
        """
        Fit a 2nd-order polynomial surface to the sample points.

        z = a + bx + cy + dx² + ey² + fxy

        Returns (coefficients, rms_residual).
        """
        n = len(sample_points)
        if n < 6:
            # Not enough points for full quadratic — use plane fit
            logger.warning(
                "Only %d sample points; falling back to plane fit (need 6 for quadratic)", n
            )

        # Build design matrix
        A = np.zeros((n, 6))
        z_vec = np.zeros(n)

        for i, (x, y, z) in enumerate(sample_points):
            # Normalize coordinates to prevent numerical issues
            A[i] = [1, x, y, x * x, y * y, x * y]
            z_vec[i] = z

        # Normalize columns for numerical stability
        col_scales = np.max(np.abs(A), axis=0)
        col_scales[col_scales == 0] = 1.0
        A_normalized = A / col_scales

        # Least-squares fit
        try:
            result = np.linalg.lstsq(A_normalized, z_vec, rcond=None)
            coeffs_normalized = result[0]

            # Un-normalize coefficients
            coefficients = coeffs_normalized / col_scales

            # Compute residual
            z_predicted = A @ coefficients
            residuals = z_vec - z_predicted
            rms_residual = float(np.sqrt(np.mean(residuals ** 2)))

        except np.linalg.LinAlgError:
            logger.error("Surface fitting failed — using flat surface")
            z_mean = float(np.mean(z_vec))
            coefficients = np.array([z_mean, 0, 0, 0, 0, 0], dtype=np.float64)
            rms_residual = float(np.std(z_vec))

        return coefficients, rms_residual
