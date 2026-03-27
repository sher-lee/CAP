"""
Synthetic Z-Stack Generator
==============================
Generates Z-stack image sets for testing the focus stacking pipeline
without real microscopy hardware. Takes a single base image and creates
N versions with different regions artificially blurred, simulating
the effect of different Z depths bringing different organisms into
focus at ~4 distinct focal planes.

Used by SimCameraInterface when capturing multi-Z-depth sequences.
"""

from __future__ import annotations

import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("zstack.sim")


class SyntheticZStackGenerator:
    """
    Generates synthetic Z-stack images by selectively blurring
    different regions of a base image at each Z depth.
    """

    def __init__(self, z_depths: int = 6, seed: int = 42) -> None:
        """
        Parameters
        ----------
        z_depths : int
            Number of Z-depth images to generate per stack.
        seed : int
            Random seed for reproducible region assignments.
        """
        self._z_depths = z_depths
        self._rng = np.random.default_rng(seed)
        logger.info("SyntheticZStackGenerator initialized: z_depths=%d", z_depths)

    def generate_stack(
        self,
        base_image: np.ndarray,
        blur_radius: int = 7,
    ) -> list[np.ndarray]:
        """
        Generate a Z-stack from a single base image.

        The image is divided into a grid of regions. Each Z-depth has
        a different subset of regions in focus (sharp) and the rest
        blurred, simulating the ~4 distinct focal planes observed
        empirically at 1000x oil immersion.

        Parameters
        ----------
        base_image : np.ndarray
            The base image, shape (H, W), any dtype.
        blur_radius : int
            Radius of the box blur applied to out-of-focus regions.
            Must be odd. Default 7.

        Returns
        -------
        list[np.ndarray]
            List of z_depths images, each the same shape as base_image.
            One image per Z-depth level.
        """
        h, w = base_image.shape[:2]

        # Divide into a grid of regions (4x4 = 16 regions)
        grid_rows, grid_cols = 4, 4
        region_h = h // grid_rows
        region_w = w // grid_cols

        # Assign each region to a "focal plane" (0 to 3)
        # This simulates organisms at different physical heights in the sample
        n_planes = min(4, self._z_depths)
        region_planes = self._rng.integers(0, n_planes, size=(grid_rows, grid_cols))

        # Map focal planes to the Z-depth where they're sharpest
        # Plane 0 → sharpest at Z-depth 0, Plane 1 → sharpest at Z-depth ~1.5, etc.
        plane_to_best_z = np.linspace(0, self._z_depths - 1, n_planes)

        stack = []
        for z_idx in range(self._z_depths):
            frame = base_image.copy()

            for row in range(grid_rows):
                for col in range(grid_cols):
                    plane = region_planes[row, col]
                    best_z = plane_to_best_z[plane]

                    # Distance from this Z-depth to the region's best Z
                    distance = abs(z_idx - best_z)

                    # Blur intensity proportional to distance from best focus
                    if distance < 0.5:
                        # In focus — keep sharp
                        continue
                    else:
                        # Out of focus — apply box blur
                        blur_strength = min(int(blur_radius * distance / 2), blur_radius * 2)
                        if blur_strength < 1:
                            continue

                        y_start = row * region_h
                        y_end = (row + 1) * region_h if row < grid_rows - 1 else h
                        x_start = col * region_w
                        x_end = (col + 1) * region_w if col < grid_cols - 1 else w

                        region = frame[y_start:y_end, x_start:x_end]
                        frame[y_start:y_end, x_start:x_end] = self._box_blur(
                            region, blur_strength
                        )

            stack.append(frame)

        logger.debug(
            "Generated Z-stack: %d depths, %dx%d regions, %d focal planes",
            self._z_depths, grid_rows, grid_cols, n_planes,
        )

        return stack

    @staticmethod
    def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply a simple box blur to a 2D array.
        Pure NumPy — no OpenCV dependency needed.

        Parameters
        ----------
        image : np.ndarray
            2D input image region.
        radius : int
            Blur radius. Kernel size = 2*radius + 1.

        Returns
        -------
        np.ndarray
            Blurred image, same shape and dtype as input.
        """
        if radius < 1:
            return image

        kernel_size = 2 * radius + 1
        dtype = image.dtype

        # Convert to float for averaging
        img = image.astype(np.float64)

        # Cumulative sum along rows then columns (box filter)
        # Pad to handle edges
        padded = np.pad(img, radius, mode="edge")

        # Horizontal pass
        cumsum_h = np.cumsum(padded, axis=1)
        blurred = (
            cumsum_h[:, kernel_size:] - cumsum_h[:, :-kernel_size]
        ) / kernel_size

        # Vertical pass
        cumsum_v = np.cumsum(blurred, axis=0)
        blurred = (
            cumsum_v[kernel_size:, :] - cumsum_v[:-kernel_size, :]
        ) / kernel_size

        # Trim to original size (the double cumsum may change dimensions)
        result = blurred[: image.shape[0], : image.shape[1]]

        return np.clip(result, 0, np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else result.max()).astype(dtype)
