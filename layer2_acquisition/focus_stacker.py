"""
Focus Stacker
==============
Produces one fully sharp composite image per field from a Z-stack
of 6 captures at different focal depths.

Pipeline per field:
1. Register frames 1-5 to frame 0 via phase correlation (correct X/Y motor drift)
2. Compute per-block (16x16 or 32x32) sharpness at each Z depth
3. Select the sharpest block from the best Z depth at each position
4. Blend block boundaries with Gaussian falloff

The result is an all-in-focus composite regardless of sample depth
variation, which is critical for the client-facing output quality.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from cap.common.dataclasses import StackedField
from cap.common.logging_setup import get_logger

logger = get_logger("stacker")


class FocusStacker:
    """
    Per-block focus stacking with phase correlation registration
    and Gaussian boundary blending.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "focus"):
            self._block_size = config.focus.block_size
            self._blend_sigma = config.focus.blend_sigma
            self._max_shift = config.focus.max_registration_shift
        else:
            focus = config.get("focus", {})
            self._block_size = focus.get("block_size", 16)
            self._blend_sigma = focus.get("blend_sigma", 4.0)
            self._max_shift = focus.get("max_registration_shift", 50)

        # Pre-compute Gaussian blending weights for block boundaries
        self._blend_weights = self._create_blend_weights(self._block_size, self._blend_sigma)

        logger.info(
            "FocusStacker initialized: block_size=%d, blend_sigma=%.1f, max_shift=%d",
            self._block_size, self._blend_sigma, self._max_shift,
        )

    def stack(
        self,
        frames: list[np.ndarray],
        slide_id: int = 0,
        field_x: int = 0,
        field_y: int = 0,
        expected_shifts_px: Optional[list[tuple[float, float]]] = None,
    ) -> StackedField:
        """
        Produce a focus-stacked composite from a list of Z-depth frames.

        Parameters
        ----------
        frames : list of np.ndarray
            Raw or debayered frames at different Z depths.
            All must be the same shape. Can be grayscale (H, W) or
            color (H, W, 3).
        slide_id : int
            For metadata tracking.
        field_x, field_y : int
            Motor coordinates for metadata.
        expected_shifts_px : list of (dx, dy) or None
            Optional per-frame translation prior in pixels, same length as
            `frames`, with frames[0] at (0.0, 0.0). Provided by continuous
            scanning (where stage motion during one Z-stack produces
            multi-hundred-pixel shifts that exceed max_registration_shift).
            When supplied, the stacker pre-warps each frame by its prior
            and phase-correlates around zero for sub-pixel residual only.
            When None (stop-and-go), behavior is unchanged.

        Returns
        -------
        StackedField
            The composite image and stacking metadata.
        """
        start_time = time.monotonic()

        if len(frames) < 2:
            # Only one frame — return it directly
            logger.debug("Single frame, no stacking needed")
            return StackedField(
                slide_id=slide_id,
                field_x=field_x,
                field_y=field_y,
                composite=frames[0].copy(),
                sharpness_map=np.array([[0.0]]),
                z_distribution={0: 1},
                stacking_duration_ms=0.0,
                registration_shifts=[],
                block_size=self._block_size,
            )

        n_frames = len(frames)
        h, w = frames[0].shape[:2]
        is_color = frames[0].ndim == 3

        # Step 1: Convert to grayscale for sharpness evaluation
        gray_frames = []
        for f in frames:
            if f.ndim == 3:
                gray_frames.append(np.mean(f, axis=2).astype(np.float64))
            else:
                gray_frames.append(f.astype(np.float64))

        # Step 2: Register frames to frame 0 via phase correlation
        # (with optional step-counter prior for continuous-mode capture)
        aligned_frames, aligned_grays, shifts = self._register_frames(
            frames, gray_frames, expected_shifts_px,
        )

        # Step 3: Per-block sharpness evaluation
        bs = self._block_size
        blocks_y = h // bs
        blocks_x = w // bs

        # Compute sharpness maps for all frames
        sharpness_maps = np.zeros((n_frames, blocks_y, blocks_x), dtype=np.float64)

        for f_idx in range(n_frames):
            gray = aligned_grays[f_idx]
            for by in range(blocks_y):
                for bx in range(blocks_x):
                    y0 = by * bs
                    x0 = bx * bs
                    block = gray[y0:y0 + bs, x0:x0 + bs]
                    sharpness_maps[f_idx, by, bx] = self._laplacian_variance(block)

        # Step 4: Select best Z per block
        best_z_map = np.argmax(sharpness_maps, axis=0)  # (blocks_y, blocks_x)
        best_sharpness_map = np.max(sharpness_maps, axis=0)

        # Step 5: Build composite from best blocks
        if is_color:
            composite = np.zeros((h, w, 3), dtype=frames[0].dtype)
        else:
            composite = np.zeros((h, w), dtype=frames[0].dtype)

        # Track Z-depth distribution
        z_distribution: dict[int, int] = {}

        for by in range(blocks_y):
            for bx in range(blocks_x):
                best_z = int(best_z_map[by, bx])
                z_distribution[best_z] = z_distribution.get(best_z, 0) + 1

                y0 = by * bs
                x0 = bx * bs

                if is_color:
                    composite[y0:y0 + bs, x0:x0 + bs, :] = (
                        aligned_frames[best_z][y0:y0 + bs, x0:x0 + bs, :]
                    )
                else:
                    composite[y0:y0 + bs, x0:x0 + bs] = (
                        aligned_frames[best_z][y0:y0 + bs, x0:x0 + bs]
                    )

        # Step 6: Apply Gaussian blending at block boundaries
        composite = self._apply_boundary_blending(composite, aligned_frames, best_z_map)

        # Handle remaining pixels at edges (if image size isn't divisible by block size)
        remainder_y = h - blocks_y * bs
        remainder_x = w - blocks_x * bs

        if remainder_y > 0:
            # Use the sharpest full frame for bottom edge
            overall_sharpness = [np.mean(sm) for sm in sharpness_maps]
            best_overall = int(np.argmax(overall_sharpness))
            if is_color:
                composite[blocks_y * bs:, :, :] = aligned_frames[best_overall][blocks_y * bs:, :, :]
            else:
                composite[blocks_y * bs:, :] = aligned_frames[best_overall][blocks_y * bs:, :]

        if remainder_x > 0:
            overall_sharpness = [np.mean(sm) for sm in sharpness_maps]
            best_overall = int(np.argmax(overall_sharpness))
            if is_color:
                composite[:, blocks_x * bs:, :] = aligned_frames[best_overall][:, blocks_x * bs:, :]
            else:
                composite[:, blocks_x * bs:] = aligned_frames[best_overall][:, blocks_x * bs:]

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Focus stacking complete: %dx%d blocks, %d Z-depths, %.1f ms, "
            "shifts=%s",
            blocks_x, blocks_y, n_frames, duration_ms,
            [f"({s[0]:.1f},{s[1]:.1f})" for s in shifts],
        )

        return StackedField(
            slide_id=slide_id,
            field_x=field_x,
            field_y=field_y,
            composite=composite,
            sharpness_map=best_sharpness_map.astype(np.float32),
            z_distribution=z_distribution,
            stacking_duration_ms=duration_ms,
            registration_shifts=shifts,
            block_size=self._block_size,
        )

    # ----- Registration -----

    def _register_frames(
        self,
        frames: list[np.ndarray],
        gray_frames: list[np.ndarray],
        expected_shifts_px: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float]]]:
        """
        Register frames 1..N to frame 0.

        Two paths:

        **Stop-and-go (no prior):** phase-correlate each frame against
        frame 0 directly. Reject shifts beyond max_registration_shift
        (likely a capture error) and fall back to unregistered.

        **Continuous (prior given):** pre-warp frame i by its prior shift,
        phase-correlate the pre-warped version against frame 0 for the
        residual (which should be sub-pixel), and combine prior + residual
        for the final translation. The max_shift threshold applies to the
        residual only — large priors are expected and OK.

        Returns aligned frames, aligned grays, and the final shifts used.
        """
        n = len(frames)
        ref_gray = gray_frames[0]

        if expected_shifts_px is not None and len(expected_shifts_px) != n:
            logger.warning(
                "expected_shifts_px length (%d) != frames length (%d); ignoring prior",
                len(expected_shifts_px), n,
            )
            expected_shifts_px = None

        aligned_frames = [frames[0]]
        aligned_grays = [gray_frames[0]]
        shifts: list[tuple[float, float]] = []
        if expected_shifts_px is not None:
            # Record (0, 0) for frame 0 to keep the shifts list parallel to frames
            # in continuous mode. Stop-and-go skips this for backward compat.
            shifts.append((0.0, 0.0))

        for i in range(1, n):
            try:
                if expected_shifts_px is None:
                    final_shift = self._register_one_no_prior(ref_gray, gray_frames[i], i)
                else:
                    final_shift = self._register_one_with_prior(
                        ref_gray, gray_frames[i], expected_shifts_px[i], i,
                    )

                # If registration failed entirely, append the frame untouched
                if final_shift is None:
                    aligned_frames.append(frames[i])
                    aligned_grays.append(gray_frames[i])
                    shifts.append((0.0, 0.0))
                    continue

                dx, dy = final_shift
                h, w = frames[i].shape[:2]
                M = np.float64([[1, 0, dx], [0, 1, dy]])

                aligned = cv2.warpAffine(
                    frames[i], M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                aligned_gray = cv2.warpAffine(
                    gray_frames[i], M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                aligned_frames.append(aligned)
                aligned_grays.append(aligned_gray)
                shifts.append((dx, dy))

            except Exception as e:
                logger.warning(
                    "Frame %d: registration failed (%s) — using unregistered",
                    i, e,
                )
                aligned_frames.append(frames[i])
                aligned_grays.append(gray_frames[i])
                shifts.append((0.0, 0.0))

        return aligned_frames, aligned_grays, shifts

    def _register_one_no_prior(
        self,
        ref_gray: np.ndarray,
        gray_i: np.ndarray,
        i: int,
    ) -> Optional[tuple[float, float]]:
        """Stop-and-go registration: direct phase correlation, max_shift gate."""
        shift, _ = cv2.phaseCorrelate(ref_gray, gray_i)
        dx, dy = shift
        if abs(dx) > self._max_shift or abs(dy) > self._max_shift:
            logger.warning(
                "Frame %d: registration shift (%.1f, %.1f) exceeds max (%d) — "
                "using unregistered frame",
                i, dx, dy, self._max_shift,
            )
            return None
        return (dx, dy)

    def _register_one_with_prior(
        self,
        ref_gray: np.ndarray,
        gray_i: np.ndarray,
        prior: tuple[float, float],
        i: int,
    ) -> Optional[tuple[float, float]]:
        """
        Continuous-mode registration: pre-warp by prior, phase-correlate
        for sub-pixel residual, combine.
        """
        prior_dx, prior_dy = prior
        h, w = gray_i.shape[:2]

        # Pre-warp frame i by its prior into frame 0's coordinate system.
        M_prior = np.float64([[1, 0, prior_dx], [0, 1, prior_dy]])
        prewarp = cv2.warpAffine(
            gray_i, M_prior, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Residual phase correlation should now be sub-pixel.
        residual, _ = cv2.phaseCorrelate(ref_gray, prewarp)
        rx, ry = residual

        if abs(rx) > self._max_shift or abs(ry) > self._max_shift:
            # Residual too large: priors don't seem trustworthy, or content
            # genuinely shifted (e.g. partial obstruction). Use prior only.
            logger.warning(
                "Frame %d: residual shift (%.1f, %.1f) after prior (%.1f, %.1f) "
                "exceeds max (%d) — using prior alone",
                i, rx, ry, prior_dx, prior_dy, self._max_shift,
            )
            return (prior_dx, prior_dy)

        return (prior_dx + rx, prior_dy + ry)

    # ----- Sharpness -----

    @staticmethod
    def _laplacian_variance(block: np.ndarray) -> float:
        """Laplacian variance sharpness metric for a single block."""
        h, w = block.shape
        if h < 3 or w < 3:
            return 0.0

        lap = (
            -4 * block[1:-1, 1:-1]
            + block[:-2, 1:-1]
            + block[2:, 1:-1]
            + block[1:-1, :-2]
            + block[1:-1, 2:]
        )
        return float(np.var(lap))

    # ----- Blending -----

    def _apply_boundary_blending(
        self,
        composite: np.ndarray,
        aligned_frames: list[np.ndarray],
        best_z_map: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Gaussian blending at block boundaries to prevent
        visible seams between adjacent blocks from different Z depths.

        Uses a weighted average in the overlap region between
        neighboring blocks.
        """
        bs = self._block_size
        blocks_y, blocks_x = best_z_map.shape
        blend_radius = max(1, bs // 4)  # Blend zone is 1/4 of block size

        result = composite.astype(np.float64)
        is_color = composite.ndim == 3

        # Horizontal boundaries (between left and right blocks in same row)
        for by in range(blocks_y):
            for bx in range(blocks_x - 1):
                z_left = int(best_z_map[by, bx])
                z_right = int(best_z_map[by, bx + 1])

                if z_left == z_right:
                    continue  # Same Z depth, no seam

                # Blend zone centered on the boundary
                boundary_x = (bx + 1) * bs
                x_start = max(0, boundary_x - blend_radius)
                x_end = min(composite.shape[1], boundary_x + blend_radius)
                y_start = by * bs
                y_end = min(composite.shape[0], (by + 1) * bs)

                if x_end <= x_start:
                    continue

                # Linear blend weights across the boundary
                width = x_end - x_start
                alpha = np.linspace(1.0, 0.0, width).reshape(1, -1)

                left_data = aligned_frames[z_left][y_start:y_end, x_start:x_end]
                right_data = aligned_frames[z_right][y_start:y_end, x_start:x_end]

                if is_color:
                    alpha_3d = alpha[:, :, np.newaxis]
                    blended = left_data.astype(np.float64) * alpha_3d + right_data.astype(np.float64) * (1 - alpha_3d)
                else:
                    blended = left_data.astype(np.float64) * alpha + right_data.astype(np.float64) * (1 - alpha)

                result[y_start:y_end, x_start:x_end] = blended

        # Vertical boundaries (between top and bottom blocks in same column)
        for by in range(blocks_y - 1):
            for bx in range(blocks_x):
                z_top = int(best_z_map[by, bx])
                z_bottom = int(best_z_map[by + 1, bx])

                if z_top == z_bottom:
                    continue

                boundary_y = (by + 1) * bs
                y_start = max(0, boundary_y - blend_radius)
                y_end = min(composite.shape[0], boundary_y + blend_radius)
                x_start = bx * bs
                x_end = min(composite.shape[1], (bx + 1) * bs)

                if y_end <= y_start:
                    continue

                height = y_end - y_start
                alpha = np.linspace(1.0, 0.0, height).reshape(-1, 1)

                top_data = aligned_frames[z_top][y_start:y_end, x_start:x_end]
                bottom_data = aligned_frames[z_bottom][y_start:y_end, x_start:x_end]

                if is_color:
                    alpha_3d = alpha[:, :, np.newaxis]
                    blended = top_data.astype(np.float64) * alpha_3d + bottom_data.astype(np.float64) * (1 - alpha_3d)
                else:
                    blended = top_data.astype(np.float64) * alpha + bottom_data.astype(np.float64) * (1 - alpha)

                result[y_start:y_end, x_start:x_end] = blended

        return result.astype(composite.dtype)

    @staticmethod
    def _create_blend_weights(block_size: int, sigma: float) -> np.ndarray:
        """
        Create 2D Gaussian weights for block boundary blending.
        Center of the block is weight 1.0, edges taper off.
        """
        x = np.arange(block_size) - block_size / 2 + 0.5
        y = np.arange(block_size) - block_size / 2 + 0.5
        xx, yy = np.meshgrid(x, y)
        weights = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return weights / weights.max()
