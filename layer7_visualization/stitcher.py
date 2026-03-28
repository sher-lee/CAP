"""
Slide Stitcher
===============
Assembles individual field composites into a single whole-slide
image using grid-based placement with phase correlation refinement
and feathered boundary blending.

The stitched composite is the primary client-facing output — quality
here directly determines product value.

Usage:
    from cap.layer7_visualization.stitcher import SlideStitcher
    stitcher = SlideStitcher(config)
    composite = stitcher.stitch(field_images, field_positions)
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("visualization.stitcher")


class SlideStitcher:
    """
    Grid-based slide stitcher with phase correlation alignment
    and feathered boundary blending.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "visualization"):
            self._overlap_px = config.visualization.stitch_overlap_px
        else:
            vis = config.get("visualization", {})
            self._overlap_px = vis.get("stitch_overlap_px", 50)

        if hasattr(config, "storage"):
            self._jpeg_quality = config.storage.stacked_jpeg_quality
        else:
            storage = config.get("storage", {})
            self._jpeg_quality = storage.get("stacked_jpeg_quality", 95)

        logger.info("SlideStitcher initialized: overlap=%dpx", self._overlap_px)

    def stitch(
        self,
        field_images: list[np.ndarray],
        field_positions: list[tuple[int, int]],
        field_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Stitch field images into a single whole-slide composite.

        Parameters
        ----------
        field_images : list of np.ndarray
            Stacked composite images for each field, shape (H, W, 3).
        field_positions : list of (motor_x, motor_y)
            Motor coordinates of each field center.
        field_size : tuple of (width_steps, height_steps), optional
            Field size in motor steps. If None, inferred from positions.

        Returns
        -------
        np.ndarray
            Stitched composite image, shape (total_H, total_W, 3), dtype uint8.
        """
        if not field_images or not field_positions:
            raise ValueError("No field images or positions provided")

        n_fields = len(field_images)
        img_h, img_w = field_images[0].shape[:2]

        logger.info(
            "Stitching %d fields, each %dx%d px",
            n_fields, img_w, img_h,
        )

        # Step 1: Compute pixel positions from motor coordinates
        pixel_positions = self._motor_to_pixel_positions(
            field_positions, img_w, img_h, field_size
        )

        # Step 2: Refine positions with phase correlation on overlaps
        refined_positions = self._refine_positions(
            field_images, pixel_positions, img_w, img_h
        )

        # Step 3: Compute canvas size
        all_x = [p[0] for p in refined_positions]
        all_y = [p[1] for p in refined_positions]
        min_x, min_y = min(all_x), min(all_y)

        # Shift so all positions are non-negative
        shifted_positions = [
            (px - min_x, py - min_y) for px, py in refined_positions
        ]

        max_x = max(px + img_w for px, _ in shifted_positions)
        max_y = max(py + img_h for _, py in shifted_positions)

        canvas_w = int(max_x)
        canvas_h = int(max_y)

        logger.info("Canvas size: %dx%d px", canvas_w, canvas_h)

        # Step 4: Composite with feathered blending
        composite = self._blend_onto_canvas(
            field_images, shifted_positions, canvas_w, canvas_h, img_w, img_h
        )

        # Step 5: Color normalization across the composite
        composite = self._normalize_composite(composite)

        logger.info("Stitching complete: %dx%d px", composite.shape[1], composite.shape[0])
        return composite

    def save(
        self,
        composite: np.ndarray,
        output_path: str,
        format: str = "jpeg",
    ) -> str:
        """
        Save the stitched composite to disk.

        Parameters
        ----------
        composite : np.ndarray
            Stitched image.
        output_path : str
            Output file path.
        format : str
            "jpeg", "png", or "tiff".

        Returns
        -------
        str
            Path written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if format == "jpeg" or output_path.endswith(".jpg"):
            cv2.imwrite(output_path, composite, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        elif format == "png":
            cv2.imwrite(output_path, composite)
        else:
            cv2.imwrite(output_path, composite)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Stitched composite saved: %s (%.1f MB)", output_path, size_mb)
        return output_path

    # ----- Internal -----

    def _motor_to_pixel_positions(
        self,
        motor_positions: list[tuple[int, int]],
        img_w: int,
        img_h: int,
        field_size: Optional[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Convert motor step positions to pixel positions on the canvas."""
        if not motor_positions:
            return []

        # Find the range of motor positions
        motor_xs = [p[0] for p in motor_positions]
        motor_ys = [p[1] for p in motor_positions]
        motor_min_x, motor_max_x = min(motor_xs), max(motor_xs)
        motor_min_y, motor_max_y = min(motor_ys), max(motor_ys)

        motor_range_x = motor_max_x - motor_min_x
        motor_range_y = motor_max_y - motor_min_y

        # If field_size provided, use it for scaling
        if field_size:
            fw_steps, fh_steps = field_size
            scale_x = img_w / fw_steps if fw_steps > 0 else 1.0
            scale_y = img_h / fh_steps if fh_steps > 0 else 1.0
        else:
            # Infer scale from position spacing
            if len(motor_positions) > 1 and motor_range_x > 0:
                # Find minimum non-zero X spacing between adjacent fields
                unique_xs = sorted(set(motor_xs))
                if len(unique_xs) > 1:
                    x_spacings = [unique_xs[i+1] - unique_xs[i] for i in range(len(unique_xs)-1)]
                    min_x_spacing = min(s for s in x_spacings if s > 0)
                    scale_x = img_w / min_x_spacing
                else:
                    scale_x = 1.0

                unique_ys = sorted(set(motor_ys))
                if len(unique_ys) > 1:
                    y_spacings = [unique_ys[i+1] - unique_ys[i] for i in range(len(unique_ys)-1)]
                    min_y_spacing = min(s for s in y_spacings if s > 0)
                    scale_y = img_h / min_y_spacing
                else:
                    scale_y = 1.0
            else:
                scale_x = 1.0
                scale_y = 1.0

        pixel_positions = []
        for mx, my in motor_positions:
            px = int((mx - motor_min_x) * scale_x)
            py = int((my - motor_min_y) * scale_y)
            pixel_positions.append((px, py))

        return pixel_positions

    def _refine_positions(
        self,
        images: list[np.ndarray],
        positions: list[tuple[int, int]],
        img_w: int,
        img_h: int,
    ) -> list[tuple[int, int]]:
        """
        Refine pixel positions using phase correlation on overlapping
        regions between adjacent fields.
        """
        if len(images) <= 1:
            return positions

        refined = list(positions)
        overlap = self._overlap_px

        # Build adjacency: find pairs of fields that should overlap
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                px_i, py_i = refined[i]
                px_j, py_j = refined[j]

                dx = abs(px_j - px_i)
                dy = abs(py_j - py_i)

                # Check if fields are adjacent (horizontally or vertically)
                is_h_adjacent = (abs(dx - img_w) < img_w * 0.3) and (dy < img_h * 0.5)
                is_v_adjacent = (abs(dy - img_h) < img_h * 0.3) and (dx < img_w * 0.5)

                if not (is_h_adjacent or is_v_adjacent):
                    continue

                # Extract overlap region and run phase correlation
                try:
                    shift = self._compute_overlap_shift(
                        images[i], images[j],
                        refined[i], refined[j],
                        img_w, img_h,
                    )
                    if shift is not None:
                        sx, sy = shift
                        # Apply half the correction to each image
                        ri_x, ri_y = refined[i]
                        rj_x, rj_y = refined[j]
                        refined[j] = (int(rj_x + sx * 0.5), int(rj_y + sy * 0.5))
                except Exception as e:
                    logger.debug("Phase correlation failed for pair (%d, %d): %s", i, j, e)

        return refined

    def _compute_overlap_shift(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        pos_a: tuple[int, int],
        pos_b: tuple[int, int],
        img_w: int,
        img_h: int,
    ) -> Optional[tuple[float, float]]:
        """Compute sub-pixel shift between two overlapping fields."""
        overlap = self._overlap_px
        ax, ay = pos_a
        bx, by = pos_b

        # Determine overlap direction
        dx = bx - ax
        dy = by - ay

        if abs(dx) > abs(dy):
            # Horizontal adjacency
            if dx > 0:
                strip_a = img_a[:, -overlap:]
                strip_b = img_b[:, :overlap]
            else:
                strip_a = img_a[:, :overlap]
                strip_b = img_b[:, -overlap:]
        else:
            # Vertical adjacency
            if dy > 0:
                strip_a = img_a[-overlap:, :]
                strip_b = img_b[:overlap, :]
            else:
                strip_a = img_a[:overlap, :]
                strip_b = img_b[-overlap:, :]

        # Convert to grayscale float64 for phase correlation
        if strip_a.ndim == 3:
            gray_a = np.mean(strip_a, axis=2).astype(np.float64)
            gray_b = np.mean(strip_b, axis=2).astype(np.float64)
        else:
            gray_a = strip_a.astype(np.float64)
            gray_b = strip_b.astype(np.float64)

        if gray_a.shape != gray_b.shape or gray_a.size == 0:
            return None

        shift, response = cv2.phaseCorrelate(gray_a, gray_b)
        return shift

    def _blend_onto_canvas(
        self,
        images: list[np.ndarray],
        positions: list[tuple[int, int]],
        canvas_w: int,
        canvas_h: int,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """
        Place images onto the canvas with feathered blending in
        overlap regions to eliminate visible seams.
        """
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
        weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float64)

        # Create per-field weight mask (feathered edges)
        field_weight = self._create_feather_mask(img_h, img_w)

        for img, (px, py) in zip(images, positions):
            px, py = int(px), int(py)

            # Compute valid region (clip to canvas bounds)
            src_y0 = max(0, -py)
            src_x0 = max(0, -px)
            src_y1 = min(img_h, canvas_h - py)
            src_x1 = min(img_w, canvas_w - px)

            dst_y0 = max(0, py)
            dst_x0 = max(0, px)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)

            if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
                continue

            region = img[src_y0:src_y1, src_x0:src_x1].astype(np.float64)
            w = field_weight[src_y0:src_y1, src_x0:src_x1]

            canvas[dst_y0:dst_y1, dst_x0:dst_x1] += region * w[:, :, np.newaxis]
            weight_map[dst_y0:dst_y1, dst_x0:dst_x1] += w

        # Normalize by weights (avoid division by zero)
        mask = weight_map > 0
        for c in range(3):
            canvas[:, :, c][mask] /= weight_map[mask]

        return np.clip(canvas, 0, 255).astype(np.uint8)

    @staticmethod
    def _create_feather_mask(height: int, width: int, feather: int = 30) -> np.ndarray:
        """
        Create a 2D weight mask that feathers (tapers) at the edges.
        Center is 1.0, edges linearly ramp from 0 to 1 over `feather` pixels.
        """
        mask = np.ones((height, width), dtype=np.float64)

        for i in range(feather):
            alpha = (i + 1) / feather
            mask[i, :] *= alpha           # top
            mask[height - 1 - i, :] *= alpha  # bottom
            mask[:, i] *= alpha           # left
            mask[:, width - 1 - i] *= alpha  # right

        return mask

    @staticmethod
    def _normalize_composite(composite: np.ndarray) -> np.ndarray:
        """Light per-channel normalization to even out brightness across the composite."""
        result = composite.astype(np.float32)

        for c in range(3):
            channel = result[:, :, c]
            nonzero = channel[channel > 0]
            if len(nonzero) == 0:
                continue

            mean = np.mean(nonzero)
            if mean > 0:
                target = 140.0
                scale = target / mean
                scale = np.clip(scale, 0.7, 1.4)  # Don't over-correct
                result[:, :, c] = channel * scale

        return np.clip(result, 0, 255).astype(np.uint8)
