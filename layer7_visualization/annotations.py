
"""
Annotation Overlay
====================
Renders color-coded bounding boxes for organism detections on
stitched or individual field images. Colors are configurable
and use a colorblind-friendly palette by default.

Annotations can be toggled on/off in the viewer — the overlay
is generated as a separate layer or burned directly into a copy.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("visualization.annotations")

# Default colorblind-friendly palette (RGB tuples)
DEFAULT_COLORS = {
    "cocci_small": (232, 89, 60),     # Coral
    "cocci_large": (216, 90, 48),     # Dark coral
    "yeast": (29, 158, 117),          # Teal
    "rods": (55, 138, 221),           # Blue
    "ear_mites": (212, 83, 126),      # Pink
    "empty_artifact": (136, 135, 128),  # Gray
}


class AnnotationRenderer:
    """
    Renders detection bounding boxes on images with class-colored
    labels and confidence scores.
    """

    def __init__(self, config: object = None) -> None:
        self._colors = dict(DEFAULT_COLORS)

        if config is not None:
            if hasattr(config, "visualization"):
                color_map = config.visualization.annotation_colors
            else:
                vis = config.get("visualization", {})
                color_map = vis.get("annotation_colors", {})

            # Parse hex colors from config
            for cls, hex_color in color_map.items():
                if isinstance(hex_color, str) and hex_color.startswith("#"):
                    self._colors[cls] = self._hex_to_rgb(hex_color)

        logger.debug("AnnotationRenderer initialized: %d classes", len(self._colors))

    def annotate_image(
        self,
        image: np.ndarray,
        detections: list[dict],
        show_confidence: bool = True,
        show_labels: bool = True,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the image.

        Parameters
        ----------
        image : np.ndarray
            RGB image to annotate, shape (H, W, 3).
        detections : list of dict
            Each dict must have: class (str), confidence (float),
            bbox_x, bbox_y, bbox_w, bbox_h (float).
        show_confidence : bool
            Show confidence percentage next to class label.
        show_labels : bool
            Show class name labels above bounding boxes.
        line_thickness : int
            Bounding box line thickness in pixels.
        font_scale : float
            Label font size scale.

        Returns
        -------
        np.ndarray
            Annotated copy of the image.
        """
        annotated = image.copy()

        for det in detections:
            cls = det.get("class", "unknown")
            conf = det.get("confidence", 0.0)
            bx = int(det.get("bbox_x", 0))
            by = int(det.get("bbox_y", 0))
            bw = int(det.get("bbox_w", 0))
            bh = int(det.get("bbox_h", 0))

            # Get color for this class (BGR for OpenCV)
            rgb = self._colors.get(cls, (200, 200, 200))
            bgr = (rgb[2], rgb[1], rgb[0])

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (bx, by),
                (bx + bw, by + bh),
                bgr,
                line_thickness,
            )

            # Draw label
            if show_labels:
                label = cls.replace("_", " ")
                if show_confidence:
                    label += f" {conf:.0%}"

                # Label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                label_y = max(by - 5, text_h + 5)

                cv2.rectangle(
                    annotated,
                    (bx, label_y - text_h - 5),
                    (bx + text_w + 4, label_y + 2),
                    bgr,
                    -1,  # Filled
                )

                # Label text (white on colored background)
                cv2.putText(
                    annotated,
                    label,
                    (bx + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        logger.debug("Annotated %d detections", len(detections))
        return annotated

    def create_overlay(
        self,
        image_shape: tuple[int, int, int],
        detections: list[dict],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Create a transparent overlay with just the bounding boxes.
        Can be composited on top of the base image with adjustable opacity.

        Parameters
        ----------
        image_shape : tuple
            Shape of the base image (H, W, 3).
        detections : list of dict
            Detection records.
        alpha : float
            Fill opacity for bounding boxes (0.0–1.0).

        Returns
        -------
        np.ndarray
            RGBA overlay image, shape (H, W, 4), dtype uint8.
        """
        h, w = image_shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        for det in detections:
            cls = det.get("class", "unknown")
            bx = int(det.get("bbox_x", 0))
            by = int(det.get("bbox_y", 0))
            bw = int(det.get("bbox_w", 0))
            bh = int(det.get("bbox_h", 0))

            rgb = self._colors.get(cls, (200, 200, 200))

            # Fill with semi-transparent color
            fill_alpha = int(alpha * 255)
            overlay[by:by+bh, bx:bx+bw, 0] = rgb[0]
            overlay[by:by+bh, bx:bx+bw, 1] = rgb[1]
            overlay[by:by+bh, bx:bx+bw, 2] = rgb[2]
            overlay[by:by+bh, bx:bx+bw, 3] = fill_alpha

            # Solid border
            cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (*rgb, 255), 2)

        return overlay

    def create_legend(self, classes: list[str] = None) -> dict[str, tuple[int, int, int]]:
        """
        Get the color legend for the current class-color mapping.

        Parameters
        ----------
        classes : list of str, optional
            Subset of classes to include. If None, include all.

        Returns
        -------
        dict
            {class_name: (R, G, B)} mapping.
        """
        if classes is None:
            return dict(self._colors)
        return {cls: self._colors.get(cls, (200, 200, 200)) for cls in classes}

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
