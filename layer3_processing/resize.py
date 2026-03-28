"""
Resize / Crop
==============
Resizes images to the dimensions expected by the AI model (640x640
for YOLOv11). This is a separate step from the stacking output —
the full-resolution stacked composite is preserved for the client-
facing visualization, while a resized copy is produced for inference.
"""

from __future__ import annotations

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("processing.resize")


def resize_for_inference(
    image: np.ndarray,
    target_width: int = 640,
    target_height: int = 640,
    maintain_aspect: bool = True,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Resize an image for AI model input.

    When maintain_aspect is True (default), the image is scaled to fit
    within the target dimensions while preserving aspect ratio, then
    padded with a neutral color. This matches YOLOv11's letterbox
    preprocessing.

    Parameters
    ----------
    image : np.ndarray
        RGB image, any size, dtype uint8.
    target_width, target_height : int
        Model input dimensions (default 640x640 for YOLOv11).
    maintain_aspect : bool
        If True, letterbox with padding. If False, stretch to fit.
    pad_color : tuple
        RGB color for letterbox padding (default gray 114).

    Returns
    -------
    np.ndarray
        Resized image, shape (target_height, target_width, 3), dtype uint8.
    """
    h, w = image.shape[:2]

    if not maintain_aspect:
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized

    # Letterbox resize: scale to fit, then pad
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)

    # Center the resized image on the canvas
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    logger.debug(
        "Resized for inference: (%d, %d) → (%d, %d) scale=%.3f, padded to (%d, %d)",
        w, h, new_w, new_h, scale, target_width, target_height,
    )

    return canvas


def resize_for_preview(
    image: np.ndarray,
    max_width: int = 960,
    max_height: int = 540,
) -> np.ndarray:
    """
    Resize an image for the live camera preview in the UI.
    Scales down while maintaining aspect ratio. Never scales up.

    Parameters
    ----------
    image : np.ndarray
        RGB image, any size.
    max_width, max_height : int
        Maximum preview dimensions.

    Returns
    -------
    np.ndarray
        Resized image, dtype uint8.
    """
    h, w = image.shape[:2]

    if w <= max_width and h <= max_height:
        return image

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_center(
    image: np.ndarray,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    """
    Crop the center region of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    crop_width, crop_height : int
        Dimensions of the crop region.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    h, w = image.shape[:2]
    x_start = max(0, (w - crop_width) // 2)
    y_start = max(0, (h - crop_height) // 2)

    return image[y_start:y_start + crop_height, x_start:x_start + crop_width]
