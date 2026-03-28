"""
Noise Reduction
================
Applies configurable noise reduction filters to processed images.
Reduces sensor noise without destroying organism detail.
"""

from __future__ import annotations

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("processing.denoise")


def denoise(
    image: np.ndarray,
    filter_type: str = "gaussian",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply noise reduction to an image.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    filter_type : str
        "gaussian" for Gaussian blur, "median" for median filter,
        or "bilateral" for edge-preserving bilateral filter.
    kernel_size : int
        Filter kernel size. Must be odd (3, 5, 7).

    Returns
    -------
    np.ndarray
        Denoised image, same shape and dtype.

    Raises
    ------
    ValueError
        If filter_type is unknown or kernel_size is even.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {kernel_size}")

    if filter_type == "gaussian":
        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    elif filter_type == "median":
        result = cv2.medianBlur(image, kernel_size)

    elif filter_type == "bilateral":
        # Bilateral filter preserves edges while smoothing
        # d=kernel_size, sigmaColor=75, sigmaSpace=75
        result = cv2.bilateralFilter(image, kernel_size, 75, 75)

    elif filter_type == "none":
        return image

    else:
        raise ValueError(
            f"Unknown filter type: '{filter_type}'. "
            f"Must be 'gaussian', 'median', 'bilateral', or 'none'."
        )

    logger.debug("Denoised: %s kernel=%d", filter_type, kernel_size)
    return result
