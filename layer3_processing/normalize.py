"""
Normalization
==============
Brightness/contrast normalization and white balance correction.
Ensures consistent appearance across all fields in a slide,
which is critical for both client-facing image quality and
AI inference consistency.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("processing.normalize")


class SlideNormalizer:
    """
    Normalizes brightness, contrast, and white balance across a slide.

    Maintains a running reference computed from the first N fields,
    then normalizes subsequent fields to match that reference.
    This ensures uniform appearance across the stitched composite.
    """

    def __init__(
        self,
        target_mean: float = 160.0,
        target_std: float = 40.0,
        reference_frames: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        target_mean : float
            Target mean brightness (0-255). Default 160 gives a
            bright, well-exposed appearance.
        target_std : float
            Target standard deviation. Controls contrast.
        reference_frames : int
            Number of initial frames to average for computing
            the slide-wide reference.
        """
        self._target_mean = target_mean
        self._target_std = target_std
        self._ref_count = reference_frames

        # Running reference stats
        self._ref_means: list[float] = []
        self._ref_stds: list[float] = []
        self._slide_mean: Optional[float] = None
        self._slide_std: Optional[float] = None

        logger.debug(
            "SlideNormalizer initialized: target_mean=%.0f, target_std=%.0f",
            target_mean, target_std,
        )

    def reset(self) -> None:
        """Reset the reference statistics for a new slide."""
        self._ref_means.clear()
        self._ref_stds.clear()
        self._slide_mean = None
        self._slide_std = None

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image's brightness and contrast.

        The first few frames build the reference; subsequent frames
        are normalized to match.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        np.ndarray
            Normalized image, same shape and dtype.
        """
        img_float = image.astype(np.float32)
        mean = np.mean(img_float)
        std = np.std(img_float)

        # Build reference from first N frames
        if len(self._ref_means) < self._ref_count:
            self._ref_means.append(mean)
            self._ref_stds.append(std)

            if len(self._ref_means) == self._ref_count:
                self._slide_mean = np.mean(self._ref_means)
                self._slide_std = np.mean(self._ref_stds)
                logger.debug(
                    "Reference established: mean=%.1f, std=%.1f",
                    self._slide_mean, self._slide_std,
                )

        # Normalize to target statistics
        if std > 0:
            normalized = (img_float - mean) / std * self._target_std + self._target_mean
        else:
            normalized = np.full_like(img_float, self._target_mean)

        return np.clip(normalized, 0, 255).astype(np.uint8)


def normalize_brightness(
    image: np.ndarray,
    target_mean: float = 160.0,
    target_std: float = 40.0,
) -> np.ndarray:
    """
    Standalone brightness/contrast normalization for a single image.
    Use SlideNormalizer for consistent normalization across a slide.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    target_mean : float
        Target mean brightness.
    target_std : float
        Target standard deviation.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    img_float = image.astype(np.float32)
    mean = np.mean(img_float)
    std = np.std(img_float)

    if std > 0:
        normalized = (img_float - mean) / std * target_std + target_mean
    else:
        normalized = np.full_like(img_float, target_mean)

    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_white_balance(
    image: np.ndarray,
    r_gain: float = 1.0,
    g_gain: float = 1.0,
    b_gain: float = 1.0,
) -> np.ndarray:
    """
    Apply manual white balance gains to an RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    r_gain, g_gain, b_gain : float
        Per-channel gain multipliers. 1.0 = no change.

    Returns
    -------
    np.ndarray
        White-balanced image.
    """
    if r_gain == 1.0 and g_gain == 1.0 and b_gain == 1.0:
        return image

    result = image.astype(np.float32)
    result[:, :, 0] *= r_gain  # R channel
    result[:, :, 1] *= g_gain  # G channel
    result[:, :, 2] *= b_gain  # B channel

    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for local contrast enhancement. Useful for slides with uneven
    staining intensity.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image.
    """
    # Convert to LAB color space — equalize L channel only
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
