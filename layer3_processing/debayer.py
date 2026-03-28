"""
Debayering
===========
Converts raw Bayer pattern sensor data (RG8/RG10) to full-color RGB.
This must be the first processing step — raw Bayer data is unusable
for display, AI inference, or any other downstream operation.

The VEN-830 camera outputs Bayer RG format. Other patterns (BG, GR, GB)
are supported for future camera compatibility.
"""

from __future__ import annotations

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("processing.debayer")

# Mapping from pattern name to OpenCV conversion code
_BAYER_CODES = {
    "RG": cv2.COLOR_BayerRG2RGB,
    "BG": cv2.COLOR_BayerBG2RGB,
    "GR": cv2.COLOR_BayerGR2RGB,
    "GB": cv2.COLOR_BayerGB2RGB,
}


def debayer(
    raw_frame: np.ndarray,
    pattern: str = "RG",
    bit_depth: int = 10,
) -> np.ndarray:
    """
    Convert a raw Bayer frame to RGB.

    Parameters
    ----------
    raw_frame : np.ndarray
        Raw Bayer frame, shape (H, W). dtype uint8 for 8-bit,
        uint16 for 10-bit or higher.
    pattern : str
        Bayer pattern: "RG", "BG", "GR", or "GB".
        The VEN-830 uses "RG".
    bit_depth : int
        Sensor bit depth (8 or 10). If 10-bit, the frame is
        scaled to 8-bit after debayering for downstream compatibility.

    Returns
    -------
    np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.

    Raises
    ------
    ValueError
        If pattern is not recognized or frame has wrong dimensions.
    """
    if raw_frame.ndim != 2:
        # Already RGB or grayscale with channels — might be from simulation
        if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
            logger.debug("Frame already has 3 channels, skipping debayer")
            if raw_frame.dtype != np.uint8:
                return _scale_to_8bit(raw_frame, bit_depth)
            return raw_frame
        raise ValueError(
            f"Expected 2D Bayer frame, got shape {raw_frame.shape}"
        )

    pattern = pattern.upper()
    if pattern not in _BAYER_CODES:
        raise ValueError(
            f"Unknown Bayer pattern: '{pattern}'. "
            f"Must be one of: {list(_BAYER_CODES.keys())}"
        )

    code = _BAYER_CODES[pattern]

    # Handle bit depth
    if bit_depth > 8 and raw_frame.dtype == np.uint16:
        # Debayer at full bit depth, then scale to 8-bit
        rgb = cv2.cvtColor(raw_frame, code)
        rgb = _scale_to_8bit(rgb, bit_depth)
    elif raw_frame.dtype == np.uint16:
        # 16-bit data but unknown bit depth — scale from full range
        rgb = cv2.cvtColor(raw_frame, code)
        rgb = (rgb / 256).astype(np.uint8)
    else:
        # 8-bit input
        rgb = cv2.cvtColor(raw_frame, code)

    logger.debug(
        "Debayered: %s %d-bit (%s) → RGB %s",
        raw_frame.shape, bit_depth, pattern, rgb.shape,
    )

    return rgb


def _scale_to_8bit(frame: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Scale a higher-bit-depth frame to 8-bit uint8.

    Parameters
    ----------
    frame : np.ndarray
        Frame with dtype uint16 (or similar).
    bit_depth : int
        Original bit depth (e.g. 10, 12, 14, 16).

    Returns
    -------
    np.ndarray
        Scaled frame with dtype uint8.
    """
    max_val = (2 ** bit_depth) - 1
    if max_val == 0:
        max_val = 65535

    scaled = (frame.astype(np.float32) / max_val * 255).clip(0, 255).astype(np.uint8)
    return scaled
