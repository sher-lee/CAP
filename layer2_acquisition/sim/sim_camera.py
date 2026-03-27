"""
Simulated Camera Interface
============================
Drop-in replacement for the real CameraInterface (harvesters).
Returns test images from a directory, or generates procedural
synthetic microscopy patterns if no test images are available.

Supports the same public API as the real camera so the capture
sequencer works identically in both modes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("camera.sim")


class SimCameraInterface:
    """Simulated camera interface for Windows development."""

    def __init__(self, config: dict | object) -> None:
        """
        Initialize the simulated camera.

        Parameters
        ----------
        config : dict or CAPConfig
            Application configuration. Reads sim and camera sections.
        """
        if hasattr(config, "sim"):
            self._test_image_dir = config.sim.camera_test_image_dir
            self._generate_synthetic = config.sim.generate_synthetic
            self._width = config.sim.synthetic_image_width
            self._height = config.sim.synthetic_image_height
            self._bit_depth = config.camera.bit_depth
        else:
            sim = config.get("sim", {})
            cam = config.get("camera", {})
            self._test_image_dir = sim.get("camera_test_image_dir", "./tests/test_images/")
            self._generate_synthetic = sim.get("generate_synthetic", True)
            self._width = sim.get("synthetic_image_width", 3840)
            self._height = sim.get("synthetic_image_height", 2160)
            self._bit_depth = cam.get("bit_depth", 10)

        self._is_connected = False
        self._exposure = 10000
        self._gain = 0.0
        self._white_balance = (1.0, 1.0, 1.0)
        self._frame_counter = 0
        self._test_images: list[np.ndarray] = []

        logger.info(
            "SimCameraInterface initialized: %dx%d, %d-bit, test_dir=%s",
            self._width, self._height, self._bit_depth, self._test_image_dir,
        )

    # ----- Connection lifecycle -----

    def initialize(self) -> None:
        """Connect to the camera and configure parameters."""
        self._is_connected = True
        self._load_test_images()
        logger.info(
            "Camera connected (simulated): %d test images loaded",
            len(self._test_images),
        )

    def release(self) -> None:
        """Disconnect from the camera and free resources."""
        self._is_connected = False
        self._test_images.clear()
        self._frame_counter = 0
        logger.info("Camera released (simulated)")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ----- Configuration -----

    def set_exposure(self, value: int) -> None:
        """Set manual exposure time in microseconds."""
        self._exposure = value
        logger.debug("Exposure set to %d µs", value)

    def set_gain(self, value: float) -> None:
        """Set manual gain in dB (0–24)."""
        self._gain = value
        logger.debug("Gain set to %.1f dB", value)

    def set_white_balance(self, r: float, g: float, b: float) -> None:
        """Set manual white balance coefficients."""
        self._white_balance = (r, g, b)
        logger.debug("White balance set to R=%.2f G=%.2f B=%.2f", r, g, b)

    # ----- Capture -----

    def trigger_capture(self) -> np.ndarray:
        """
        Trigger a single frame capture.

        Returns
        -------
        np.ndarray
            Raw Bayer frame, shape (H, W).
            dtype is uint8 for 8-bit or uint16 for 10-bit.
        """
        if not self._is_connected:
            raise RuntimeError("Camera not connected. Call initialize() first.")

        if self._test_images:
            # Cycle through loaded test images
            frame = self._test_images[self._frame_counter % len(self._test_images)]
        else:
            # Generate synthetic frame
            frame = self._generate_synthetic_frame()

        self._frame_counter += 1
        logger.debug("Frame captured (simulated): #%d", self._frame_counter)
        return frame.copy()

    def get_frame_buffer(self) -> Optional[np.ndarray]:
        """
        Retrieve the latest frame from the buffer.
        In simulation, equivalent to trigger_capture().
        """
        return self.trigger_capture()

    @property
    def frame_count(self) -> int:
        """Total number of frames captured in this session."""
        return self._frame_counter

    @property
    def resolution(self) -> tuple[int, int]:
        """Camera resolution as (width, height)."""
        return (self._width, self._height)

    # ----- Internal: test image loading -----

    def _load_test_images(self) -> None:
        """Load test images from the configured directory, if it exists."""
        test_dir = Path(self._test_image_dir)
        if not test_dir.is_dir():
            logger.debug(
                "Test image directory not found: %s — will generate synthetic",
                self._test_image_dir,
            )
            return

        supported = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".npy"}
        image_files = sorted(
            f for f in test_dir.iterdir()
            if f.suffix.lower() in supported
        )

        if not image_files:
            logger.debug("No test images found in %s", self._test_image_dir)
            return

        for img_path in image_files:
            try:
                if img_path.suffix == ".npy":
                    img = np.load(str(img_path))
                else:
                    # Use OpenCV if available, fall back to basic loading
                    try:
                        import cv2
                        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                        if img is not None and len(img.shape) == 3:
                            # Convert to single channel (simulate Bayer)
                            img = img[:, :, 0]
                    except ImportError:
                        logger.debug("OpenCV not available; skipping %s", img_path.name)
                        continue

                if img is not None:
                    # Resize to configured resolution if needed
                    if img.shape != (self._height, self._width):
                        img = self._resize_simple(img, self._height, self._width)
                    self._test_images.append(img)
                    logger.debug("Loaded test image: %s (%s)", img_path.name, img.shape)
            except Exception as e:
                logger.warning("Failed to load test image %s: %s", img_path.name, e)

    # ----- Internal: synthetic image generation -----

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generate a procedural synthetic microscopy image.
        Creates a light background with randomly placed dark circles
        (simulating cocci), ovals (yeast), and lines (rods).
        """
        dtype = np.uint16 if self._bit_depth > 8 else np.uint8
        max_val = (2 ** self._bit_depth) - 1

        # Light pinkish background (typical stained slide)
        rng = np.random.default_rng(seed=self._frame_counter)
        base_level = int(max_val * 0.75)
        noise_std = int(max_val * 0.03)
        frame = rng.normal(base_level, noise_std, (self._height, self._width)).astype(np.float64)

        # Add random dark circles (cocci-like)
        n_cocci = rng.integers(10, 40)
        for _ in range(n_cocci):
            cx = rng.integers(20, self._width - 20)
            cy = rng.integers(20, self._height - 20)
            radius = rng.integers(3, 12)
            darkness = rng.uniform(0.15, 0.45)

            y_coords, x_coords = np.ogrid[
                max(0, cy - radius):min(self._height, cy + radius),
                max(0, cx - radius):min(self._width, cx + radius),
            ]
            mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) <= radius ** 2
            frame[
                max(0, cy - radius):min(self._height, cy + radius),
                max(0, cx - radius):min(self._width, cx + radius),
            ][mask] *= darkness

        # Add random elongated shapes (rod-like)
        n_rods = rng.integers(2, 10)
        for _ in range(n_rods):
            cx = rng.integers(30, self._width - 30)
            cy = rng.integers(30, self._height - 30)
            length = rng.integers(15, 40)
            width = rng.integers(2, 5)
            darkness = rng.uniform(0.2, 0.5)

            y_start = max(0, cy - width)
            y_end = min(self._height, cy + width)
            x_start = max(0, cx - length)
            x_end = min(self._width, cx + length)
            frame[y_start:y_end, x_start:x_end] *= darkness

        # Clip and convert to integer type
        frame = np.clip(frame, 0, max_val).astype(dtype)
        return frame

    @staticmethod
    def _resize_simple(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Simple nearest-neighbor resize without OpenCV dependency."""
        h, w = img.shape[:2]
        y_indices = (np.arange(target_h) * h / target_h).astype(int)
        x_indices = (np.arange(target_w) * w / target_w).astype(int)
        return img[np.ix_(y_indices, x_indices)]
