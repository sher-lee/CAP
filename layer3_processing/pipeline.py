"""
Processing Pipeline
=====================
Chains all image processing steps: debayer → normalize → denoise → resize.

Accepts raw Bayer frames (RawFrame) or already-stacked composites,
and produces processed RGB images (ProcessedFrame) ready for AI inference.

The pipeline can operate on individual frames or process an entire
slide's worth of stacked composites in batch.

Usage:
    from cap.layer3_processing.pipeline import ProcessingPipeline
    proc = ProcessingPipeline(config)
    processed = proc.process_frame(raw_bayer_frame)
    inference_ready = proc.process_for_inference(stacked_composite)
"""

from __future__ import annotations

import time

import numpy as np

from cap.common.dataclasses import ProcessedFrame, RawFrame
from cap.common.logging_setup import get_logger
from cap.layer3_processing.debayer import debayer
from cap.layer3_processing.normalize import (
    SlideNormalizer, apply_white_balance, normalize_brightness,
)
from cap.layer3_processing.denoise import denoise
from cap.layer3_processing.resize import resize_for_inference

logger = get_logger("processing.pipeline")


class ProcessingPipeline:
    """
    Full image processing chain from raw Bayer to inference-ready RGB.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "camera"):
            self._bayer_pattern = config.camera.bayer_pattern
            self._bit_depth = config.camera.bit_depth
            wb = config.camera.white_balance_rgb
            self._wb_r = wb.r
            self._wb_g = wb.g
            self._wb_b = wb.b
        else:
            cam = config.get("camera", {})
            self._bayer_pattern = cam.get("bayer_pattern", "RG")
            self._bit_depth = cam.get("bit_depth", 10)
            wb = cam.get("white_balance_rgb", {})
            self._wb_r = wb.get("r", 1.0)
            self._wb_g = wb.get("g", 1.0)
            self._wb_b = wb.get("b", 1.0)

        if hasattr(config, "processing"):
            self._filter_type = config.processing.noise_filter_type
            self._filter_kernel = config.processing.noise_filter_kernel
            self._model_w = config.processing.model_input_width
            self._model_h = config.processing.model_input_height
        else:
            proc = config.get("processing", {})
            self._filter_type = proc.get("noise_filter_type", "gaussian")
            self._filter_kernel = proc.get("noise_filter_kernel", 3)
            self._model_w = proc.get("model_input_width", 640)
            self._model_h = proc.get("model_input_height", 640)

        # Slide-level normalizer (maintains running reference)
        self._normalizer = SlideNormalizer()

        self._frames_processed = 0

        logger.info(
            "ProcessingPipeline initialized: bayer=%s %d-bit, "
            "filter=%s k=%d, model=%dx%d",
            self._bayer_pattern, self._bit_depth,
            self._filter_type, self._filter_kernel,
            self._model_w, self._model_h,
        )

    def reset_for_new_slide(self) -> None:
        """Reset the normalizer reference for a new slide."""
        self._normalizer.reset()
        self._frames_processed = 0
        logger.debug("Pipeline reset for new slide")

    def process_raw_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Process a raw Bayer frame through the full pipeline
        (debayer → white balance → normalize → denoise).

        Does NOT resize for inference — returns full resolution.
        Use process_for_inference() for the resized version.

        Parameters
        ----------
        raw_frame : np.ndarray
            Raw Bayer frame, shape (H, W), dtype uint8 or uint16.

        Returns
        -------
        np.ndarray
            Processed RGB image, shape (H, W, 3), dtype uint8.
        """
        # Step 1: Debayer
        rgb = debayer(raw_frame, pattern=self._bayer_pattern, bit_depth=self._bit_depth)

        # Step 2: White balance
        rgb = apply_white_balance(rgb, self._wb_r, self._wb_g, self._wb_b)

        # Step 3: Normalize brightness/contrast
        rgb = self._normalizer.normalize(rgb)

        # Step 4: Denoise
        rgb = denoise(rgb, filter_type=self._filter_type, kernel_size=self._filter_kernel)

        self._frames_processed += 1
        return rgb

    def process_stacked_composite(self, composite: np.ndarray) -> np.ndarray:
        """
        Process an already-stacked composite image.
        Applies normalization and denoising but skips debayering
        (the composite is already RGB from the focus stacker).

        Parameters
        ----------
        composite : np.ndarray
            Focus-stacked composite, shape (H, W, 3) or (H, W), dtype uint8.

        Returns
        -------
        np.ndarray
            Processed RGB image, full resolution, dtype uint8.
        """
        # Ensure 3-channel
        if composite.ndim == 2:
            rgb = np.stack([composite] * 3, axis=2)
        else:
            rgb = composite.copy()

        # Ensure uint8
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = (rgb / rgb.max() * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        # White balance
        rgb = apply_white_balance(rgb, self._wb_r, self._wb_g, self._wb_b)

        # Normalize
        rgb = self._normalizer.normalize(rgb)

        # Denoise
        rgb = denoise(rgb, filter_type=self._filter_type, kernel_size=self._filter_kernel)

        self._frames_processed += 1
        return rgb

    def process_for_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Resize a processed image for AI model input.
        Call this AFTER process_raw_frame() or process_stacked_composite().

        Parameters
        ----------
        image : np.ndarray
            Processed RGB image at full resolution.

        Returns
        -------
        np.ndarray
            Resized image, shape (model_h, model_w, 3), dtype uint8.
        """
        return resize_for_inference(
            image,
            target_width=self._model_w,
            target_height=self._model_h,
        )

    def process_raw_to_inference(self, raw_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience: process a raw frame and return both full-resolution
        and inference-ready versions.

        Parameters
        ----------
        raw_frame : np.ndarray
            Raw Bayer frame.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (full_resolution_rgb, inference_ready_rgb)
        """
        full_res = self.process_raw_frame(raw_frame)
        inference = self.process_for_inference(full_res)
        return full_res, inference

    def build_processed_frame(
        self,
        image: np.ndarray,
        slide_id: int,
        field_x: int,
        field_y: int,
        stacked: bool = True,
    ) -> ProcessedFrame:
        """
        Build a ProcessedFrame dataclass from a processed image.
        Computes the focus score (Laplacian variance) automatically.

        Parameters
        ----------
        image : np.ndarray
            Processed RGB image at inference resolution.
        slide_id : int
        field_x, field_y : int
        stacked : bool
            Whether this came from a focus-stacked composite.

        Returns
        -------
        ProcessedFrame
        """
        # Compute focus score
        gray = np.mean(image.astype(np.float64), axis=2) if image.ndim == 3 else image.astype(np.float64)
        h, w = gray.shape
        if h >= 3 and w >= 3:
            lap = (
                -4 * gray[1:-1, 1:-1]
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
            )
            focus_score = float(np.var(lap))
        else:
            focus_score = 0.0

        return ProcessedFrame(
            slide_id=slide_id,
            field_x=field_x,
            field_y=field_y,
            rgb_data=image,
            stacked=stacked,
            focus_score=focus_score,
        )

    @property
    def frames_processed(self) -> int:
        return self._frames_processed
