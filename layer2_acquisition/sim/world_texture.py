"""
World Texture (Sim)
===================
Deterministic position-aware microscopy texture used by SimCameraInterface
in continuous mode. Given a world position (x_um, y_um, z_um) and a
camera resolution + pixel scale, returns the frame the camera would
"see" at that position.

Key properties:

  - **Spatially deterministic.** Same world (x, y) → identical pixel
    content (modulo Z-focus effects). Two adjacent fields with 10%
    overlap will share content in the overlap region, exactly as a real
    slide does. This is what makes the canvas placer's overlap
    arbitration testable.

  - **Z-focus simulated.** Each (x, y) location has an "ideal Z" derived
    from a smooth tilt across the slide. Frames at z != z_ideal are
    blurred by an amount proportional to the distance, simulating
    out-of-focus capture. The focus stacker can then pick the sharpest
    block per location.

  - **No external assets required.** Pattern is procedural — combinations
    of low-frequency cell-like blobs and high-frequency noise — so the
    sim runs without test images. Frequencies are tuned to give
    cv2.phaseCorrelate enough texture to lock onto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorldTextureConfig:
    """Tunables for the procedural slide texture."""
    blob_density_per_um2: float = 0.15
    """Roughly how many simulated cells per μm² across the whole slide."""
    blob_radius_um_min: float = 0.1
    blob_radius_um_max: float = 0.5
    z_tilt_um_per_mm_x: float = 0.5
    """Slide tilt: the ideal-focus Z plane changes this much per mm of X."""
    z_tilt_um_per_mm_y: float = -0.3
    z_curvature_um_per_mm2: float = 0.05
    """Slight quadratic curvature, so the focus surface isn't flat."""
    base_intensity: float = 0.78
    """Background intensity in [0, 1]."""
    noise_std: float = 0.04
    """Per-pixel noise (Gaussian σ in [0, 1]) before quantization."""
    blob_darkening_min: float = 0.25
    blob_darkening_max: float = 0.65
    """Multiplicative darkening for in-focus blobs."""
    seed: int = 0xC07A


class WorldTexture:
    """
    Procedural deterministic slide texture.

    The slide is divided into a coarse grid for blob generation. Each
    grid cell hashes deterministically to a fixed RNG state, so the
    same cell always produces the same blobs regardless of when or how
    often it's sampled.
    """

    # Coarse grid for blob layout (μm). Larger = more blobs share lookup work.
    _GRID_UM = 2.0

    def __init__(self, config: WorldTextureConfig | None = None) -> None:
        self._cfg = config or WorldTextureConfig()
        self._cell_blob_cache: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}

    # ----- Public API -----

    def render_frame(
        self,
        x_um_center: float,
        y_um_center: float,
        z_um: float,
        width_px: int,
        height_px: int,
        px_per_um: float,
        bit_depth: int = 8,
    ) -> np.ndarray:
        """
        Render the camera's view at (x_um_center, y_um_center, z_um).

        Returns
        -------
        np.ndarray of shape (height_px, width_px), dtype uint8 or uint16
        per bit_depth. Single-channel (Bayer-shaped, but content is
        already grayscale — this is sim).
        """
        # World extent of this frame, in μm
        half_w_um = (width_px / 2.0) / px_per_um
        half_h_um = (height_px / 2.0) / px_per_um
        x0_um = x_um_center - half_w_um
        x1_um = x_um_center + half_w_um
        y0_um = y_um_center - half_h_um
        y1_um = y_um_center + half_h_um

        max_val = (2 ** bit_depth) - 1
        cfg = self._cfg

        # Background + noise (deterministic on world position)
        bg_seed = self._world_seed(int(round(x_um_center * 100)), int(round(y_um_center * 100)))
        rng = np.random.default_rng(bg_seed)
        frame_f = np.full((height_px, width_px), cfg.base_intensity, dtype=np.float32)
        if cfg.noise_std > 0:
            frame_f += rng.normal(0.0, cfg.noise_std, size=frame_f.shape).astype(np.float32)

        # Find every blob whose grid cell intersects the FOV (with margin
        # for blob radius).
        margin = cfg.blob_radius_um_max
        gx0 = int(np.floor((x0_um - margin) / self._GRID_UM))
        gx1 = int(np.ceil((x1_um + margin) / self._GRID_UM))
        gy0 = int(np.floor((y0_um - margin) / self._GRID_UM))
        gy1 = int(np.ceil((y1_um + margin) / self._GRID_UM))

        blobs: list[tuple[float, float, float, float]] = []
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                blobs.extend(self._blobs_for_cell(gx, gy))

        # Render blobs (with Z-distance defocus)
        z_ideal = self._ideal_z_um(x_um_center, y_um_center)
        z_dist = abs(z_um - z_ideal)
        # Sharpness factor: 1.0 at perfect focus, decays smoothly with z distance
        sharpness = float(np.exp(-(z_dist ** 2) / (2 * 0.7 ** 2)))
        # When out of focus, blobs are dimmer/wider — simulate by reducing
        # darkening contrast and softening the edge falloff.
        contrast_scale = sharpness  # 1.0 sharp, 0 invisible
        edge_softness = 1.0 + 4.0 * (1.0 - sharpness)  # >=1; bigger = softer edge

        for bx_um, by_um, br_um, darkness in blobs:
            self._draw_blob(
                frame_f, bx_um, by_um, br_um, darkness,
                x0_um=x0_um, y0_um=y0_um, px_per_um=px_per_um,
                contrast_scale=contrast_scale,
                edge_softness=edge_softness,
            )

        np.clip(frame_f, 0.0, 1.0, out=frame_f)
        if bit_depth > 8:
            return (frame_f * max_val).astype(np.uint16)
        return (frame_f * max_val).astype(np.uint8)

    def ideal_z_um(self, x_um: float, y_um: float) -> float:
        """Public access to the slide's tilt model — useful for tests."""
        return self._ideal_z_um(x_um, y_um)

    # ----- Internals -----

    def _ideal_z_um(self, x_um: float, y_um: float) -> float:
        """Smooth quadratic tilt — 'the slide isn't flat'."""
        cfg = self._cfg
        x_mm = x_um / 1000.0
        y_mm = y_um / 1000.0
        return (
            cfg.z_tilt_um_per_mm_x * x_mm
            + cfg.z_tilt_um_per_mm_y * y_mm
            + cfg.z_curvature_um_per_mm2 * (x_mm * x_mm + y_mm * y_mm)
        )

    def _world_seed(self, *parts: int) -> int:
        """Hash arbitrary integers into a 64-bit RNG seed deterministically."""
        h = self._cfg.seed & 0xFFFFFFFFFFFFFFFF
        for p in parts:
            h = (h ^ (int(p) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            h = ((h << 13) | (h >> 51)) & 0xFFFFFFFFFFFFFFFF
            h ^= (h >> 7)
        return h & 0xFFFFFFFFFFFFFFFF

    def _blobs_for_cell(self, gx: int, gy: int) -> list[tuple[float, float, float, float]]:
        """Return [(x_um, y_um, radius_um, darkness)] for one grid cell, cached."""
        key = (gx, gy)
        cached = self._cell_blob_cache.get(key)
        if cached is not None:
            return cached

        cfg = self._cfg
        rng = np.random.default_rng(self._world_seed(gx, gy))
        cell_area_um2 = self._GRID_UM * self._GRID_UM
        expected = cfg.blob_density_per_um2 * cell_area_um2
        n_blobs = int(rng.poisson(expected))

        out: list[tuple[float, float, float, float]] = []
        for _ in range(n_blobs):
            bx = (gx + float(rng.random())) * self._GRID_UM
            by = (gy + float(rng.random())) * self._GRID_UM
            br = float(rng.uniform(cfg.blob_radius_um_min, cfg.blob_radius_um_max))
            dk = float(rng.uniform(cfg.blob_darkening_min, cfg.blob_darkening_max))
            out.append((bx, by, br, dk))

        self._cell_blob_cache[key] = out
        return out

    def _draw_blob(
        self,
        frame: np.ndarray,
        bx_um: float, by_um: float, br_um: float, darkness: float,
        *,
        x0_um: float, y0_um: float, px_per_um: float,
        contrast_scale: float, edge_softness: float,
    ) -> None:
        """Multiply a circular dark blob into `frame` (in-place, float32)."""
        if contrast_scale <= 0:
            return
        cx_px = (bx_um - x0_um) * px_per_um
        cy_px = (by_um - y0_um) * px_per_um
        r_px = br_um * px_per_um * edge_softness  # widen when defocused

        h, w = frame.shape
        # Bounding box in pixel coords (clamped to frame)
        ix0 = max(0, int(np.floor(cx_px - r_px)))
        ix1 = min(w, int(np.ceil(cx_px + r_px)) + 1)
        iy0 = max(0, int(np.floor(cy_px - r_px)))
        iy1 = min(h, int(np.ceil(cy_px + r_px)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return

        ys = np.arange(iy0, iy1, dtype=np.float32) - cy_px
        xs = np.arange(ix0, ix1, dtype=np.float32) - cx_px
        dist2 = (ys[:, None] ** 2) + (xs[None, :] ** 2)
        # Gaussian blob: 1 at center, ~0 at radius. Wider when defocused.
        sigma = max(r_px * 0.45, 1e-6)
        bell = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        # darkness in [0, 1]: 0 = pitch black, 1 = no effect
        # Effective multiplier per pixel = 1 - bell * (1 - darkness) * contrast_scale
        attenuation = 1.0 - bell * (1.0 - darkness) * contrast_scale
        frame[iy0:iy1, ix0:ix1] *= attenuation
