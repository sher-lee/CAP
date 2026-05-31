"""
Canvas Placer
=============
Places focus-stacked composites onto a master slide canvas using
position tags from continuous-mode capture.

Two design choices that distinguish this from the existing `stitcher.py`:

1. **Direct placement, not pairwise stitch.** Each field's (x_um, y_um)
   tag from open-loop step counts is treated as authoritative. Fields
   are dropped onto the canvas at their tagged pixel coordinates. This
   avoids running cross-correlation against every neighbor; the
   step-counter prior is good enough for placement, and any residual
   per-field misregistration was already absorbed by phase correlation
   inside the focus stacker.

2. **No feathering. Sharpness-pick in overlap zones.** Where two fields
   overlap, the canvas keeps the per-block sharper of the two — same
   rule the focus stacker uses for in-field Z arbitration. The
   rationale is clinical: organisms sit at varied focal planes, and
   the sharper of two overlapping captures is the one that has the
   organism in focus at that location. Blending or center-distance
   bias would dilute that signal.

   If checkerboard artifacts caused by per-block source flipping become
   visually objectionable, the fix is illumination normalization on raw
   frames *before* stacking — not relaxing this rule.

Status: Chunk 1 single-process in-memory canvas. DZI tile generation,
PDF integration, and on-disk canvas paging are out of scope here; the
existing tile_builder.py + pdf_report.py handle those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cap.common.dataclasses import StackedField
from cap.common.logging_setup import get_logger

logger = get_logger("canvas")


@dataclass
class CanvasPlacementResult:
    """Per-placement diagnostics — returned by MasterCanvas.place()."""
    placed: bool
    blocks_painted: int
    """Blocks that won arbitration and were painted onto the canvas."""
    blocks_skipped: int
    """Blocks that lost to a previously-placed sharper block."""
    px_x: int
    """Canvas pixel x of the field's top-left corner after snapping."""
    px_y: int
    """Canvas pixel y of the field's top-left corner after snapping."""


class MasterCanvas:
    """
    In-memory whole-slide canvas with per-block sharpness arbitration.

    Block size is set at construction (matches the focus stacker's
    block_size for consistent arbitration granularity). Field placement
    snaps to canvas-block-aligned pixel coordinates so that every
    placed block aligns 1:1 with a canvas block; no resampling or
    interpolation is performed on placement.

    The sharpness ledger is one float32 per canvas block, holding the
    sharpness score of whatever field currently owns that block. New
    placements compare per-block and overwrite only when strictly
    greater — ties keep the existing block (first-placed wins).
    """

    def __init__(
        self,
        canvas_width_px: int,
        canvas_height_px: int,
        px_per_um: float,
        block_size: int = 16,
        channels: int = 3,
        dtype: np.dtype = np.uint8,
    ) -> None:
        if canvas_width_px <= 0 or canvas_height_px <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if px_per_um <= 0:
            raise ValueError("px_per_um must be positive")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")

        # Round canvas to whole blocks
        bs = block_size
        self._blocks_x = (canvas_width_px + bs - 1) // bs
        self._blocks_y = (canvas_height_px + bs - 1) // bs
        self._w = self._blocks_x * bs
        self._h = self._blocks_y * bs

        self._block_size = bs
        self._px_per_um = px_per_um
        self._channels = channels

        if channels == 3:
            self._image = np.zeros((self._h, self._w, 3), dtype=dtype)
        else:
            self._image = np.zeros((self._h, self._w), dtype=dtype)

        # Per-block sharpness claim. -inf means "unclaimed"; first placement
        # always wins because any finite sharpness > -inf.
        self._sharpness = np.full(
            (self._blocks_y, self._blocks_x), -np.inf, dtype=np.float32,
        )
        # Per-block source tracking (slide_id, field_x_steps, field_y_steps)
        # — handy for diagnostics. Encoded as -1 sentinels until claimed.
        self._owner_field_x = np.full((self._blocks_y, self._blocks_x), -1, dtype=np.int64)
        self._owner_field_y = np.full((self._blocks_y, self._blocks_x), -1, dtype=np.int64)

        logger.info(
            "MasterCanvas initialized: %d×%d px (%d×%d blocks of %d), "
            "%.4f px/um, %d channels, dtype=%s",
            self._w, self._h, self._blocks_x, self._blocks_y, bs,
            px_per_um, channels, dtype,
        )

    # ----- Geometry -----

    @property
    def width_px(self) -> int:
        return self._w

    @property
    def height_px(self) -> int:
        return self._h

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def image(self) -> np.ndarray:
        """Direct view of the canvas image (read-write — caller beware)."""
        return self._image

    @property
    def sharpness_map(self) -> np.ndarray:
        """Per-block sharpness ledger; finite values mean 'claimed'."""
        return self._sharpness

    def um_to_px(self, x_um: float, y_um: float) -> tuple[int, int]:
        """Convert microns to canvas pixel coordinates (origin = canvas top-left)."""
        return (int(round(x_um * self._px_per_um)), int(round(y_um * self._px_per_um)))

    # ----- Placement -----

    def place(
        self,
        stacked: StackedField,
        x_um: float,
        y_um: float,
    ) -> CanvasPlacementResult:
        """
        Drop a focus-stacked composite onto the canvas at (x_um, y_um),
        which is the field's CENTER. Per-block sharpness arbitration
        keeps the sharper source where the new field overlaps existing
        canvas content.

        Parameters
        ----------
        stacked : StackedField
            From FocusStacker.stack(). Must have a sharpness_map whose
            block_size matches this canvas's block_size; if not, the
            placement is rejected (no graceful resampling in Chunk 1).
        x_um, y_um : float
            Field center in microns from the slide origin. Top-left of
            the placed field is then (x_um - W/2, y_um - H/2).

        Returns
        -------
        CanvasPlacementResult
            Block-level diagnostics. `placed=False` means rejection
            (mismatched block size, off-canvas, etc.).
        """
        if stacked.block_size != self._block_size:
            logger.error(
                "Cannot place: stacked block_size=%d, canvas block_size=%d",
                stacked.block_size, self._block_size,
            )
            return CanvasPlacementResult(False, 0, 0, 0, 0)

        comp = stacked.composite
        sm = stacked.sharpness_map
        if comp is None or sm is None:
            logger.warning("Cannot place: missing composite or sharpness_map")
            return CanvasPlacementResult(False, 0, 0, 0, 0)

        # Center-to-top-left
        comp_h, comp_w = comp.shape[:2]
        top_left_x_um = x_um - (comp_w / 2.0) / self._px_per_um
        top_left_y_um = y_um - (comp_h / 2.0) / self._px_per_um
        px_x, px_y = self.um_to_px(top_left_x_um, top_left_y_um)

        # Snap to canvas block grid so block-level arbitration is exact
        bs = self._block_size
        snapped_x = (px_x // bs) * bs
        snapped_y = (px_y // bs) * bs

        # Field's block-grid extent
        field_blocks_y, field_blocks_x = sm.shape
        # Limit to canvas extent
        max_bx = min(field_blocks_x, self._blocks_x - snapped_x // bs)
        max_by = min(field_blocks_y, self._blocks_y - snapped_y // bs)
        start_bx = max(0, -(snapped_x // bs))
        start_by = max(0, -(snapped_y // bs))

        if max_bx <= start_bx or max_by <= start_by:
            logger.warning(
                "Field at (%.2f, %.2f) μm fully off-canvas — skipping",
                x_um, y_um,
            )
            return CanvasPlacementResult(False, 0, 0, 0, snapped_x, snapped_y)

        painted = 0
        skipped = 0
        is_color = (comp.ndim == 3)

        for by in range(start_by, max_by):
            cy = (snapped_y // bs) + by
            if cy < 0 or cy >= self._blocks_y:
                continue
            for bx in range(start_bx, max_bx):
                cx = (snapped_x // bs) + bx
                if cx < 0 or cx >= self._blocks_x:
                    continue

                incoming = float(sm[by, bx])
                current = float(self._sharpness[cy, cx])

                if incoming <= current:
                    skipped += 1
                    continue

                # Paint this block
                src_y0 = by * bs
                src_x0 = bx * bs
                dst_y0 = cy * bs
                dst_x0 = cx * bs
                if is_color:
                    self._image[dst_y0:dst_y0 + bs, dst_x0:dst_x0 + bs, :] = (
                        comp[src_y0:src_y0 + bs, src_x0:src_x0 + bs, :]
                    )
                else:
                    self._image[dst_y0:dst_y0 + bs, dst_x0:dst_x0 + bs] = (
                        comp[src_y0:src_y0 + bs, src_x0:src_x0 + bs]
                    )
                self._sharpness[cy, cx] = incoming
                self._owner_field_x[cy, cx] = stacked.field_x
                self._owner_field_y[cy, cx] = stacked.field_y
                painted += 1

        logger.debug(
            "Placed slide=%d field=(%d,%d) at canvas (%d,%d): %d blocks painted, %d skipped",
            stacked.slide_id, stacked.field_x, stacked.field_y,
            snapped_x, snapped_y, painted, skipped,
        )

        return CanvasPlacementResult(
            placed=True,
            blocks_painted=painted,
            blocks_skipped=skipped,
            px_x=snapped_x,
            px_y=snapped_y,
        )

    def coverage(self) -> float:
        """Fraction of canvas blocks that have been claimed by some field."""
        total = self._blocks_x * self._blocks_y
        if total == 0:
            return 0.0
        claimed = int(np.sum(np.isfinite(self._sharpness)))
        return claimed / total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canvas_for_slide(
    slide_width_mm: float,
    slide_height_mm: float,
    px_per_um: float,
    block_size: int = 16,
    channels: int = 3,
) -> MasterCanvas:
    """Convenience constructor sized to a standard slide."""
    canvas_w_px = int(math.ceil(slide_width_mm * 1000.0 * px_per_um))
    canvas_h_px = int(math.ceil(slide_height_mm * 1000.0 * px_per_um))
    return MasterCanvas(
        canvas_width_px=canvas_w_px,
        canvas_height_px=canvas_h_px,
        px_per_um=px_per_um,
        block_size=block_size,
        channels=channels,
    )
