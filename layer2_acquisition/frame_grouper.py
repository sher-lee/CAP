"""
Frame Grouper
=============
Consumes a stream of position-tagged RawFrames produced by continuous
scanning and partitions them into VirtualFields — one per FOV-width of
X travel along each scan line. Each VirtualField holds the
frames_per_field captures (one per Z-depth) that the focus stacker then
turns into a single composite.

Two design points worth understanding:

1. **Spatial grouping, not control-flow grouping.** A frame's virtual
   field is determined by where it was captured (`motor_position`
   converted to microns, projected onto the scan-line origin), not by
   when in the Z-cycle the controller thought it was. This keeps the
   grouper honest if the Z-motor falls behind, frames are missed, or
   the X stage drifts off its commanded trajectory.

2. **Step-counter shifts become the stacker's prior.** Because frames
   within one virtual field are captured at slightly different X
   positions, frames 1..N drift by hundreds of pixels relative to
   frame 0 — way outside the FocusStacker's normal max_registration_shift.
   The grouper computes the expected (dx, dy) per frame in pixels using
   the open-loop step counts and emits them with the VirtualField, so
   the stacker can pre-warp and then phase-correlate around zero for
   sub-pixel residual correction only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from cap.common.dataclasses import RawFrame
from cap.common.logging_setup import get_logger

logger = get_logger("grouper")


# ---------------------------------------------------------------------------
# VirtualField — what the grouper emits to the focus stacker
# ---------------------------------------------------------------------------

@dataclass
class VirtualField:
    """
    A single field's worth of continuous-mode frames, ready for stacking.

    Lives entirely inside Layer 2 — does not cross layer boundaries.
    Crossing Layer 2 → Layer 5/7 still uses StackedField (unchanged).
    """
    scan_line: int
    scan_direction: int
    """+1 for left-to-right travel on this line, -1 for right-to-left."""
    field_index_in_line: int
    """0-based index of this virtual field within its scan line, in
    travel order (so the first emitted field on a line has index 0
    regardless of scan_direction)."""
    field_x_um: float
    """X coordinate of the field center, in microns. This is the value
    canvas placement uses to position the composite on the master canvas."""
    field_y_um: float
    """Y coordinate of the field center, in microns."""
    frames: list[RawFrame] = field(default_factory=list)
    """Frames in capture order (NOT depth-index order). Length is
    frames_per_field for a complete field; less if the line ended early
    or frames were dropped."""
    expected_shifts_px: list[tuple[float, float]] = field(default_factory=list)
    """Per-frame (dx, dy) shift in pixels relative to frames[0], computed
    from open-loop step counts. Same length as frames. The stacker uses
    this as a prior so phase correlation only has to find sub-pixel
    residuals instead of full multi-hundred-pixel offsets."""
    is_complete: bool = False
    """True if frames_per_field captures landed in this field. False
    means partial — the line ended or frames were lost mid-cycle."""

    @property
    def n_frames(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# Grouper
# ---------------------------------------------------------------------------

class FrameGrouper:
    """
    Streaming grouper. Call `begin_scan_line(...)` once per scan line,
    then `feed(...)` for each captured RawFrame. `feed` returns a
    completed VirtualField the moment one fills up; otherwise None.
    Call `flush_line()` at the end of a line to drain any partials.
    """

    def __init__(
        self,
        fov_width_um: float,
        fov_height_um: float,
        frames_per_field: int,
        x_steps_per_mm: float,
        y_steps_per_mm: float,
    ) -> None:
        if fov_width_um <= 0 or fov_height_um <= 0:
            raise ValueError("FOV dimensions must be positive")
        if frames_per_field < 1:
            raise ValueError("frames_per_field must be >= 1")

        self._fov_w_um = fov_width_um
        self._fov_h_um = fov_height_um
        self._n_per_field = frames_per_field
        self._x_steps_per_mm = x_steps_per_mm
        self._y_steps_per_mm = y_steps_per_mm

        # Per-line state, set by begin_scan_line()
        self._scan_line: int = -1
        self._scan_direction: int = 0
        self._line_origin_x_um: float = 0.0
        self._line_y_um: float = 0.0

        # Active buffer: keys are field_index_in_line → list of frames in arrival order
        self._buffers: dict[int, list[RawFrame]] = {}
        # Field index of the most recently observed frame; used to detect
        # "we've moved past field K" and emit any straggler partials.
        self._current_field_index: int = -1

        # Counters (diagnostics)
        self._fields_emitted_complete = 0
        self._fields_emitted_partial = 0
        self._frames_received = 0

        logger.debug(
            "FrameGrouper initialized: fov=%.2f×%.2f μm, frames_per_field=%d",
            fov_width_um, fov_height_um, frames_per_field,
        )

    # ----- Per-line lifecycle -----

    def begin_scan_line(
        self,
        scan_line: int,
        scan_direction: int,
        line_origin_x_um: float,
        line_y_um: float,
    ) -> list[VirtualField]:
        """
        Start a new scan line. Returns any partial fields left over from
        the previous line (so the caller can route them to retry / flag
        as incomplete).

        Parameters
        ----------
        scan_line : int
            Y row index, 0-based.
        scan_direction : int
            +1 for X increasing, -1 for X decreasing.
        line_origin_x_um : float
            X position (microns) at the start of this travel direction.
            For +1: leftmost X of the line. For -1: rightmost X.
        line_y_um : float
            Y position (microns) for this line.
        """
        if scan_direction not in (-1, 1):
            raise ValueError(f"scan_direction must be ±1; got {scan_direction}")

        leftover = self.flush_line()

        self._scan_line = scan_line
        self._scan_direction = scan_direction
        self._line_origin_x_um = line_origin_x_um
        self._line_y_um = line_y_um
        self._buffers.clear()
        self._current_field_index = -1

        logger.debug(
            "Begin scan line %d (dir=%+d, origin_x=%.2f, y=%.2f), "
            "carried over %d partial fields",
            scan_line, scan_direction, line_origin_x_um, line_y_um, len(leftover),
        )
        return leftover

    def flush_line(self) -> list[VirtualField]:
        """Emit any buffered fields from the current line, complete or partial."""
        if not self._buffers:
            return []

        out: list[VirtualField] = []
        for idx in sorted(self._buffers.keys()):
            vf = self._build_virtual_field(idx, self._buffers[idx])
            out.append(vf)
            if vf.is_complete:
                self._fields_emitted_complete += 1
            else:
                self._fields_emitted_partial += 1

        self._buffers.clear()
        self._current_field_index = -1
        return out

    # ----- Per-frame -----

    def feed(self, frame: RawFrame, frame_x_um: float, frame_y_um: float) -> Optional[VirtualField]:
        """
        Buffer a frame and emit a VirtualField if one just became complete.

        Returns None when the field this frame belongs to needs more frames.
        Returns a complete VirtualField when adding this frame made one
        complete. Partial fields are not emitted from feed() — call
        flush_line() at end-of-line to drain them.

        Parameters
        ----------
        frame : RawFrame
            The tagged frame from FrameTagger.tag().
        frame_x_um, frame_y_um : float
            X/Y position in microns at the moment this frame was captured.
            Caller converts from microsteps using frame_tagger helpers.
        """
        if self._scan_line < 0:
            raise RuntimeError("feed() called before begin_scan_line()")

        idx = self._field_index_for(frame_x_um)
        self._buffers.setdefault(idx, []).append(frame)
        self._current_field_index = idx
        self._frames_received += 1

        # If this fill just completed the field, emit it.
        if len(self._buffers[idx]) >= self._n_per_field:
            buf = self._buffers.pop(idx)
            vf = self._build_virtual_field(idx, buf)
            self._fields_emitted_complete += 1
            return vf

        return None

    # ----- Internals -----

    def _field_index_for(self, frame_x_um: float) -> int:
        """Map a frame's X position to a 0-based field index within the current line."""
        distance_into_line = self._scan_direction * (frame_x_um - self._line_origin_x_um)
        # Clamp negative distances (can happen if a frame arrives slightly
        # before the line origin due to motor settling) to field 0.
        if distance_into_line < 0:
            return 0
        return int(distance_into_line // self._fov_w_um)

    def _build_virtual_field(self, idx: int, frames: list[RawFrame]) -> VirtualField:
        """Assemble a VirtualField, populate field_x/field_y on each frame, and compute prior shifts."""
        center_offset_um = (idx + 0.5) * self._fov_w_um
        field_x_um = self._line_origin_x_um + self._scan_direction * center_offset_um
        field_y_um = self._line_y_um

        # Stamp the virtual field coordinates onto each frame for downstream
        # consumers (Layer 5 stores them, Layer 7 uses them for placement).
        # Convert back to microsteps so the existing field_x/field_y
        # contract (motor coordinates, microsteps) is preserved.
        fx_steps = int(round(field_x_um * self._x_steps_per_mm / 1000.0))
        fy_steps = int(round(field_y_um * self._y_steps_per_mm / 1000.0))
        for fr in frames:
            fr.field_x = fx_steps
            fr.field_y = fy_steps

        expected_shifts_px = self._compute_prior_shifts(frames)

        return VirtualField(
            scan_line=self._scan_line,
            scan_direction=self._scan_direction,
            field_index_in_line=idx,
            field_x_um=field_x_um,
            field_y_um=field_y_um,
            frames=frames,
            expected_shifts_px=expected_shifts_px,
            is_complete=(len(frames) >= self._n_per_field),
        )

    def _compute_prior_shifts(self, frames: list[RawFrame]) -> list[tuple[float, float]]:
        """
        Compute (dx, dy) in pixels for each frame relative to frames[0],
        derived purely from open-loop step counts. The stacker uses these
        as a prior to pre-warp before phase-correlating for residuals.

        Returns a list the same length as `frames`, with frames[0] always
        at (0.0, 0.0).
        """
        if not frames:
            return []

        # Microns-per-pixel from the actual frame width. Falls back to a
        # zero shift if we can't determine pixel scale (no image data, etc.).
        bayer = frames[0].bayer_data
        if bayer is None or bayer.ndim < 2 or bayer.shape[1] == 0:
            return [(0.0, 0.0)] * len(frames)
        px_per_um_x = bayer.shape[1] / self._fov_w_um
        px_per_um_y = bayer.shape[0] / self._fov_h_um

        x0_steps, y0_steps, _ = frames[0].motor_position
        x0_um = (x0_steps / self._x_steps_per_mm) * 1000.0
        y0_um = (y0_steps / self._y_steps_per_mm) * 1000.0

        shifts: list[tuple[float, float]] = []
        for fr in frames:
            xs, ys, _ = fr.motor_position
            x_um = (xs / self._x_steps_per_mm) * 1000.0
            y_um = (ys / self._y_steps_per_mm) * 1000.0
            # Sign convention matches cv2.phaseCorrelate + cv2.warpAffine:
            # when the stage moves +Δx in world coords between frame 0 and
            # frame i, frame i's content shifts -Δx·px_per_um in the image.
            # To align frame i to frame 0 we then translate frame i by
            # +Δx·px_per_um via warpAffine — i.e. the prior dx is positive.
            dx_px = (x_um - x0_um) * px_per_um_x
            dy_px = (y_um - y0_um) * px_per_um_y
            shifts.append((dx_px, dy_px))

        return shifts

    # ----- Diagnostics -----

    def stats(self) -> dict:
        return {
            "frames_received": self._frames_received,
            "fields_emitted_complete": self._fields_emitted_complete,
            "fields_emitted_partial": self._fields_emitted_partial,
            "buffered_partial_fields": len(self._buffers),
        }


# ---------------------------------------------------------------------------
# Batch helper (for tests / offline reprocessing)
# ---------------------------------------------------------------------------

def group_frames_batch(
    frames: list[RawFrame],
    line_origins_um: dict[int, tuple[float, float, int]],
    fov_width_um: float,
    fov_height_um: float,
    frames_per_field: int,
    x_steps_per_mm: float,
    y_steps_per_mm: float,
) -> Iterator[VirtualField]:
    """
    Group an already-collected stream of frames into VirtualFields.

    Useful for unit testing and for offline reprocessing of a saved scan.
    Real-time scans should use FrameGrouper directly.

    Parameters
    ----------
    frames : list of RawFrame
        Already tagged with scan_line, scan_direction, motor_position.
    line_origins_um : dict[int, tuple[float, float, int]]
        Maps scan_line → (origin_x_um, y_um, scan_direction). Required
        because the stream alone doesn't tell you where each line started.
    """
    # Group input frames by scan_line first (preserve arrival order within line)
    by_line: dict[int, list[RawFrame]] = {}
    for fr in frames:
        by_line.setdefault(fr.scan_line, []).append(fr)

    grouper = FrameGrouper(
        fov_width_um=fov_width_um,
        fov_height_um=fov_height_um,
        frames_per_field=frames_per_field,
        x_steps_per_mm=x_steps_per_mm,
        y_steps_per_mm=y_steps_per_mm,
    )

    for line in sorted(by_line.keys()):
        if line not in line_origins_um:
            logger.warning("No line origin provided for scan_line=%d; skipping", line)
            continue
        origin_x_um, y_um, direction = line_origins_um[line]
        for vf in grouper.begin_scan_line(line, direction, origin_x_um, y_um):
            yield vf
        for fr in by_line[line]:
            xs, ys, _ = fr.motor_position
            x_um = (xs / x_steps_per_mm) * 1000.0
            y_um_frame = (ys / y_steps_per_mm) * 1000.0
            vf = grouper.feed(fr, x_um, y_um_frame)
            if vf is not None:
                yield vf

    # Final flush
    yield from grouper.flush_line()
