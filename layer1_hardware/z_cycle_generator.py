"""
Z-Cycle Generator
=================
Computes Z-motor target positions for continuous-mode scanning.

In continuous mode the Z motor oscillates through `frames_per_field`
discrete depths while X moves at constant velocity. This module is the
single source of truth for:

    1. The depth-index sequence within one field's cycle (sawtooth or
       triangle waveform), so the camera knows which Z-bucket each
       captured frame belongs to.
    2. The absolute Z positions (in microns) for those depths, optionally
       tracking a polynomial focus surface so the cycle stays centered
       on the predicted best-focus plane as (x, y) changes.

All functions here are pure — no motor I/O, no camera calls, no state.
The continuous_scan_controller calls them; tests can call them directly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Focus surface evaluation
# ---------------------------------------------------------------------------

def evaluate_focus_surface(
    x_um: float,
    y_um: float,
    coefficients: Sequence[float] | np.ndarray,
) -> float:
    """
    Evaluate the 2nd-order polynomial focus surface at (x, y).

        z(x, y) = a + b*x + c*y + d*x² + e*y² + f*x*y

    Parameters
    ----------
    x_um, y_um : float
        Stage position in microns.
    coefficients : sequence of 6 floats
        Polynomial coefficients [a, b, c, d, e, f] from FocusMapResult.

    Returns
    -------
    float
        Predicted best-focus Z in microns at the given (x, y).
    """
    if len(coefficients) != 6:
        raise ValueError(
            f"Focus surface needs exactly 6 coefficients [a,b,c,d,e,f]; "
            f"got {len(coefficients)}"
        )
    a, b, c, d, e, f = (float(v) for v in coefficients)
    return a + b * x_um + c * y_um + d * x_um * x_um + e * y_um * y_um + f * x_um * y_um


# ---------------------------------------------------------------------------
# Depth offsets (the Z values relative to the cycle center)
# ---------------------------------------------------------------------------

def generate_depth_offsets(n_depths: int, total_range_um: float) -> list[float]:
    """
    Build the list of Z offsets (relative to cycle center) for one cycle.

    For n_depths=6 and total_range_um=2.5, returns
    [-1.25, -0.75, -0.25, +0.25, +0.75, +1.25].

    The offsets are evenly spaced and symmetric around zero. They do not
    include both endpoints of the range (a half-step inset on each side)
    so adjacent cycles in a triangle waveform don't double-sample the
    extreme depths.
    """
    if n_depths < 2:
        raise ValueError(f"n_depths must be >= 2; got {n_depths}")
    if total_range_um <= 0:
        raise ValueError(f"total_range_um must be > 0; got {total_range_um}")

    step = total_range_um / n_depths
    half = total_range_um / 2.0
    return [(-half + step / 2.0) + i * step for i in range(n_depths)]


# ---------------------------------------------------------------------------
# Cycle ordering (sawtooth vs triangle)
# ---------------------------------------------------------------------------

def depth_index_sequence(
    n_depths: int,
    cycle_index: int,
    waveform: str = "triangle",
) -> list[int]:
    """
    Return the order in which the n_depths buckets are visited during
    cycle number `cycle_index`.

    Sawtooth: every cycle traverses 0 → n_depths-1, then snaps back.
    Triangle: even cycles go 0 → n_depths-1, odd cycles go n_depths-1 → 0.
    Triangle is recommended because the motor never has to reset to the
    bottom of its range — it just reverses at the top — which halves the
    number of direction-reversal events (and the associated backlash and
    wear).

    Parameters
    ----------
    n_depths : int
        Number of Z-depth buckets per cycle.
    cycle_index : int
        Which cycle this is (0, 1, 2, ...). For continuous scanning, one
        cycle == one virtual field.
    waveform : str
        "triangle" or "sawtooth".

    Returns
    -------
    list[int]
        Depth indices in capture order, length n_depths. Each entry is
        in [0, n_depths-1].
    """
    if n_depths < 2:
        raise ValueError(f"n_depths must be >= 2; got {n_depths}")

    base = list(range(n_depths))

    if waveform == "sawtooth":
        return base
    if waveform == "triangle":
        return base if (cycle_index % 2 == 0) else list(reversed(base))
    raise ValueError(f"Unknown waveform '{waveform}'; expected 'triangle' or 'sawtooth'")


# ---------------------------------------------------------------------------
# Combined: target Z positions for the next cycle
# ---------------------------------------------------------------------------

def generate_cycle_targets(
    x_um: float,
    y_um: float,
    coefficients: Sequence[float] | np.ndarray | None,
    n_depths: int,
    total_range_um: float,
    cycle_index: int,
    waveform: str = "triangle",
    fallback_z_center_um: float = 0.0,
) -> list[tuple[int, float]]:
    """
    Compute the (depth_index, z_um) pairs for one full Z cycle, in
    capture order.

    If `coefficients` is None or `z_tracking_enabled` is off, the cycle
    centers on `fallback_z_center_um` (typically the system's reference
    Z plane). Otherwise the cycle centers on the predicted focus surface
    at (x_um, y_um).

    Parameters
    ----------
    x_um, y_um : float
        Stage position at the start of this cycle, in microns.
    coefficients : sequence of 6 floats, or None
        Focus surface polynomial. None disables surface tracking.
    n_depths : int
        Frames per field (typically 6).
    total_range_um : float
        Total Z sweep range (e.g. 2.5 = ±1.25 μm around center).
    cycle_index : int
        Which cycle this is (0, 1, 2, ...). Determines triangle direction.
    waveform : str
        "triangle" or "sawtooth".
    fallback_z_center_um : float
        Z center to use when surface tracking is disabled.

    Returns
    -------
    list of (depth_index, z_um_absolute)
        Length n_depths. depth_index is the bucket id (0..n_depths-1)
        used to group frames in the stacker; z_um_absolute is the motor
        target for that capture.
    """
    if coefficients is None:
        z_center = fallback_z_center_um
    else:
        z_center = evaluate_focus_surface(x_um, y_um, coefficients)

    offsets = generate_depth_offsets(n_depths, total_range_um)
    order = depth_index_sequence(n_depths, cycle_index, waveform)

    return [(d, z_center + offsets[d]) for d in order]
