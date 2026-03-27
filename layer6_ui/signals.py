
"""
CAP UI Signal Definitions
============================
All Qt signals used for communication between the UI thread
and worker threads. Defined centrally so both sides import
from the same place.

Signals use PySide6's Signal class. They are emitted by worker
threads and received by UI slots via Qt's queued connection
(thread-safe by default when connecting across threads).

Usage:
    from cap.layer6_ui.signals import ScanSignals
    signals = ScanSignals()
    signals.progress.connect(self.on_scan_progress)
    signals.progress.emit(scan_progress_object)
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class ScanSignals(QObject):
    """Signals emitted during the scan lifecycle."""

    # Scan control
    scan_start_requested = Signal(object)       # ScanRegion
    scan_pause_requested = Signal()
    scan_resume_requested = Signal()
    scan_stop_requested = Signal()

    # Scan progress (emitted by capture sequencer)
    progress = Signal(object)                   # ScanProgress dataclass
    field_captured = Signal(int, int)            # field_x, field_y
    field_stacked = Signal(int, int)             # field_x, field_y

    # Scan completion
    scan_complete = Signal(int)                  # slide_id
    scan_failed = Signal(str)                    # error message


class InferenceSignals(QObject):
    """Signals emitted during AI inference."""

    inference_started = Signal(int)              # slide_id
    inference_progress = Signal(int, int)        # fields_done, fields_total
    inference_complete = Signal(int, object)     # slide_id, SlideResults
    inference_failed = Signal(str)               # error message
    inference_skipped = Signal(str)              # reason (e.g. "AI disabled")


class MotorSignals(QObject):
    """Signals for motor state updates."""

    position_updated = Signal(int, int, int)     # x, y, z
    homing_complete = Signal()
    emergency_stop_triggered = Signal()
    emergency_stop_cleared = Signal()


class FocusSignals(QObject):
    """Signals for focus operations."""

    preliminary_focus_started = Signal()
    preliminary_focus_complete = Signal(object)  # FocusMapResult
    preliminary_focus_failed = Signal(str)       # error message
    drift_warning = Signal(int, int, float)      # field_x, field_y, drift_amount


class SystemSignals(QObject):
    """Signals for system-level events."""

    error_occurred = Signal(str, str, str)       # layer, message, severity
    oil_warning = Signal(str)                    # warning message
    thermal_warning = Signal(int)                # temperature_c
    thermal_pause = Signal(int)                  # temperature_c
    disk_usage_warning = Signal(float)           # usage_gb


class NavigationSignals(QObject):
    """Signals for screen navigation."""

    go_to_screen = Signal(str)                   # screen name
    go_back = Signal()
