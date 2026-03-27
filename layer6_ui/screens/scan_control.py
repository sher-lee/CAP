
"""
Screen 3: Scan Control Panel
===============================
Start/pause/stop controls, live progress bar, ETA,
stage position map, and camera preview placeholder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy,
    QMessageBox,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.scan_control")


class ScanControlScreen(QWidget):
    """Scan control panel with progress tracking."""

    def __init__(self, app_context: AppContext, nav_signals: NavigationSignals) -> None:
        super().__init__()
        self._ctx = app_context
        self._nav = nav_signals
        self._is_scanning = False
        self._is_paused = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(12)

        # Title
        title = QLabel("Scan Control")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # --- Main content grid ---
        content = QGridLayout()
        content.setSpacing(15)

        # Left: Controls and progress
        left_panel = QVBoxLayout()

        # Control buttons
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self._start_btn = QPushButton("▶  Start Scan")
        self._start_btn.setMinimumHeight(50)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #1D9E75; color: white; font-size: 16px; "
            "font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #0F6E56; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.setMinimumHeight(50)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setStyleSheet(
            "QPushButton { background-color: #BA7517; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #854F0B; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self._pause_btn.clicked.connect(self._on_pause)
        ctrl_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setMinimumHeight(50)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #E24B4A; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #A32D2D; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._stop_btn)

        left_panel.addWidget(ctrl_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumHeight(30)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("0 / 0 fields (%p%)")
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        # Stats row
        stats_layout = QHBoxLayout()

        self._fields_label = QLabel("Fields: 0 / 0")
        self._fields_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self._fields_label)

        self._eta_label = QLabel("ETA: —")
        self._eta_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self._eta_label)

        self._speed_label = QLabel("Speed: — fields/sec")
        self._speed_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self._speed_label)

        progress_layout.addLayout(stats_layout)
        left_panel.addWidget(progress_group)

        # Status section
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self._status_label = QLabel("Ready to scan")
        self._status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        status_layout.addWidget(self._status_label)

        self._detail_label = QLabel("Press Start to begin scanning the defined region")
        self._detail_label.setStyleSheet("font-size: 12px; color: gray;")
        self._detail_label.setWordWrap(True)
        status_layout.addWidget(self._detail_label)

        left_panel.addWidget(status_group)

        content.addLayout(left_panel, 0, 0)

        # Right: Stage map and camera preview
        right_panel = QVBoxLayout()

        # Stage position map (placeholder)
        map_group = QGroupBox("Stage Position")
        map_layout = QVBoxLayout(map_group)

        self._stage_map_placeholder = QLabel("[ Stage map will show here ]\n\n"
                                              "Green = scanned\n"
                                              "Blue = current position\n"
                                              "Gray = outside region")
        self._stage_map_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stage_map_placeholder.setMinimumSize(300, 200)
        self._stage_map_placeholder.setStyleSheet(
            "background-color: #f5f5f0; border: 1px solid #ccc; border-radius: 4px; "
            "color: #888; font-size: 12px;"
        )
        map_layout.addWidget(self._stage_map_placeholder)

        right_panel.addWidget(map_group)

        # Camera preview (placeholder)
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_placeholder = QLabel("[ Live camera preview ]\n(simulation mode: no camera)")
        self._preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_placeholder.setMinimumSize(300, 180)
        self._preview_placeholder.setStyleSheet(
            "background-color: #2c2c2a; color: #888; border-radius: 4px; font-size: 12px;"
        )
        preview_layout.addWidget(self._preview_placeholder)

        right_panel.addWidget(preview_group)

        content.addLayout(right_panel, 0, 1)
        content.setColumnStretch(0, 3)
        content.setColumnStretch(1, 2)

        layout.addLayout(content, stretch=1)

        # --- Bottom buttons ---
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("← Back to Region")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("scan_region"))
        btn_layout.addWidget(back_btn)

        btn_layout.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._results_btn = QPushButton("View Results →")
        self._results_btn.setMinimumHeight(40)
        self._results_btn.setEnabled(False)
        self._results_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("results_dashboard"))
        btn_layout.addWidget(self._results_btn)

        layout.addLayout(btn_layout)

    def on_shown(self) -> None:
        """Called when screen becomes visible."""
        self._status_label.setText("Ready to scan")
        self._detail_label.setText("Press Start to begin scanning the defined region")

        # Show field count from region
        region_count = getattr(self._ctx, "current_scan_region_vertices", [])
        if region_count:
            self._detail_label.setText(
                f"Region defined with {len(region_count)} vertices. Press Start to begin."
            )

    def _on_start(self) -> None:
        """Start or resume the scan."""
        if self._is_paused:
            self._is_paused = False
            self._status_label.setText("Scanning...")
            self._pause_btn.setText("⏸  Pause")
            logger.info("Scan resumed")
        else:
            self._is_scanning = True
            self._status_label.setText("Scanning...")
            self._status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1D9E75;")
            self._detail_label.setText("Scan in progress — do not disturb the microscope")
            logger.info("Scan started")

            # TODO: Emit scan_start signal to worker thread
            # self._scan_signals.scan_start_requested.emit(scan_region)

        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._results_btn.setEnabled(False)

    def _on_pause(self) -> None:
        """Pause the scan."""
        self._is_paused = True
        self._status_label.setText("Paused")
        self._status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #BA7517;")
        self._start_btn.setEnabled(True)
        self._start_btn.setText("▶  Resume")
        self._pause_btn.setText("⏸  Paused")
        self._pause_btn.setEnabled(False)
        logger.info("Scan paused")

    def _on_stop(self) -> None:
        """Stop the scan (requires confirmation)."""
        reply = QMessageBox.question(
            self,
            "Stop Scan",
            "Are you sure you want to stop the scan?\n\n"
            "Progress will be saved and can be reviewed,\n"
            "but the scan cannot be resumed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._is_scanning = False
            self._is_paused = False
            self._status_label.setText("Scan stopped")
            self._status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #E24B4A;")
            self._start_btn.setEnabled(True)
            self._start_btn.setText("▶  Start Scan")
            self._pause_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._results_btn.setEnabled(True)
            logger.info("Scan stopped by user")

    def update_progress(self, fields_done: int, fields_total: int, eta_sec: float) -> None:
        """
        Update progress display. Called by the scan worker via signal.

        Parameters
        ----------
        fields_done : int
        fields_total : int
        eta_sec : float
        """
        if fields_total > 0:
            pct = int(100 * fields_done / fields_total)
            self._progress_bar.setMaximum(fields_total)
            self._progress_bar.setValue(fields_done)
            self._progress_bar.setFormat(f"{fields_done} / {fields_total} fields ({pct}%)")

        self._fields_label.setText(f"Fields: {fields_done} / {fields_total}")

        if eta_sec < 60:
            self._eta_label.setText(f"ETA: {eta_sec:.0f}s")
        else:
            self._eta_label.setText(f"ETA: {eta_sec / 60:.1f} min")

    def on_scan_complete(self) -> None:
        """Called when scan finishes successfully."""
        self._is_scanning = False
        self._status_label.setText("Scan complete!")
        self._status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1D9E75;")
        self._detail_label.setText("All fields captured. View results or proceed to review.")
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._results_btn.setEnabled(True)
        self._progress_bar.setValue(self._progress_bar.maximum())
        logger.info("Scan complete — UI updated")
