"""
Screen 7: Settings
====================
Configuration editor showing all tunable parameters.
Changes are saved back to cap_config.yaml on save.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QLineEdit, QCheckBox, QScrollArea,
    QSpacerItem, QSizePolicy, QMessageBox,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger
from cap.config import save_config

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.settings")


class SettingsScreen(QWidget):
    """Settings editor for all configurable parameters."""

    def __init__(self, app_context: AppContext, nav_signals: NavigationSignals) -> None:
        super().__init__()
        self._ctx = app_context
        self._nav = nav_signals
        self._widgets: dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(20, 15, 20, 15)

        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(title)

        note = QLabel("Changes to motor and camera settings require a restart to take effect.")
        note.setStyleSheet("font-size: 11px; color: #BA7517;")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(note)

        # Scrollable area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(12)

        config = self._ctx.config

        # --- Scan settings ---
        scan_group = QGroupBox("Scan")
        scan_form = QFormLayout(scan_group)

        fps = self._add_spinbox(scan_form, "Fields per second:", config.scan.fields_per_second, 1, 5, "scan.fields_per_second")
        retries = self._add_spinbox(scan_form, "Max capture retries:", config.scan.max_capture_retries, 0, 10, "scan.max_capture_retries")
        self._add_checkbox(scan_form, "Serpentine pattern:", config.scan.serpentine_enabled, "scan.serpentine_enabled")

        layout.addWidget(scan_group)

        # --- Focus settings ---
        focus_group = QGroupBox("Focus Stacking")
        focus_form = QFormLayout(focus_group)

        self._add_spinbox(focus_form, "Z depths per field:", config.focus.z_depths_per_field, 2, 12, "focus.z_depths_per_field")
        self._add_spinbox(focus_form, "Block size (px):", config.focus.block_size, 8, 64, "focus.block_size")
        self._add_double_spinbox(focus_form, "Blend sigma:", config.focus.blend_sigma, 0.5, 20.0, "focus.blend_sigma")
        self._add_spinbox(focus_form, "Max registration shift (px):", config.focus.max_registration_shift, 10, 200, "focus.max_registration_shift")
        self._add_spinbox(focus_form, "Drift threshold (steps):", config.focus.drift_threshold, 10, 1000, "focus.drift_threshold")

        layout.addWidget(focus_group)

        # --- Motor settings ---
        motor_group = QGroupBox("Motor")
        motor_form = QFormLayout(motor_group)

        self._add_spinbox(motor_form, "Settle delay (ms):", config.motor.settle_delay_ms, 0, 1000, "motor.settle_delay_ms")
        self._add_spinbox(motor_form, "Speed (steps/sec):", config.motor.speed, 100, 10000, "motor.speed")
        self._add_spinbox(motor_form, "Microsteps:", config.motor.microsteps, 1, 256, "motor.microsteps")
        self._add_spinbox(motor_form, "Motor current (mA):", config.motor.motor_current_ma, 100, 2000, "motor.motor_current_ma")

        layout.addWidget(motor_group)

        # --- AI inference settings ---
        ai_group = QGroupBox("AI Inference")
        ai_form = QFormLayout(ai_group)

        self._add_checkbox(ai_form, "AI enabled:", config.inference.enabled, "inference.enabled")
        self._add_double_spinbox(ai_form, "Confidence threshold:", config.inference.confidence_threshold, 0.0, 1.0, "inference.confidence_threshold")
        self._add_double_spinbox(ai_form, "NMS IoU threshold:", config.inference.nms_iou_threshold, 0.0, 1.0, "inference.nms_iou_threshold")
        self._add_lineedit(ai_form, "Model path:", config.inference.model_path, "inference.model_path")

        layout.addWidget(ai_group)

        # --- Storage settings ---
        storage_group = QGroupBox("Storage")
        storage_form = QFormLayout(storage_group)

        self._add_spinbox(storage_form, "Raw retention (days):", config.storage.raw_retention_days, 1, 365, "storage.raw_retention_days")
        self._add_spinbox(storage_form, "JPEG quality:", config.storage.stacked_jpeg_quality, 50, 100, "storage.stacked_jpeg_quality")
        self._add_spinbox(storage_form, "Max disk usage (GB):", config.storage.max_disk_usage_gb, 10, 2000, "storage.max_disk_usage_gb")

        layout.addWidget(storage_group)

        # --- Visualization settings ---
        vis_group = QGroupBox("Visualization")
        vis_form = QFormLayout(vis_group)

        self._add_spinbox(vis_form, "Stitch overlap (px):", config.visualization.stitch_overlap_px, 10, 200, "visualization.stitch_overlap_px")
        self._add_spinbox(vis_form, "Tile size (px):", config.visualization.tile_size, 128, 512, "visualization.tile_size")

        layout.addWidget(vis_group)

        # --- Thermal settings ---
        thermal_group = QGroupBox("Thermal Monitoring")
        thermal_form = QFormLayout(thermal_group)

        self._add_spinbox(thermal_form, "Warning temp (\u00B0C):", config.thermal.warn_temp_c, 50, 100, "thermal.warn_temp_c")
        self._add_spinbox(thermal_form, "Pause temp (\u00B0C):", config.thermal.pause_temp_c, 60, 110, "thermal.pause_temp_c")

        layout.addWidget(thermal_group)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll, stretch=1)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("← Back")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("session_start"))
        btn_layout.addWidget(back_btn)

        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(40)
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        save_btn = QPushButton("Save Settings")
        save_btn.setMinimumHeight(45)
        save_btn.setStyleSheet(
            "QPushButton { background-color: #1D9E75; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 6px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #0F6E56; }"
        )
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        outer_layout.addLayout(btn_layout)

    # ----- Widget helpers -----

    def _add_spinbox(self, form: QFormLayout, label: str, value: int, min_val: int, max_val: int, key: str) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setValue(value)
        self._widgets[key] = widget
        form.addRow(label, widget)
        return widget

    def _add_double_spinbox(self, form: QFormLayout, label: str, value: float, min_val: float, max_val: float, key: str) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(min_val, max_val)
        widget.setDecimals(3)
        widget.setSingleStep(0.01)
        widget.setValue(value)
        self._widgets[key] = widget
        form.addRow(label, widget)
        return widget

    def _add_checkbox(self, form: QFormLayout, label: str, value: bool, key: str) -> QCheckBox:
        widget = QCheckBox()
        widget.setChecked(value)
        self._widgets[key] = widget
        form.addRow(label, widget)
        return widget

    def _add_lineedit(self, form: QFormLayout, label: str, value: str, key: str) -> QLineEdit:
        widget = QLineEdit(value)
        self._widgets[key] = widget
        form.addRow(label, widget)
        return widget

    # ----- Actions -----

    def _on_save(self) -> None:
        """Save current settings to config and write YAML."""
        config = self._ctx.config

        for key, widget in self._widgets.items():
            parts = key.split(".")
            section = getattr(config, parts[0])
            attr = parts[1]

            if isinstance(widget, QSpinBox):
                setattr(section, attr, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                setattr(section, attr, widget.value())
            elif isinstance(widget, QCheckBox):
                setattr(section, attr, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                setattr(section, attr, widget.text())

        save_config(config)
        logger.info("Settings saved to config file")

        QMessageBox.information(
            self, "Settings Saved",
            "Settings have been saved.\n\n"
            "Motor and camera changes will take effect after restart.",
        )

    def _on_reset(self) -> None:
        """Reset displayed values to current config values (discard unsaved changes)."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Discard unsaved changes and reset to current values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.on_shown()

    def on_shown(self) -> None:
        """Refresh displayed values from config."""
        config = self._ctx.config
        for key, widget in self._widgets.items():
            parts = key.split(".")
            section = getattr(config, parts[0], None)
            if section is None:
                continue
            value = getattr(section, parts[1], None)
            if value is None:
                continue

            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
