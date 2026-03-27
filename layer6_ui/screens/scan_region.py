
"""
Screen 2: Scan Region Drawing
================================
Technician draws a polygon to define the scan area.
Shows estimated field count, scan time, and disk usage
before proceeding to the scan control panel.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger
from cap.layer6_ui.widgets.polygon_tool import PolygonDrawWidget
from cap.layer5_data import crud

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.scan_region")


class ScanRegionScreen(QWidget):
    """Scan region polygon drawing screen."""

    def __init__(self, app_context: AppContext, nav_signals: NavigationSignals) -> None:
        super().__init__()
        self._ctx = app_context
        self._nav = nav_signals
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(12)

        # Title
        title = QLabel("Define Scan Region")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        instructions = QLabel(
            "Click on the slide to draw the scan boundary around the oil-covered area. "
            "Double-click to close the shape. Right-click to undo."
        )
        instructions.setStyleSheet("font-size: 12px; color: gray;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # --- Main content: polygon tool + controls side by side ---
        content_layout = QHBoxLayout()

        # Polygon drawing widget (takes most of the space)
        slide_w = self._ctx.config.slide.width_mm
        slide_h = self._ctx.config.slide.height_mm
        self._polygon_widget = PolygonDrawWidget(slide_w, slide_h)
        self._polygon_widget.polygon_changed.connect(self._on_polygon_changed)
        self._polygon_widget.polygon_closed.connect(self._on_polygon_closed)
        content_layout.addWidget(self._polygon_widget, stretch=3)

        # Right panel: controls and estimates
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Presets
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QVBoxLayout(preset_group)

        preset_combo = QComboBox()
        preset_combo.addItem("— Select preset —", None)
        preset_combo.addItem("Full slide", "full_slide")
        preset_combo.addItem("Center half", "center_half")
        preset_combo.currentIndexChanged.connect(
            lambda: self._on_preset_selected(preset_combo.currentData())
        )
        preset_layout.addWidget(preset_combo)

        right_panel.addWidget(preset_group)

        # Edit controls
        edit_group = QGroupBox("Edit")
        edit_layout = QVBoxLayout(edit_group)

        undo_btn = QPushButton("Undo last point")
        undo_btn.clicked.connect(self._polygon_widget.undo_last_vertex)
        edit_layout.addWidget(undo_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._on_clear)
        edit_layout.addWidget(clear_btn)

        right_panel.addWidget(edit_group)

        # Estimates
        estimate_group = QGroupBox("Scan Estimates")
        estimate_layout = QVBoxLayout(estimate_group)

        self._field_count_label = QLabel("Fields: —")
        self._field_count_label.setStyleSheet("font-size: 14px;")
        estimate_layout.addWidget(self._field_count_label)

        self._scan_time_label = QLabel("Estimated time: —")
        self._scan_time_label.setStyleSheet("font-size: 14px;")
        estimate_layout.addWidget(self._scan_time_label)

        self._disk_label = QLabel("Estimated disk: —")
        self._disk_label.setStyleSheet("font-size: 14px;")
        estimate_layout.addWidget(self._disk_label)

        self._vertex_count_label = QLabel("Vertices: 0")
        self._vertex_count_label.setStyleSheet("font-size: 12px; color: gray;")
        estimate_layout.addWidget(self._vertex_count_label)

        self._status_label = QLabel("Draw a polygon to begin")
        self._status_label.setStyleSheet("font-size: 12px; color: #D85A30; font-weight: bold;")
        self._status_label.setWordWrap(True)
        estimate_layout.addWidget(self._status_label)

        right_panel.addWidget(estimate_group)

        right_panel.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        content_layout.addLayout(right_panel, stretch=1)
        layout.addLayout(content_layout, stretch=1)

        # --- Bottom buttons ---
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("← Back to Session")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("session_start"))
        btn_layout.addWidget(back_btn)

        btn_layout.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._confirm_btn = QPushButton("Confirm Region → Start Scan")
        self._confirm_btn.setMinimumHeight(45)
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.setStyleSheet(
            "QPushButton { background-color: #1D9E75; color: white; font-size: 16px; "
            "font-weight: bold; border-radius: 6px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #0F6E56; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self._confirm_btn.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self._confirm_btn)

        layout.addLayout(btn_layout)

    def on_shown(self) -> None:
        """Called when this screen becomes visible."""
        self._polygon_widget.clear()
        self._update_estimates([])
        self._status_label.setText("Draw a polygon to begin")
        self._confirm_btn.setEnabled(False)

    def _on_preset_selected(self, preset: str) -> None:
        """Apply a preset polygon shape."""
        if preset:
            self._polygon_widget.set_preset(preset)

    def _on_clear(self) -> None:
        """Clear polygon and reset estimates."""
        self._polygon_widget.clear()
        self._update_estimates([])
        self._confirm_btn.setEnabled(False)
        self._status_label.setText("Draw a polygon to begin")

    def _on_polygon_changed(self, vertices: list) -> None:
        """Called whenever the polygon vertices change."""
        self._vertex_count_label.setText(f"Vertices: {len(vertices)}")

        if self._polygon_widget.is_closed:
            self._update_estimates(vertices)

    def _on_polygon_closed(self, vertices: list) -> None:
        """Called when the polygon is finalized (double-click)."""
        self._update_estimates(vertices)
        self._confirm_btn.setEnabled(True)
        self._status_label.setText("Region defined — ready to scan")
        self._status_label.setStyleSheet("font-size: 12px; color: #1D9E75; font-weight: bold;")

    def _update_estimates(self, vertices: list) -> None:
        """Calculate and display scan estimates from the polygon."""
        if len(vertices) < 3:
            self._field_count_label.setText("Fields: —")
            self._scan_time_label.setText("Estimated time: —")
            self._disk_label.setText("Estimated disk: —")
            return

        # Estimate field count from polygon area
        area_fraction = self._polygon_area_fraction(vertices)
        slide_w = self._ctx.config.slide.width_mm
        slide_h = self._ctx.config.slide.height_mm
        fov_w = self._ctx.config.camera.fov_width_mm
        fov_h = self._ctx.config.camera.fov_height_mm

        total_fields_full_slide = int((slide_w / fov_w) * (slide_h / fov_h))
        estimated_fields = max(1, int(total_fields_full_slide * area_fraction))

        # Scan time
        fps = self._ctx.config.scan.fields_per_second
        scan_seconds = estimated_fields / fps
        if scan_seconds < 60:
            time_str = f"{scan_seconds:.0f} seconds"
        else:
            time_str = f"{scan_seconds / 60:.1f} minutes"

        # Disk usage (stacked JPEG ~4MB per field)
        disk_mb = estimated_fields * 4
        if disk_mb < 1024:
            disk_str = f"{disk_mb:.0f} MB"
        else:
            disk_str = f"{disk_mb / 1024:.1f} GB"

        self._field_count_label.setText(f"Fields: ~{estimated_fields}")
        self._scan_time_label.setText(f"Estimated time: ~{time_str}")
        self._disk_label.setText(f"Estimated disk: ~{disk_str}")

    def _on_confirm(self) -> None:
        """Lock the region and proceed to scan control."""
        vertices = self._polygon_widget.get_polygon_fractional()
        if len(vertices) < 3:
            return

        # Convert fractional coordinates to motor coordinates
        slide_w = self._ctx.config.slide.width_mm
        slide_h = self._ctx.config.slide.height_mm
        x_steps_mm = self._ctx.config.motor.x_steps_per_mm
        y_steps_mm = self._ctx.config.motor.y_steps_per_mm

        motor_vertices = [
            (int(x * slide_w * x_steps_mm), int(y * slide_h * y_steps_mm))
            for x, y in vertices
        ]

        # Save to slide record
        slide_id = getattr(self._ctx, "current_slide_id", None)
        if slide_id:
            region_json = json.dumps(motor_vertices)
            self._ctx.db.execute(
                """UPDATE slides SET scan_region_json = ?, scan_region_field_count = ?
                   WHERE slide_id = ?""",
                (region_json, len(vertices), slide_id),
            )

        # Store region in context for the scan control screen
        self._ctx.current_scan_region_vertices = motor_vertices
        self._ctx.current_scan_region_fractional = vertices

        logger.info(
            "Scan region confirmed: %d vertices, slide=%s",
            len(vertices), slide_id,
        )

        self._nav.go_to_screen.emit("scan_control")

    @staticmethod
    def _polygon_area_fraction(vertices: list[tuple[float, float]]) -> float:
        """
        Calculate polygon area as a fraction of the unit square (0-1 range).
        Uses the shoelace formula.
        """
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0
