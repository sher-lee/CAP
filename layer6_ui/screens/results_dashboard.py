"""
Screen 4: Results Dashboard
===============================
Displays slide summary with organism counts, severity grades,
and plain-English summary. Shows field thumbnails and a slide map.

Receives inference results via InferenceSignals or loads from DB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger
from cap.layer5_data import crud

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.results")


class ResultsDashboardScreen(QWidget):
    """Results dashboard showing scan analysis summary."""

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
        title = QLabel("Results Dashboard")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # AI status banner (hidden by default)
        self._ai_banner = QLabel("")
        self._ai_banner.setWordWrap(True)
        self._ai_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ai_banner.setVisible(False)
        layout.addWidget(self._ai_banner)

        # Summary section
        summary_group = QGroupBox("Slide Summary")
        summary_layout = QVBoxLayout(summary_group)

        self._summary_label = QLabel("No results available yet — complete a scan first.")
        self._summary_label.setStyleSheet("font-size: 16px; padding: 10px;")
        self._summary_label.setWordWrap(True)
        summary_layout.addWidget(self._summary_label)

        self._severity_label = QLabel("")
        self._severity_label.setStyleSheet("font-size: 14px; color: gray;")
        summary_layout.addWidget(self._severity_label)

        layout.addWidget(summary_group)

        # Organism counts table
        counts_group = QGroupBox("Organism Counts")
        counts_layout = QVBoxLayout(counts_group)

        self._counts_table = QTableWidget()
        self._counts_table.setColumnCount(3)
        self._counts_table.setHorizontalHeaderLabels(["Organism", "Count", "Severity"])
        self._counts_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._counts_table.setAlternatingRowColors(True)
        self._counts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._counts_table.setMinimumHeight(200)
        counts_layout.addWidget(self._counts_table)

        layout.addWidget(counts_group)

        # Slide map and field thumbnails placeholder
        visual_layout = QHBoxLayout()

        map_group = QGroupBox("Slide Map")
        map_layout = QVBoxLayout(map_group)
        self._map_placeholder = QLabel("[ Interactive slide map ]\nClick a region to view that field")
        self._map_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._map_placeholder.setMinimumHeight(150)
        self._map_placeholder.setStyleSheet("background-color: #f5f5f0; border: 1px solid #ccc; border-radius: 4px; color: #888;")
        map_layout.addWidget(self._map_placeholder)
        visual_layout.addWidget(map_group)

        thumb_group = QGroupBox("Flagged Fields")
        thumb_layout = QVBoxLayout(thumb_group)
        self._thumb_placeholder = QLabel("[ Field thumbnails with detections ]\nFlagged fields shown first")
        self._thumb_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_placeholder.setMinimumHeight(150)
        self._thumb_placeholder.setStyleSheet("background-color: #f5f5f0; border: 1px solid #ccc; border-radius: 4px; color: #888;")
        thumb_layout.addWidget(self._thumb_placeholder)
        visual_layout.addWidget(thumb_group)

        layout.addLayout(visual_layout)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("← Back to Scan")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("scan_control"))
        btn_layout.addWidget(back_btn)

        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        review_btn = QPushButton("Review and Confirm →")
        review_btn.setMinimumHeight(45)
        review_btn.setStyleSheet(
            "QPushButton { background-color: #1D9E75; color: white; font-size: 16px; "
            "font-weight: bold; border-radius: 6px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #0F6E56; }"
        )
        review_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("review_confirm"))
        btn_layout.addWidget(review_btn)

        layout.addLayout(btn_layout)

    def on_shown(self) -> None:
        """Refresh results when screen becomes visible."""
        slide_id = getattr(self._ctx, "current_slide_id", None)
        if slide_id:
            self._load_results(slide_id)

    # -------------------------------------------------------------------
    # Signal handlers (connected by scan_control after inference)
    # -------------------------------------------------------------------

    def on_inference_complete(self, slide_id: int, slide_results) -> None:
        """
        Called when InferenceWorker finishes.
        Refreshes the display with live results from the SlideResults object.
        """
        self._ai_banner.setVisible(False)
        self._display_slide_results(slide_results)
        logger.info("Results dashboard updated from live inference: slide %d", slide_id)

    def on_inference_skipped(self, reason: str) -> None:
        """
        Called when inference is skipped (AI-disabled mode).
        Shows a banner and displays image-only results.
        """
        self._ai_banner.setText(
            f"AI analysis was not available ({reason}). "
            f"Images were captured successfully and can be reviewed manually."
        )
        self._ai_banner.setStyleSheet(
            "background-color: #FFF3CD; color: #856404; padding: 10px; "
            "border: 1px solid #FFEEBA; border-radius: 4px; font-size: 13px;"
        )
        self._ai_banner.setVisible(True)

        self._summary_label.setText("No AI analysis available — manual review recommended.")
        self._severity_label.setText("")
        self._counts_table.setRowCount(0)
        logger.info("Results dashboard: inference skipped — %s", reason)

    # -------------------------------------------------------------------
    # Data display
    # -------------------------------------------------------------------

    def _display_slide_results(self, slide_results) -> None:
        """Display a SlideResults object directly (from live inference)."""
        # Summary
        if slide_results.plain_english_summary is not None:
            self._summary_label.setText(slide_results.plain_english_summary)
        else:
            self._summary_label.setText("Scan complete — no organisms detected.")

        # Overall severity
        if slide_results.overall_severity is not None:
            severity_val = slide_results.overall_severity.value
            self._severity_label.setText(f"Overall severity: {severity_val}")
        else:
            self._severity_label.setText("Overall severity: N/A")

        # Organism counts table
        counts = slide_results.organism_counts or {}
        grades = {}
        if slide_results.severity_grades is not None:
            grades = {
                cls: grade.value
                for cls, grade in slide_results.severity_grades.items()
            }

        self._populate_counts_table(counts, grades)

        # Update flagged fields placeholder
        flagged = slide_results.flagged_field_ids or []
        if flagged:
            self._thumb_placeholder.setText(
                f"[ {len(flagged)} flagged field(s) ]\n"
                f"Field IDs: {', '.join(str(f) for f in flagged[:10])}"
            )

    def _load_results(self, slide_id: int) -> None:
        """Load and display results from the database."""
        results = crud.get_results(self._ctx.db, slide_id)

        if not results:
            # Check if we have live results in context
            live_results = getattr(self._ctx, "last_slide_results", None)
            if live_results and live_results.slide_id == slide_id:
                self._display_slide_results(live_results)
                return

            self._summary_label.setText(
                "No AI results available. The scan may still be processing, "
                "or AI inference was skipped."
            )
            self._severity_label.setText("")
            self._counts_table.setRowCount(0)
            return

        # Summary text
        summary = results.get("plain_english_summary")
        if summary:
            self._summary_label.setText(summary)
        else:
            self._summary_label.setText("Scan complete — see table below for details.")

        severity = results.get("severity_score", "0")
        self._severity_label.setText(f"Overall severity: {severity}")

        # Organism counts table
        counts = results.get("organism_counts", {})
        grades = results.get("severity_grades", {})
        self._populate_counts_table(counts, grades)

        logger.debug("Results loaded for slide %d: %d organism classes", slide_id, len(counts))

    def _populate_counts_table(
        self,
        counts: dict[str, int],
        grades: dict[str, str],
    ) -> None:
        """Fill the organism counts table."""
        self._counts_table.setRowCount(len(counts))
        for row, (organism, count) in enumerate(sorted(counts.items())):
            self._counts_table.setItem(row, 0, QTableWidgetItem(organism.replace("_", " ").title()))
            self._counts_table.setItem(row, 1, QTableWidgetItem(str(count)))
            grade = grades.get(organism, "0")
            grade_item = QTableWidgetItem(grade)
            grade_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._counts_table.setItem(row, 2, grade_item)
