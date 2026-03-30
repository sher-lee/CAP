"""
Screen 5: Review and Confirm
===============================
Technician reviews AI findings. Accept, reject, or modify
individual detections. Add notes and confirm final diagnosis.
Corrections are logged for the retraining pipeline.

On confirmation, triggers ReportWorker to generate PDF and
transfer to the exam room.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QComboBox, QSpacerItem, QSizePolicy, QMessageBox,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger
from cap.layer5_data import crud
from cap.layer5_data.audit import AuditLogger, EventType

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.review")


class ReviewConfirmScreen(QWidget):
    """Review AI detections and confirm results."""

    def __init__(self, app_context: AppContext, nav_signals: NavigationSignals) -> None:
        super().__init__()
        self._ctx = app_context
        self._nav = nav_signals
        self._detections: list[dict] = []
        self._report_worker = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(12)

        # Title
        title = QLabel("Review and Confirm")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Review AI detections. Modify any incorrect classifications before confirming.")
        subtitle.setStyleSheet("font-size: 12px; color: gray;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Detection table
        det_group = QGroupBox("Detections")
        det_layout = QVBoxLayout(det_group)

        self._det_table = QTableWidget()
        self._det_table.setColumnCount(6)
        self._det_table.setHorizontalHeaderLabels([
            "Field (X,Y)", "AI Class", "Confidence", "Corrected Class", "Action", "Status",
        ])
        self._det_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._det_table.setAlternatingRowColors(True)
        self._det_table.setMinimumHeight(300)
        det_layout.addWidget(self._det_table)

        # Bulk actions
        bulk_layout = QHBoxLayout()
        accept_all_btn = QPushButton("Accept All")
        accept_all_btn.clicked.connect(self._accept_all)
        bulk_layout.addWidget(accept_all_btn)

        bulk_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        det_count_label = QLabel("")
        self._det_count_label = det_count_label
        bulk_layout.addWidget(det_count_label)

        det_layout.addLayout(bulk_layout)
        layout.addWidget(det_group)

        # Notes section
        notes_group = QGroupBox("Technician Notes")
        notes_layout = QVBoxLayout(notes_group)

        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText(
            "Add any observations, notes for the veterinarian, "
            "or final diagnosis comments here..."
        )
        self._notes_edit.setMaximumHeight(100)
        notes_layout.addWidget(self._notes_edit)

        layout.addWidget(notes_group)

        # Report status label (hidden until report generation starts)
        self._report_status = QLabel("")
        self._report_status.setStyleSheet("font-size: 13px; color: #1D9E75; font-weight: bold;")
        self._report_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._report_status.setVisible(False)
        layout.addWidget(self._report_status)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("← Back to Results")
        back_btn.setMinimumHeight(40)
        back_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("results_dashboard"))
        btn_layout.addWidget(back_btn)

        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self._confirm_btn = QPushButton("Confirm and Generate Report")
        self._confirm_btn.setMinimumHeight(45)
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
        """Load detections when screen becomes visible."""
        slide_id = getattr(self._ctx, "current_slide_id", None)
        if slide_id:
            self._load_detections(slide_id)

    def _load_detections(self, slide_id: int) -> None:
        """Load detections for review."""
        self._detections = crud.get_detections_for_slide(self._ctx.db, slide_id)
        self._det_table.setRowCount(len(self._detections))

        # Available classes for correction dropdown
        classes = self._ctx.config.inference.classes

        for row, det in enumerate(self._detections):
            # Field position
            self._det_table.setItem(row, 0, QTableWidgetItem(
                f"({det.get('x', '?')}, {det.get('y', '?')})"
            ))

            # AI class
            self._det_table.setItem(row, 1, QTableWidgetItem(det["class"]))

            # Confidence
            conf_item = QTableWidgetItem(f"{det['confidence']:.2f}")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._det_table.setItem(row, 2, conf_item)

            # Correction dropdown
            combo = QComboBox()
            combo.addItem(det["class"])  # Default to AI class
            for cls in classes:
                if cls != det["class"]:
                    combo.addItem(cls)
            combo.addItem("false_positive")
            self._det_table.setCellWidget(row, 3, combo)

            # Accept button
            accept_btn = QPushButton("Accept")
            accept_btn.setStyleSheet("font-size: 11px;")
            accept_btn.clicked.connect(lambda checked, r=row: self._accept_row(r))
            self._det_table.setCellWidget(row, 4, accept_btn)

            # Status
            self._det_table.setItem(row, 5, QTableWidgetItem("Pending"))

        if self._detections:
            self._det_count_label.setText(f"{len(self._detections)} detections to review")
        else:
            self._det_count_label.setText("No detections — AI may have been disabled")

        logger.debug("Loaded %d detections for review", len(self._detections))

    def _accept_row(self, row: int) -> None:
        """Mark a single detection as accepted."""
        status_item = QTableWidgetItem("Accepted")
        status_item.setForeground(Qt.GlobalColor.darkGreen)
        self._det_table.setItem(row, 5, status_item)

    def _accept_all(self) -> None:
        """Mark all detections as accepted with their current AI class."""
        for row in range(self._det_table.rowCount()):
            self._accept_row(row)
        logger.info("All detections accepted")

    def _on_confirm(self) -> None:
        """Confirm all reviews and trigger report generation."""
        slide_id = getattr(self._ctx, "current_slide_id", None)
        tech_id = getattr(self._ctx, "current_tech_id", None)

        if not slide_id:
            return

        # Process corrections
        corrections_count = 0
        for row in range(self._det_table.rowCount()):
            combo = self._det_table.cellWidget(row, 3)
            if combo and row < len(self._detections):
                corrected_class = combo.currentText()
                original_class = self._detections[row]["class"]

                if corrected_class != original_class:
                    crud.insert_correction(
                        self._ctx.db,
                        detection_id=self._detections[row]["detection_id"],
                        tech_id=tech_id or 0,
                        original_class=original_class,
                        corrected_class=corrected_class,
                    )
                    corrections_count += 1

        # Update slide status
        crud.update_slide_status(self._ctx.db, slide_id, "complete")

        # Save notes
        notes = self._notes_edit.toPlainText().strip()
        if notes:
            self._ctx.db.execute(
                "UPDATE slides SET notes = ? WHERE slide_id = ?",
                (notes, slide_id),
            )

        # Audit log
        audit = AuditLogger(self._ctx.db)
        audit.log(
            EventType.RESULTS_CONFIRMED,
            user_id=tech_id,
            details=f"Slide {slide_id}: {corrections_count} corrections made",
        )

        logger.info(
            "Results confirmed for slide %d: %d corrections",
            slide_id, corrections_count,
        )

        # Disable confirm button while report generates
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.setText("Generating report...")

        # Start report generation
        self._start_report_worker(notes)

    def _start_report_worker(self, notes: str) -> None:
        """Create and start the report worker thread."""
        from cap.workers.report_worker import ReportWorker

        self._report_worker = ReportWorker(self._ctx, technician_notes=notes)

        self._report_worker.report_started.connect(
            lambda: self._show_report_status("Generating report...")
        )
        self._report_worker.report_progress.connect(self._show_report_status)
        self._report_worker.report_complete.connect(self._on_report_complete)
        self._report_worker.report_failed.connect(self._on_report_failed)

        self._report_worker.start()
        logger.info("ReportWorker thread started")

    def _show_report_status(self, message: str) -> None:
        """Show report generation progress."""
        self._report_status.setText(message)
        self._report_status.setVisible(True)

    def _on_report_complete(self, pdf_path: str) -> None:
        """Handle successful report generation."""
        self._report_status.setText(f"Report saved: {pdf_path}")
        self._confirm_btn.setText("Confirm and Generate Report")
        self._confirm_btn.setEnabled(True)

        logger.info("Report generated: %s", pdf_path)

        QMessageBox.information(
            self,
            "Report Generated",
            f"Results have been confirmed and the report has been generated.\n\n"
            f"Report: {pdf_path}\n\n"
            f"You can start a new scan or view patient history.",
        )

    def _on_report_failed(self, error_msg: str) -> None:
        """Handle report generation failure."""
        self._report_status.setText(f"Report failed: {error_msg}")
        self._report_status.setStyleSheet(
            "font-size: 13px; color: #E24B4A; font-weight: bold;"
        )
        self._confirm_btn.setText("Confirm and Generate Report")
        self._confirm_btn.setEnabled(True)

        logger.error("Report generation failed: %s", error_msg)

        QMessageBox.warning(
            self,
            "Report Failed",
            f"Results were confirmed but report generation failed:\n\n{error_msg}\n\n"
            f"You can try again or continue without a report.",
        )
