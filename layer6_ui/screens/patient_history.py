
"""
Screen 6: Patient History
============================
Search patients, view prior scans, and compare results over time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger
from cap.layer5_data import crud

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.history")


class PatientHistoryScreen(QWidget):
    """Patient history and prior scan viewer."""

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
        title = QLabel("Patient History")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Search bar
        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search by pet name, owner name, or date...")
        self._search_input.setMinimumHeight(35)
        self._search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self._search_input)

        search_btn = QPushButton("Search")
        search_btn.setMinimumHeight(35)
        search_btn.clicked.connect(self._on_search)
        search_layout.addWidget(search_btn)

        show_all_btn = QPushButton("Show Recent")
        show_all_btn.setMinimumHeight(35)
        show_all_btn.clicked.connect(self._show_recent)
        search_layout.addWidget(show_all_btn)

        layout.addLayout(search_layout)

        # Patient table
        patient_group = QGroupBox("Patients")
        patient_layout = QVBoxLayout(patient_group)

        self._patient_table = QTableWidget()
        self._patient_table.setColumnCount(5)
        self._patient_table.setHorizontalHeaderLabels([
            "Pet Name", "Species", "Owner", "Scans", "Last Scan",
        ])
        self._patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._patient_table.setAlternatingRowColors(True)
        self._patient_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._patient_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._patient_table.doubleClicked.connect(self._on_patient_double_clicked)
        patient_layout.addWidget(self._patient_table)

        layout.addWidget(patient_group)

        # Scan history for selected patient
        scan_group = QGroupBox("Scan History")
        scan_layout = QVBoxLayout(scan_group)

        self._scan_table = QTableWidget()
        self._scan_table.setColumnCount(5)
        self._scan_table.setHorizontalHeaderLabels([
            "Date", "Status", "Duration", "Fields", "Severity",
        ])
        self._scan_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._scan_table.setAlternatingRowColors(True)
        self._scan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._scan_table.setMinimumHeight(200)
        scan_layout.addWidget(self._scan_table)

        self._selected_patient_label = QLabel("Select a patient above to view scan history")
        self._selected_patient_label.setStyleSheet("color: gray; font-style: italic;")
        scan_layout.addWidget(self._selected_patient_label)

        layout.addWidget(scan_group)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        new_scan_btn = QPushButton("← New Scan")
        new_scan_btn.setMinimumHeight(40)
        new_scan_btn.clicked.connect(lambda: self._nav.go_to_screen.emit("session_start"))
        btn_layout.addWidget(new_scan_btn)

        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        layout.addLayout(btn_layout)

    def on_shown(self) -> None:
        """Refresh data when screen becomes visible."""
        self._show_recent()

    def _on_search(self) -> None:
        """Search for patients."""
        query = self._search_input.text().strip()
        if not query:
            self._show_recent()
            return

        patients = crud.search_patients(self._ctx.db, query)
        self._populate_patient_table(patients)

    def _show_recent(self) -> None:
        """Show all patients with their scan counts."""
        patients = crud.get_all_patients(self._ctx.db)
        self._populate_patient_table(patients)

    def _populate_patient_table(self, patients: list[dict]) -> None:
        """Fill the patient table."""
        self._patient_table.setRowCount(len(patients))

        for row, p in enumerate(patients):
            self._patient_table.setItem(row, 0, QTableWidgetItem(p["name"]))
            self._patient_table.setItem(row, 1, QTableWidgetItem(p.get("species", "")))
            self._patient_table.setItem(row, 2, QTableWidgetItem(p.get("owner_name", "")))

            # Get scan count for this patient
            slides = crud.get_slides_for_patient(self._ctx.db, p["patient_id"])
            self._patient_table.setItem(row, 3, QTableWidgetItem(str(len(slides))))

            last_date = slides[0]["date"] if slides else "—"
            self._patient_table.setItem(row, 4, QTableWidgetItem(str(last_date)))

            # Store patient_id in the first column's data
            item = self._patient_table.item(row, 0)
            item.setData(Qt.ItemDataRole.UserRole, p["patient_id"])

    def _on_patient_double_clicked(self, index) -> None:
        """Show scan history for the selected patient."""
        row = index.row()
        item = self._patient_table.item(row, 0)
        if not item:
            return

        patient_id = item.data(Qt.ItemDataRole.UserRole)
        patient_name = item.text()

        self._selected_patient_label.setText(f"Scan history for: {patient_name}")
        self._selected_patient_label.setStyleSheet("color: black; font-weight: bold;")

        slides = crud.get_slides_for_patient(self._ctx.db, patient_id)
        self._scan_table.setRowCount(len(slides))

        for row, s in enumerate(slides):
            self._scan_table.setItem(row, 0, QTableWidgetItem(str(s["date"])))
            self._scan_table.setItem(row, 1, QTableWidgetItem(s["status"]))

            duration = s.get("scan_duration")
            dur_str = f"{duration:.1f}s" if duration else "—"
            self._scan_table.setItem(row, 2, QTableWidgetItem(dur_str))

            field_count = s.get("scan_region_field_count")
            self._scan_table.setItem(row, 3, QTableWidgetItem(str(field_count or "—")))

            # Get severity from results
            results = crud.get_results(self._ctx.db, s["slide_id"])
            severity = results.get("severity_score", "—") if results else "—"
            self._scan_table.setItem(row, 4, QTableWidgetItem(str(severity)))
