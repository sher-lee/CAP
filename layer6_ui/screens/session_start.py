
"""
Screen 1: Session Start
=========================
Technician login (dropdown), patient information entry,
and slide preparation confirmation. Validates required fields
before allowing navigation to the scan region screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QGroupBox,
    QMessageBox, QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt

from cap.common.logging_setup import get_logger
from cap.layer5_data import crud

if TYPE_CHECKING:
    from cap.app import AppContext
    from cap.layer6_ui.signals import NavigationSignals

logger = get_logger("ui.session_start")


class SessionStartScreen(QWidget):
    """Technician login and patient information entry."""

    def __init__(self, app_context: AppContext, nav_signals: NavigationSignals) -> None:
        super().__init__()
        self._ctx = app_context
        self._nav = nav_signals
        self._session_id = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Title
        title = QLabel("New Scan Session")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Select technician and enter patient information to begin")
        subtitle.setStyleSheet("font-size: 14px; color: gray;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # --- Technician section ---
        tech_group = QGroupBox("Technician")
        tech_layout = QFormLayout(tech_group)

        self._tech_combo = QComboBox()
        self._tech_combo.setMinimumWidth(250)
        self._tech_combo.setPlaceholderText("Select technician...")
        tech_layout.addRow("Technician:", self._tech_combo)

        layout.addWidget(tech_group)

        # --- Patient section ---
        patient_group = QGroupBox("Patient Information")
        patient_layout = QFormLayout(patient_group)

        self._patient_name = QLineEdit()
        self._patient_name.setPlaceholderText("e.g. Buddy")
        patient_layout.addRow("Pet name *:", self._patient_name)

        self._species = QComboBox()
        self._species.addItems(["canine", "feline", "other"])
        patient_layout.addRow("Species *:", self._species)

        self._breed = QLineEdit()
        self._breed.setPlaceholderText("e.g. Golden Retriever (optional)")
        patient_layout.addRow("Breed:", self._breed)

        self._owner_name = QLineEdit()
        self._owner_name.setPlaceholderText("e.g. Smith")
        patient_layout.addRow("Owner name:", self._owner_name)

        self._owner_contact = QLineEdit()
        self._owner_contact.setPlaceholderText("Phone or email (optional)")
        patient_layout.addRow("Owner contact:", self._owner_contact)

        self._notes = QLineEdit()
        self._notes.setPlaceholderText("Additional notes (optional)")
        patient_layout.addRow("Notes:", self._notes)

        layout.addWidget(patient_group)

        # --- Existing patient search ---
        search_group = QGroupBox("Or Search Existing Patient")
        search_layout = QHBoxLayout(search_group)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search by pet name or owner name...")
        self._search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self._search_input)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._on_search)
        search_layout.addWidget(search_btn)

        self._search_results = QComboBox()
        self._search_results.setMinimumWidth(300)
        self._search_results.setPlaceholderText("Search results will appear here")
        self._search_results.currentIndexChanged.connect(self._on_patient_selected)
        search_layout.addWidget(self._search_results)

        layout.addWidget(search_group)

        # --- Buttons ---
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        btn_layout = QHBoxLayout()

        self._proceed_btn = QPushButton("Proceed to Scan Region →")
        self._proceed_btn.setMinimumHeight(45)
        self._proceed_btn.setStyleSheet(
            "QPushButton { background-color: #1D9E75; color: white; font-size: 16px; "
            "font-weight: bold; border-radius: 6px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #0F6E56; }"
        )
        self._proceed_btn.clicked.connect(self._on_proceed)
        btn_layout.addWidget(self._proceed_btn)

        layout.addLayout(btn_layout)

    def on_shown(self) -> None:
        """Called when this screen becomes visible. Refresh technician list."""
        self._load_technicians()
        self._patient_name.setFocus()

    def _load_technicians(self) -> None:
        """Load active technicians from the database into the dropdown."""
        self._tech_combo.clear()
        techs = crud.get_all_technicians(self._ctx.db)

        if not techs:
            # Create a default technician if none exist
            crud.insert_technician(self._ctx.db, name="Default Tech", login="default")
            techs = crud.get_all_technicians(self._ctx.db)

        for tech in techs:
            self._tech_combo.addItem(tech["name"], tech["tech_id"])

    def _on_search(self) -> None:
        """Search for existing patients."""
        query = self._search_input.text().strip()
        if not query:
            return

        results = crud.search_patients(self._ctx.db, query)
        self._search_results.clear()
        self._search_results.addItem("— Select patient —", None)

        for p in results:
            label = f"{p['name']} ({p['species']}) — Owner: {p.get('owner_name', 'N/A')}"
            self._search_results.addItem(label, p["patient_id"])

        if not results:
            self._search_results.addItem("No patients found", None)

    def _on_patient_selected(self, index: int) -> None:
        """Fill in patient fields from search selection."""
        patient_id = self._search_results.currentData()
        if patient_id is None:
            return

        patient = crud.get_patient(self._ctx.db, patient_id)
        if patient:
            self._patient_name.setText(patient["name"])
            species_idx = self._species.findText(patient["species"])
            if species_idx >= 0:
                self._species.setCurrentIndex(species_idx)
            self._breed.setText(patient.get("breed") or "")
            self._owner_name.setText(patient.get("owner_name") or "")
            self._owner_contact.setText(patient.get("owner_contact") or "")

    def _on_proceed(self) -> None:
        """Validate inputs and proceed to scan region screen."""
        # Validate technician
        tech_id = self._tech_combo.currentData()
        if tech_id is None:
            QMessageBox.warning(self, "Missing Information", "Please select a technician.")
            return

        # Validate patient name
        patient_name = self._patient_name.text().strip()
        if not patient_name:
            QMessageBox.warning(self, "Missing Information", "Please enter the pet name.")
            self._patient_name.setFocus()
            return

        # Start session
        self._session_id = crud.start_session(self._ctx.db, tech_id)

        # Insert or find patient
        species = self._species.currentText()
        breed = self._breed.text().strip() or None
        owner_name = self._owner_name.text().strip() or None
        owner_contact = self._owner_contact.text().strip() or None
        notes = self._notes.text().strip() or None

        # Check if patient was selected from search
        selected_patient_id = self._search_results.currentData()
        if selected_patient_id and self._patient_name.text().strip() == crud.get_patient(self._ctx.db, selected_patient_id).get("name"):
            patient_id = selected_patient_id
        else:
            patient_id = crud.insert_patient(
                self._ctx.db,
                name=patient_name,
                species=species,
                breed=breed,
                owner_name=owner_name,
                owner_contact=owner_contact,
                notes=notes,
            )

        # Create slide record
        slide_id = crud.insert_slide(
            self._ctx.db,
            patient_id=patient_id,
            session_id=self._session_id,
            technician_id=tech_id,
        )

        # Store in context for downstream screens
        self._ctx.current_session_id = self._session_id
        self._ctx.current_patient_id = patient_id
        self._ctx.current_slide_id = slide_id
        self._ctx.current_tech_id = tech_id

        logger.info(
            "Session started: tech=%d, patient=%d (%s), slide=%d",
            tech_id, patient_id, patient_name, slide_id,
        )

        self._nav.go_to_screen.emit("scan_region")
