"""
PDF Report Generator
=====================
Generates the client-facing PDF report shown to pet owners in
the exam room. This is part of the primary product deliverable.

Layout (3 pages):
    Page 1: Header/branding, patient info, plain-English summary
    Page 2: Annotated stitched image (full width)
    Page 3: Organism count table, severity grades, vet notes section
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER

from cap.common.logging_setup import get_logger

logger = get_logger("visualization.pdf")


class PDFReportGenerator:
    """Generates client-facing PDF reports from scan results."""

    def __init__(self, config: object = None) -> None:
        self._styles = getSampleStyleSheet()
        self._styles.add(ParagraphStyle(name="ReportTitle", parent=self._styles["Title"], fontSize=24, spaceAfter=6, textColor=colors.HexColor("#1B3A5C")))
        self._styles.add(ParagraphStyle(name="ReportSubtitle", parent=self._styles["Normal"], fontSize=12, textColor=colors.gray, alignment=TA_CENTER, spaceAfter=20))
        self._styles.add(ParagraphStyle(name="SectionHead", parent=self._styles["Heading2"], fontSize=14, textColor=colors.HexColor("#2E5D8A"), spaceBefore=16, spaceAfter=8))
        self._styles.add(ParagraphStyle(name="Summary", parent=self._styles["Normal"], fontSize=13, leading=20, spaceBefore=10, spaceAfter=10))
        self._styles.add(ParagraphStyle(name="PatientInfo", parent=self._styles["Normal"], fontSize=12, leading=18))
        self._styles.add(ParagraphStyle(name="SmallNote", parent=self._styles["Normal"], fontSize=9, textColor=colors.gray))
        logger.debug("PDFReportGenerator initialized")

    def generate(self, output_path: str, patient_name: str, species: str, owner_name: str = "", scan_date: str = None, summary_text: str = "", organism_counts: dict[str, int] = None, severity_grades: dict[str, str] = None, overall_severity: str = "0", annotated_image_path: str = None, technician_name: str = "", notes: str = "", clinic_name: str = "CAP Cytology Analysis") -> str:
        """Generate the full PDF report. Returns path to the generated PDF."""
        if scan_date is None:
            scan_date = datetime.now().strftime("%B %d, %Y")
        if organism_counts is None:
            organism_counts = {}
        if severity_grades is None:
            severity_grades = {}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)

        story = []
        story.extend(self._build_page1(clinic_name, scan_date, patient_name, species, owner_name, technician_name, summary_text, overall_severity))
        story.append(PageBreak())
        story.extend(self._build_page2(annotated_image_path))
        story.append(PageBreak())
        story.extend(self._build_page3(organism_counts, severity_grades, overall_severity, notes))

        doc.build(story)
        size_kb = os.path.getsize(output_path) / 1024
        logger.info("PDF report generated: %s (%.0f KB)", output_path, size_kb)
        return output_path

    def _build_page1(self, clinic_name, scan_date, patient_name, species, owner_name, technician_name, summary_text, overall_severity):
        elements = []
        elements.append(Paragraph(clinic_name, self._styles["ReportTitle"]))
        elements.append(Paragraph(f"Ear Cytology Report &mdash; {scan_date}", self._styles["ReportSubtitle"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1B3A5C"), spaceAfter=16))
        elements.append(Paragraph("Patient Information", self._styles["SectionHead"]))

        info_data = [
            ["Pet Name:", patient_name, "Date:", scan_date],
            ["Species:", species.title(), "Technician:", technician_name or "\u2014"],
            ["Owner:", owner_name or "\u2014", "", ""],
        ]
        info_table = Table(info_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
        info_table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 11),
            ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 11),
            ("FONT", (1, 0), (1, -1), "Helvetica", 11),
            ("FONT", (3, 0), (3, -1), "Helvetica", 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Findings Summary", self._styles["SectionHead"]))
        elements.append(Paragraph(summary_text or "No organisms detected or AI analysis not available.", self._styles["Summary"]))
        if overall_severity and overall_severity != "0":
            elements.append(Paragraph(f"<b>Overall Severity: {overall_severity}</b>", self._styles["Summary"]))
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("See page 2 for the annotated slide image and page 3 for detailed counts.", self._styles["SmallNote"]))
        return elements

    def _build_page2(self, annotated_image_path):
        elements = []
        elements.append(Paragraph("Annotated Slide Image", self._styles["SectionHead"]))
        if annotated_image_path and os.path.isfile(annotated_image_path):
            max_w, max_h = 7.0 * inch, 8.5 * inch
            img = RLImage(annotated_image_path)
            iw, ih = img.drawWidth, img.drawHeight
            scale = min(max_w / iw, max_h / ih)
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale
            elements.append(img)
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Bounding boxes indicate detected organisms. Colors correspond to organism type (see page 3).", self._styles["SmallNote"]))
        else:
            elements.append(Spacer(1, 40))
            elements.append(Paragraph("[ Annotated slide image not available ]", ParagraphStyle(name="Placeholder", parent=self._styles["Normal"], fontSize=14, textColor=colors.gray, alignment=TA_CENTER)))
        return elements

    def _build_page3(self, organism_counts, severity_grades, overall_severity, notes):
        elements = []
        elements.append(Paragraph("Organism Counts", self._styles["SectionHead"]))
        if organism_counts:
            table_data = [["Organism", "Count", "Severity Grade"]]
            for cls in sorted(organism_counts.keys()):
                display = cls.replace("_", " ").title()
                table_data.append([display, str(organism_counts[cls]), severity_grades.get(cls, "0")])
            total = sum(organism_counts.values())
            table_data.append(["TOTAL", str(total), f"Overall: {overall_severity}"])

            table = Table(table_data, colWidths=[3.0*inch, 1.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B3A5C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 11),
                ("FONT", (0, 1), (-1, -2), "Helvetica", 11),
                ("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 11),
                ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F2F6FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No organisms detected.", self._styles["Summary"]))

        elements.append(Spacer(1, 30))
        elements.append(Paragraph("Detection Color Key", self._styles["SectionHead"]))
        legend = "<font color='#E8593C'>\u25a0</font> Cocci (small)&nbsp;&nbsp;&nbsp;<font color='#D85A30'>\u25a0</font> Cocci (large)&nbsp;&nbsp;&nbsp;<font color='#1D9E75'>\u25a0</font> Yeast&nbsp;&nbsp;&nbsp;<font color='#378ADD'>\u25a0</font> Rods&nbsp;&nbsp;&nbsp;<font color='#D4537E'>\u25a0</font> Ear mites&nbsp;&nbsp;&nbsp;<font color='#888780'>\u25a0</font> Artifact"
        elements.append(Paragraph(legend, self._styles["Normal"]))
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("Veterinarian Notes", self._styles["SectionHead"]))
        if notes:
            elements.append(Paragraph(notes, self._styles["Normal"]))
            elements.append(Spacer(1, 20))
        elements.append(Paragraph("Treatment Plan:", self._styles["PatientInfo"]))
        for _ in range(6):
            elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDDDDD"), spaceBefore=18, spaceAfter=0))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Veterinarian Signature: ________________________________  Date: ________________", self._styles["SmallNote"]))
        return elements
