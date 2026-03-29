"""
Report Worker
==============
QThread that generates the final deliverables after the technician
confirms results: stitched composite → annotated image → PDF report
→ exam room transfer.

Usage:
    worker = ReportWorker(app_context, technician_notes="...")
    worker.report_complete.connect(self.on_report_done)
    worker.start()
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

from cap.common.logging_setup import get_logger
from cap.layer5_data import crud
from cap.layer5_data.audit import AuditLogger, EventType

if TYPE_CHECKING:
    from cap.app import AppContext

logger = get_logger("workers.report")


class ReportWorker(QThread):
    """
    Generates the PDF report and transfers it to the exam room.

    Lifecycle:
        1. Stitch field composites into a full slide image
        2. Render detection annotations on the composite
        3. Generate PDF report with patient info + findings
        4. Transfer PDF to exam room (network share or local)
        5. Emit report_complete
    """

    # Signals
    report_started = Signal()
    report_progress = Signal(str)               # status message
    report_complete = Signal(str)               # pdf_path
    report_failed = Signal(str)                 # error message

    def __init__(
        self,
        app_context: AppContext,
        technician_notes: str = "",
    ) -> None:
        super().__init__()
        self._ctx = app_context
        self._notes = technician_notes

    def run(self) -> None:
        """Thread entry point."""
        slide_id = getattr(self._ctx, "current_slide_id", None)
        config = self._ctx.config
        audit = AuditLogger(self._ctx.db)

        try:
            self.report_started.emit()
            self.report_progress.emit("Preparing report...")

            # Get data from DB
            slide = crud.get_slide(self._ctx.db, slide_id) if slide_id else None
            results = crud.get_results(self._ctx.db, slide_id) if slide_id else None
            patient = None
            technician = None

            if slide:
                if slide.get("patient_id"):
                    patient = crud.get_patient(self._ctx.db, slide["patient_id"])
                if slide.get("technician_id"):
                    technician = crud.get_technician(self._ctx.db, slide["technician_id"])

            patient_name = patient["name"] if patient else "Unknown"
            species = patient["species"] if patient else "unknown"
            owner_name = patient.get("owner_name", "") if patient else ""
            tech_name = technician["name"] if technician else ""

            # ---------------------------------------------------------------
            # Step 1: Stitch composite (if stacked fields available)
            # ---------------------------------------------------------------
            self.report_progress.emit("Stitching slide composite...")

            annotated_image_path = None
            pipeline_result = getattr(self._ctx, "last_pipeline_result", None)

            if pipeline_result and pipeline_result.get("stacked_fields"):
                import numpy as np
                from cap.layer7_visualization.stitcher import Stitcher

                stacked_fields = pipeline_result["stacked_fields"]
                stitcher = Stitcher(config)

                field_images = [sf.composite for sf in stacked_fields]
                field_positions = [
                    (sf.field_x, sf.field_y) for sf in stacked_fields
                ]

                composite = stitcher.stitch(field_images, field_positions)

                # -----------------------------------------------------------
                # Step 2: Annotate composite with detections
                # -----------------------------------------------------------
                self.report_progress.emit("Rendering annotations...")

                detections = []
                if slide_id:
                    detections = crud.get_detections_for_slide(self._ctx.db, slide_id)

                if detections:
                    from cap.layer7_visualization.annotations import AnnotationRenderer

                    renderer = AnnotationRenderer(config)
                    composite = renderer.annotate_image(composite, detections)

                # Save annotated composite to disk
                import cv2

                slide_dir = pipeline_result.get(
                    "slide_dir",
                    os.path.join(config.storage.image_root, str(slide_id or 0)),
                )
                os.makedirs(slide_dir, exist_ok=True)
                annotated_image_path = os.path.join(slide_dir, "annotated_composite.jpg")
                cv2.imwrite(
                    annotated_image_path,
                    composite,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

                logger.info("Annotated composite saved: %s", annotated_image_path)

            # ---------------------------------------------------------------
            # Step 3: Generate PDF
            # ---------------------------------------------------------------
            self.report_progress.emit("Generating PDF report...")

            from cap.layer7_visualization.pdf_report import PDFReportGenerator

            pdf_gen = PDFReportGenerator(config)

            # Build output path
            report_dir = os.path.join(
                config.storage.image_root, str(slide_id or 0), "reports",
            )
            os.makedirs(report_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = patient_name.replace(" ", "_")
            pdf_filename = f"cytology_{safe_name}_{timestamp}.pdf"
            pdf_path = os.path.join(report_dir, pdf_filename)

            # Extract results data
            organism_counts = results.get("organism_counts", {}) if results else {}
            severity_grades = results.get("severity_grades", {}) if results else {}
            overall_severity = results.get("severity_score", "0") if results else "0"
            summary_text = results.get("plain_english_summary", "") if results else ""

            pdf_gen.generate(
                output_path=pdf_path,
                patient_name=patient_name,
                species=species,
                owner_name=owner_name,
                scan_date=slide.get("date") if slide else None,
                summary_text=summary_text or "",
                organism_counts=organism_counts,
                severity_grades=severity_grades,
                overall_severity=overall_severity,
                annotated_image_path=annotated_image_path,
                technician_name=tech_name,
                notes=self._notes,
            )

            logger.info("PDF report generated: %s", pdf_path)

            audit.log(
                EventType.REPORT_GENERATED,
                details=f"Slide {slide_id}: {pdf_filename}",
            )

            # ---------------------------------------------------------------
            # Step 4: Transfer to exam room
            # ---------------------------------------------------------------
            self.report_progress.emit("Transferring to exam room...")

            from cap.layer7_visualization.transfer import ExamRoomTransfer

            transfer = ExamRoomTransfer(config)
            transfer_path = transfer.transfer(pdf_path, pdf_filename)

            audit.log(
                EventType.REPORT_TRANSFERRED,
                details=f"Slide {slide_id}: → {transfer_path}",
            )

            logger.info("Report transferred: %s", transfer_path)

            self.report_progress.emit("Report complete!")
            self.report_complete.emit(pdf_path)

        except Exception as exc:
            logger.error("Report generation failed: %s", exc, exc_info=True)
            self.report_failed.emit(str(exc))
