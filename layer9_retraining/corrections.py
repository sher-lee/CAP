"""
Correction Manager
====================
Aggregates technician corrections into a curated dataset for
model retraining. Identifies patterns in corrections (which
classes are most often wrong, which fields have disagreements)
and prepares batches for review before export.

Usage:
    mgr = CorrectionManager(db, config)
    pending = mgr.get_pending_review()
    stats = mgr.get_correction_stats()
    batch = mgr.prepare_retraining_batch(min_corrections=50)
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from cap.layer5_data.db_manager import DatabaseManager
from cap.layer5_data import crud
from cap.common.logging_setup import get_logger

logger = get_logger("retraining.corrections")


class CorrectionManager:
    """Manages the correction-to-retraining workflow."""

    def __init__(self, db: DatabaseManager, config: object = None) -> None:
        self._db = db
        self._config = config

    def get_pending_review(self) -> list[dict]:
        """
        Get all corrections that haven't been reviewed yet.
        These need Noah's review before being used for retraining.

        Returns
        -------
        list of dict
            Unreviewed corrections with detection and field context.
        """
        return crud.get_unreviewed_corrections(self._db)

    def get_correction_stats(self) -> dict:
        """
        Get summary statistics about corrections.

        Returns
        -------
        dict
            Keys: total, reviewed, pending, by_class (dict),
            by_technician (dict), false_positive_count,
            most_confused_pairs (list of tuples).
        """
        total = self._scalar("SELECT COUNT(*) FROM corrections")
        reviewed = self._scalar("SELECT COUNT(*) FROM corrections WHERE reviewed = 1")
        pending = total - reviewed

        false_positives = self._scalar(
            "SELECT COUNT(*) FROM corrections WHERE corrected_class = 'false_positive'"
        )

        # Corrections by original class
        by_class_rows = self._db.fetchall(
            """SELECT original_class, COUNT(*) as cnt
               FROM corrections GROUP BY original_class
               ORDER BY cnt DESC"""
        )
        by_class = {r["original_class"]: r["cnt"] for r in by_class_rows}

        # Corrections by technician
        by_tech_rows = self._db.fetchall(
            """SELECT t.name, COUNT(*) as cnt
               FROM corrections c
               JOIN technicians t ON c.tech_id = t.tech_id
               GROUP BY c.tech_id
               ORDER BY cnt DESC"""
        )
        by_technician = {r["name"]: r["cnt"] for r in by_tech_rows}

        # Most commonly confused class pairs
        pair_rows = self._db.fetchall(
            """SELECT original_class, corrected_class, COUNT(*) as cnt
               FROM corrections
               WHERE corrected_class != 'false_positive'
               GROUP BY original_class, corrected_class
               ORDER BY cnt DESC
               LIMIT 10"""
        )
        confused_pairs = [
            (r["original_class"], r["corrected_class"], r["cnt"])
            for r in pair_rows
        ]

        return {
            "total": total,
            "reviewed": reviewed,
            "pending": pending,
            "false_positive_count": false_positives,
            "by_class": by_class,
            "by_technician": by_technician,
            "most_confused_pairs": confused_pairs,
        }

    def prepare_retraining_batch(
        self,
        min_corrections: int = 50,
        only_reviewed: bool = True,
    ) -> Optional[dict]:
        """
        Prepare a batch of corrected annotations for retraining.

        Only proceeds if enough corrections have accumulated.
        Returns metadata about the batch (slide IDs, correction
        counts, affected classes) for the export step.

        Parameters
        ----------
        min_corrections : int
            Minimum number of corrections required to proceed.
        only_reviewed : bool
            If True, only include corrections that have been
            reviewed by Noah.

        Returns
        -------
        dict or None
            Batch metadata if enough corrections exist, else None.
            Keys: batch_id, correction_count, slide_ids,
            affected_classes, created_at.
        """
        review_filter = "AND reviewed = 1" if only_reviewed else ""

        count = self._scalar(
            f"SELECT COUNT(*) FROM corrections WHERE 1=1 {review_filter}"
        )

        if count < min_corrections:
            logger.info(
                "Not enough corrections for retraining batch: %d < %d",
                count, min_corrections,
            )
            return None

        # Get affected slides
        slide_rows = self._db.fetchall(
            f"""SELECT DISTINCT f.slide_id
                FROM corrections c
                JOIN detections d ON c.detection_id = d.detection_id
                JOIN fields f ON d.field_id = f.field_id
                WHERE 1=1 {review_filter}
                ORDER BY f.slide_id"""
        )
        slide_ids = [r["slide_id"] for r in slide_rows]

        # Get affected classes
        class_rows = self._db.fetchall(
            f"""SELECT DISTINCT original_class FROM corrections
                WHERE 1=1 {review_filter}
                UNION
                SELECT DISTINCT corrected_class FROM corrections
                WHERE corrected_class != 'false_positive'
                {review_filter.replace('AND', 'AND')}"""
        )
        affected_classes = [r["original_class"] for r in class_rows]

        batch = {
            "batch_id": datetime.now().strftime("batch_%Y%m%d_%H%M%S"),
            "correction_count": count,
            "slide_ids": slide_ids,
            "affected_classes": affected_classes,
            "only_reviewed": only_reviewed,
            "created_at": datetime.now().isoformat(),
        }

        logger.info(
            "Retraining batch prepared: %d corrections across %d slides",
            count, len(slide_ids),
        )

        return batch

    def get_corrected_annotations(self, slide_id: int) -> list[dict]:
        """
        Get the final corrected annotations for a slide, applying
        all technician corrections to the original AI detections.

        False positives are excluded. Class changes are applied.

        Parameters
        ----------
        slide_id : int

        Returns
        -------
        list of dict
            Corrected detections with final class labels.
        """
        # Get all detections for the slide
        detections = crud.get_detections_for_slide(self._db, slide_id)

        # Get corrections for those detections
        corrections = {}
        for det in detections:
            det_id = det["detection_id"]
            corr_row = self._db.fetchone(
                """SELECT corrected_class FROM corrections
                   WHERE detection_id = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (det_id,),
            )
            if corr_row:
                corrections[det_id] = corr_row["corrected_class"]

        # Apply corrections
        result = []
        for det in detections:
            det_id = det["detection_id"]
            if det_id in corrections:
                corrected_class = corrections[det_id]
                if corrected_class == "false_positive":
                    continue  # Skip false positives
                det["class"] = corrected_class
                det["was_corrected"] = True
            else:
                det["was_corrected"] = False
            result.append(det)

        return result

    def mark_batch_reviewed(
        self,
        reviewer_notes: str = None,
    ) -> int:
        """
        Mark all pending corrections as reviewed.

        Parameters
        ----------
        reviewer_notes : str, optional
            Notes from the reviewer.

        Returns
        -------
        int
            Number of corrections marked as reviewed.
        """
        pending = crud.get_unreviewed_corrections(self._db)

        for corr in pending:
            crud.mark_correction_reviewed(
                self._db,
                corr["correction_id"],
                reviewer_notes=reviewer_notes,
            )

        logger.info("Marked %d corrections as reviewed", len(pending))
        return len(pending)

    def _scalar(self, sql: str, params: tuple = ()) -> int:
        row = self._db.fetchone(sql, params)
        return row[0] if row else 0
