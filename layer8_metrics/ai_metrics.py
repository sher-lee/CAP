"""
AI Performance Metrics
=======================
Measures model accuracy by comparing AI predictions against
technician corrections. Tracks confidence distributions,
per-class precision, and correction rates to guide retraining.

Usage:
    metrics = AIMetrics(db)
    accuracy = metrics.get_accuracy_summary()
    confusion = metrics.get_confusion_matrix()
    confidence = metrics.get_confidence_analysis()
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from cap.layer5_data.db_manager import DatabaseManager
from cap.common.logging_setup import get_logger

logger = get_logger("metrics.ai")


class AIMetrics:
    """AI model performance metrics computed from corrections data."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def get_accuracy_summary(self) -> dict:
        """
        Compute overall model accuracy from correction data.

        Accuracy = detections not corrected / total reviewed detections.

        Returns
        -------
        dict
            Keys: total_detections, total_corrections,
            correction_rate, estimated_accuracy,
            false_positive_count, class_change_count.
        """
        total_detections = self._scalar("SELECT COUNT(*) FROM detections")
        total_corrections = self._scalar("SELECT COUNT(*) FROM corrections")

        false_positives = self._scalar(
            "SELECT COUNT(*) FROM corrections WHERE corrected_class = 'false_positive'"
        )
        class_changes = total_corrections - false_positives

        correction_rate = (
            total_corrections / total_detections
            if total_detections > 0
            else 0.0
        )
        estimated_accuracy = 1.0 - correction_rate

        return {
            "total_detections": total_detections,
            "total_corrections": total_corrections,
            "correction_rate": round(correction_rate, 4),
            "estimated_accuracy": round(estimated_accuracy, 4),
            "false_positive_count": false_positives,
            "class_change_count": class_changes,
        }

    def get_per_class_accuracy(self) -> list[dict]:
        """
        Compute accuracy per organism class.

        Returns
        -------
        list of dict
            Each dict: {class_name, total_detections, corrections,
            false_positives, accuracy}
        """
        # Total detections per class
        det_rows = self._db.fetchall(
            """SELECT class, COUNT(*) as cnt
               FROM detections GROUP BY class"""
        )
        det_counts = {r["class"]: r["cnt"] for r in det_rows}

        # Corrections per original class
        corr_rows = self._db.fetchall(
            """SELECT original_class, corrected_class, COUNT(*) as cnt
               FROM corrections
               GROUP BY original_class, corrected_class"""
        )

        class_corrections: dict[str, int] = Counter()
        class_fp: dict[str, int] = Counter()

        for r in corr_rows:
            orig = r["original_class"]
            corr = r["corrected_class"]
            count = r["cnt"]
            class_corrections[orig] += count
            if corr == "false_positive":
                class_fp[orig] += count

        result = []
        for cls in sorted(set(list(det_counts.keys()) + list(class_corrections.keys()))):
            total = det_counts.get(cls, 0)
            corrections = class_corrections.get(cls, 0)
            fps = class_fp.get(cls, 0)
            accuracy = (total - corrections) / total if total > 0 else 1.0

            result.append({
                "class_name": cls,
                "total_detections": total,
                "corrections": corrections,
                "false_positives": fps,
                "accuracy": round(accuracy, 4),
            })

        return result

    def get_confusion_matrix(self) -> dict:
        """
        Build a confusion matrix from corrections data.

        Returns
        -------
        dict
            Keys: matrix (dict of dicts), classes (sorted list),
            total_corrections.
            matrix[original][corrected] = count
        """
        rows = self._db.fetchall(
            """SELECT original_class, corrected_class, COUNT(*) as cnt
               FROM corrections
               GROUP BY original_class, corrected_class
               ORDER BY original_class, corrected_class"""
        )

        classes = set()
        matrix: dict[str, dict[str, int]] = defaultdict(Counter)

        for r in rows:
            orig = r["original_class"]
            corr = r["corrected_class"]
            count = r["cnt"]
            matrix[orig][corr] = count
            classes.add(orig)
            classes.add(corr)

        return {
            "matrix": dict(matrix),
            "classes": sorted(classes),
            "total_corrections": sum(r["cnt"] for r in rows),
        }

    def get_confidence_analysis(self) -> dict:
        """
        Analyze model confidence distribution and correlation
        with correction rates.

        Returns
        -------
        dict
            Keys: avg_confidence, confidence_buckets (list of dicts),
            corrected_avg_confidence, uncorrected_avg_confidence.
        """
        # Overall average
        avg_conf = self._scalar_float(
            "SELECT AVG(confidence) FROM detections"
        )

        # Average confidence of corrected vs uncorrected detections
        corrected_avg = self._scalar_float(
            """SELECT AVG(d.confidence) FROM detections d
               JOIN corrections c ON d.detection_id = c.detection_id"""
        )
        uncorrected_avg = self._scalar_float(
            """SELECT AVG(d.confidence) FROM detections d
               WHERE d.detection_id NOT IN (
                   SELECT detection_id FROM corrections
               )"""
        )

        # Confidence buckets: [0.5-0.6, 0.6-0.7, ..., 0.9-1.0]
        buckets = []
        for low in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high = low + 0.1
            total_in_bucket = self._scalar(
                "SELECT COUNT(*) FROM detections WHERE confidence >= ? AND confidence < ?",
                (low, high if high < 1.0 else 1.01),
            )
            corrected_in_bucket = self._scalar(
                """SELECT COUNT(*) FROM detections d
                   JOIN corrections c ON d.detection_id = c.detection_id
                   WHERE d.confidence >= ? AND d.confidence < ?""",
                (low, high if high < 1.0 else 1.01),
            )
            correction_rate = (
                corrected_in_bucket / total_in_bucket
                if total_in_bucket > 0
                else 0.0
            )
            buckets.append({
                "range": f"{low:.1f}-{high:.1f}",
                "total": total_in_bucket,
                "corrected": corrected_in_bucket,
                "correction_rate": round(correction_rate, 4),
            })

        return {
            "avg_confidence": round(avg_conf, 4) if avg_conf else 0.0,
            "corrected_avg_confidence": round(corrected_avg, 4) if corrected_avg else None,
            "uncorrected_avg_confidence": round(uncorrected_avg, 4) if uncorrected_avg else None,
            "confidence_buckets": buckets,
        }

    def get_model_version_comparison(self) -> list[dict]:
        """
        Compare accuracy across different model versions.

        Returns
        -------
        list of dict
            Each dict: {model_version, total_detections, corrections,
            accuracy, avg_confidence}
        """
        rows = self._db.fetchall(
            """SELECT d.model_version,
                      COUNT(*) as total,
                      AVG(d.confidence) as avg_conf
               FROM detections d
               GROUP BY d.model_version
               ORDER BY d.model_version"""
        )

        result = []
        for r in rows:
            version = r["model_version"] or "unknown"
            total = r["total"]
            corrections = self._scalar(
                """SELECT COUNT(*) FROM corrections c
                   JOIN detections d ON c.detection_id = d.detection_id
                   WHERE d.model_version = ?""",
                (version,),
            )
            accuracy = (total - corrections) / total if total > 0 else 1.0

            result.append({
                "model_version": version,
                "total_detections": total,
                "corrections": corrections,
                "accuracy": round(accuracy, 4),
                "avg_confidence": round(r["avg_conf"], 4) if r["avg_conf"] else 0.0,
            })

        return result

    def _scalar(self, sql: str, params: tuple = ()) -> int:
        row = self._db.fetchone(sql, params)
        return row[0] if row else 0

    def _scalar_float(self, sql: str, params: tuple = ()) -> Optional[float]:
        row = self._db.fetchone(sql, params)
        return float(row[0]) if row and row[0] is not None else None
