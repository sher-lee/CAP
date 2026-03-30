"""
Clinic Dashboard Metrics
=========================
Queries the database for clinic-level statistics: scan volume
over time, severity distribution, organism frequency, and
per-technician productivity.

All methods return plain dicts/lists so they can be rendered
by the UI (Layer 6) or exported to PDF/Excel (Layer 8 export).

Usage:
    dashboard = ClinicDashboard(db)
    stats = dashboard.get_overview()
    trend = dashboard.get_severity_trend(days=30)
"""

from __future__ import annotations

from typing import Optional

from cap.layer5_data.db_manager import DatabaseManager
from cap.common.logging_setup import get_logger

logger = get_logger("metrics.clinic")


class ClinicDashboard:
    """Clinic-level metrics computed from the database."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def get_overview(self) -> dict:
        """
        Get a high-level overview of all clinic activity.

        Returns
        -------
        dict
            Keys: total_scans, total_patients, total_detections,
            scans_with_findings, scans_clean, ai_disabled_scans,
            total_corrections.
        """
        total_scans = self._scalar(
            "SELECT COUNT(*) FROM slides WHERE status = 'complete'"
        )
        total_patients = self._scalar("SELECT COUNT(*) FROM patients")
        total_detections = self._scalar("SELECT COUNT(*) FROM detections")
        total_corrections = self._scalar("SELECT COUNT(*) FROM corrections")

        scans_with_findings = self._scalar(
            """SELECT COUNT(*) FROM results
               WHERE severity_score != '0' AND severity_score IS NOT NULL"""
        )
        scans_clean = self._scalar(
            "SELECT COUNT(*) FROM results WHERE severity_score = '0'"
        )
        ai_disabled_scans = self._scalar(
            "SELECT COUNT(*) FROM results WHERE model_version = 'none'"
        )

        return {
            "total_scans": total_scans,
            "total_patients": total_patients,
            "total_detections": total_detections,
            "scans_with_findings": scans_with_findings,
            "scans_clean": scans_clean,
            "ai_disabled_scans": ai_disabled_scans,
            "total_corrections": total_corrections,
        }

    def get_severity_distribution(self) -> dict[str, int]:
        """
        Get the distribution of overall severity scores across all slides.

        Returns
        -------
        dict
            {severity_score: count}, e.g. {'0': 15, '1+': 8, '2+': 3, ...}
        """
        rows = self._db.fetchall(
            """SELECT severity_score, COUNT(*) as cnt
               FROM results
               GROUP BY severity_score
               ORDER BY severity_score"""
        )
        return {r["severity_score"]: r["cnt"] for r in rows}

    def get_organism_frequency(self) -> dict[str, int]:
        """
        Get total detection counts across all slides, by organism class.

        Returns
        -------
        dict
            {class_name: total_count}
        """
        rows = self._db.fetchall(
            """SELECT class, COUNT(*) as cnt
               FROM detections
               GROUP BY class
               ORDER BY cnt DESC"""
        )
        return {r["class"]: r["cnt"] for r in rows}

    def get_severity_trend(self, days: int = 30) -> list[dict]:
        """
        Get daily severity trend over the last N days.

        Returns
        -------
        list of dict
            Each dict: {date, total_scans, avg_severity_numeric,
            severity_counts: {score: count}}
        """
        rows = self._db.fetchall(
            """SELECT DATE(s.date) as scan_date,
                      r.severity_score,
                      COUNT(*) as cnt
               FROM slides s
               JOIN results r ON s.slide_id = r.slide_id
               WHERE s.date >= DATE('now', ?)
               GROUP BY scan_date, r.severity_score
               ORDER BY scan_date""",
            (f"-{days} days",),
        )

        # Group by date
        by_date: dict[str, dict] = {}
        severity_numeric = {"0": 0, "1+": 1, "2+": 2, "3+": 3, "4+": 4}

        for r in rows:
            date = r["scan_date"]
            if date not in by_date:
                by_date[date] = {
                    "date": date,
                    "total_scans": 0,
                    "severity_sum": 0,
                    "severity_counts": {},
                }
            entry = by_date[date]
            score = r["severity_score"] or "0"
            count = r["cnt"]
            entry["total_scans"] += count
            entry["severity_sum"] += severity_numeric.get(score, 0) * count
            entry["severity_counts"][score] = count

        # Compute averages
        result = []
        for entry in by_date.values():
            total = entry["total_scans"]
            entry["avg_severity_numeric"] = (
                entry["severity_sum"] / total if total > 0 else 0
            )
            del entry["severity_sum"]
            result.append(entry)

        return result

    def get_scan_volume(self, days: int = 30) -> list[dict]:
        """
        Get daily scan count over the last N days.

        Returns
        -------
        list of dict
            Each dict: {date, count}
        """
        rows = self._db.fetchall(
            """SELECT DATE(date) as scan_date, COUNT(*) as cnt
               FROM slides
               WHERE status = 'complete'
               AND date >= DATE('now', ?)
               GROUP BY scan_date
               ORDER BY scan_date""",
            (f"-{days} days",),
        )
        return [{"date": r["scan_date"], "count": r["cnt"]} for r in rows]

    def get_technician_stats(self) -> list[dict]:
        """
        Get per-technician scan statistics.

        Returns
        -------
        list of dict
            Each dict: {tech_id, name, total_scans, total_corrections,
            avg_scan_duration_sec}
        """
        rows = self._db.fetchall(
            """SELECT t.tech_id, t.name,
                      COUNT(DISTINCT s.slide_id) as total_scans,
                      COALESCE(AVG(s.scan_duration), 0) as avg_duration
               FROM technicians t
               LEFT JOIN slides s ON s.technician_id = t.tech_id
                   AND s.status = 'complete'
               GROUP BY t.tech_id
               ORDER BY total_scans DESC"""
        )

        result = []
        for r in rows:
            corrections = self._scalar(
                """SELECT COUNT(*) FROM corrections
                   WHERE tech_id = ?""",
                (r["tech_id"],),
            )
            result.append({
                "tech_id": r["tech_id"],
                "name": r["name"],
                "total_scans": r["total_scans"],
                "total_corrections": corrections,
                "avg_scan_duration_sec": round(r["avg_duration"], 1),
            })

        return result

    def get_species_breakdown(self) -> dict[str, int]:
        """
        Get scan counts by species.

        Returns
        -------
        dict
            {species: count}
        """
        rows = self._db.fetchall(
            """SELECT p.species, COUNT(*) as cnt
               FROM slides s
               JOIN patients p ON s.patient_id = p.patient_id
               WHERE s.status = 'complete'
               GROUP BY p.species
               ORDER BY cnt DESC"""
        )
        return {r["species"]: r["cnt"] for r in rows}

    def _scalar(self, sql: str, params: tuple = ()) -> int:
        """Execute a query and return the first column of the first row."""
        row = self._db.fetchone(sql, params)
        return row[0] if row else 0
