"""
Metrics Exporter
=================
Renders clinic dashboard and AI performance metrics into
exportable formats (summary dict, CSV, or PDF report).

The PDF export is optional and requires the PDF report
generator from Layer 7. Without it, metrics are still
available as plain dicts for the UI.

Usage:
    exporter = MetricsExporter(db, config)
    summary = exporter.generate_summary()
    exporter.export_csv(output_dir)
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Optional

from cap.layer5_data.db_manager import DatabaseManager
from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
from cap.layer8_metrics.ai_metrics import AIMetrics
from cap.common.logging_setup import get_logger

logger = get_logger("metrics.export")


class MetricsExporter:
    """Exports metrics in various formats."""

    def __init__(self, db: DatabaseManager, config: object = None) -> None:
        self._db = db
        self._config = config
        self._clinic = ClinicDashboard(db)
        self._ai = AIMetrics(db)

    def generate_summary(self) -> dict:
        """
        Generate a comprehensive metrics summary combining
        clinic and AI metrics.

        Returns
        -------
        dict
            Keys: generated_at, clinic (overview, severity_distribution,
            organism_frequency, species_breakdown, technician_stats),
            ai (accuracy_summary, per_class_accuracy, confidence_analysis,
            model_versions).
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "clinic": {
                "overview": self._clinic.get_overview(),
                "severity_distribution": self._clinic.get_severity_distribution(),
                "organism_frequency": self._clinic.get_organism_frequency(),
                "species_breakdown": self._clinic.get_species_breakdown(),
                "technician_stats": self._clinic.get_technician_stats(),
            },
            "ai": {
                "accuracy_summary": self._ai.get_accuracy_summary(),
                "per_class_accuracy": self._ai.get_per_class_accuracy(),
                "confidence_analysis": self._ai.get_confidence_analysis(),
                "model_versions": self._ai.get_model_version_comparison(),
            },
        }

    def export_csv(self, output_dir: str) -> list[str]:
        """
        Export metrics as a set of CSV files.

        Parameters
        ----------
        output_dir : str
            Directory to write CSV files.

        Returns
        -------
        list of str
            Paths to the generated CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        # 1. Clinic overview
        overview = self._clinic.get_overview()
        path = os.path.join(output_dir, "clinic_overview.csv")
        self._dict_to_csv(overview, path)
        paths.append(path)

        # 2. Severity distribution
        severity = self._clinic.get_severity_distribution()
        path = os.path.join(output_dir, "severity_distribution.csv")
        self._dict_to_csv(severity, path, key_header="severity", value_header="count")
        paths.append(path)

        # 3. Organism frequency
        organisms = self._clinic.get_organism_frequency()
        path = os.path.join(output_dir, "organism_frequency.csv")
        self._dict_to_csv(organisms, path, key_header="organism", value_header="count")
        paths.append(path)

        # 4. Technician stats
        tech_stats = self._clinic.get_technician_stats()
        path = os.path.join(output_dir, "technician_stats.csv")
        self._list_to_csv(tech_stats, path)
        paths.append(path)

        # 5. AI accuracy per class
        per_class = self._ai.get_per_class_accuracy()
        path = os.path.join(output_dir, "ai_accuracy_per_class.csv")
        self._list_to_csv(per_class, path)
        paths.append(path)

        # 6. Confidence analysis buckets
        conf = self._ai.get_confidence_analysis()
        path = os.path.join(output_dir, "confidence_buckets.csv")
        self._list_to_csv(conf.get("confidence_buckets", []), path)
        paths.append(path)

        # 7. Model version comparison
        versions = self._ai.get_model_version_comparison()
        path = os.path.join(output_dir, "model_versions.csv")
        self._list_to_csv(versions, path)
        paths.append(path)

        logger.info("Metrics exported: %d CSV files → %s", len(paths), output_dir)
        return paths

    def export_json(self, output_path: str) -> str:
        """
        Export the full metrics summary as a JSON file.

        Parameters
        ----------
        output_path : str
            Path to write the JSON file.

        Returns
        -------
        str
            The output path written.
        """
        summary = self.generate_summary()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Metrics JSON exported: %s", output_path)
        return output_path

    @staticmethod
    def _dict_to_csv(
        data: dict,
        path: str,
        key_header: str = "metric",
        value_header: str = "value",
    ) -> None:
        """Write a flat dict as a two-column CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([key_header, value_header])
            for key, value in data.items():
                writer.writerow([key, value])

    @staticmethod
    def _list_to_csv(data: list[dict], path: str) -> None:
        """Write a list of dicts as a CSV with headers from the first dict."""
        if not data:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
