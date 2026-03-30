"""
Layer 8: Metrics
=================
Clinic dashboards and AI performance metrics. Queries the database
for historical scan data and computes statistics for technician
review, clinic management, and model evaluation.

Modules:
    clinic_dashboard — scan volume, severity trends, technician stats
    ai_metrics       — model accuracy vs corrections, confidence analysis
    export           — render metrics to PDF or Excel reports
"""

from cap.layer8_metrics.clinic_dashboard import ClinicDashboard
from cap.layer8_metrics.ai_metrics import AIMetrics
from cap.layer8_metrics.export import MetricsExporter

__all__ = ["ClinicDashboard", "AIMetrics", "MetricsExporter"]
