"""
Layer 9: Retraining Pipeline
==============================
Manages the correction-to-retraining workflow. Collects technician
corrections, exports annotated datasets in CVAT format, and provides
stubs for future automated retraining.

Modules:
    corrections    — correction aggregation and dataset curation
    cvat_export    — CVAT-compatible annotation export for labeling tools
"""

from cap.layer9_retraining.corrections import CorrectionManager
from cap.layer9_retraining.cvat_export import CVATExporter

__all__ = ["CorrectionManager", "CVATExporter"]
