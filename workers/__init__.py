"""
CAP Workers
============
QThread-based workers that bridge UI signals to backend pipelines.
Each worker runs a long-running operation off the UI thread and
communicates progress/results back via Qt signals.

Workers:
    ScanWorker       — scan pipeline (focus → capture → stack → disk)
    InferenceWorker  — AI inference + aggregation + DB storage
    ReportWorker     — stitching + annotation + PDF + exam room transfer
"""
