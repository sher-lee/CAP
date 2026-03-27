"""
CAP Common Package
==================
Shared utilities, dataclasses, and infrastructure used across all layers.
No layer-specific logic belongs here — only cross-cutting concerns.
"""

from cap.common.dataclasses import (
    Detection,
    FieldStatus,
    FocusMapResult,
    ProcessedFrame,
    RawFrame,
    ScanProgress,
    ScanRegion,
    SeverityGrade,
    SlideResults,
    SlideStatus,
    StackedField,
)
from cap.common.logging_setup import configure_logging, get_logger
from cap.common.backend import Backend, get_backend, HardwareMode

__all__ = [
    # Dataclasses
    "Detection",
    "FieldStatus",
    "FocusMapResult",
    "ProcessedFrame",
    "RawFrame",
    "ScanProgress",
    "ScanRegion",
    "SeverityGrade",
    "SlideResults",
    "SlideStatus",
    "StackedField",
    # Logging
    "configure_logging",
    "get_logger",
    # Backend
    "Backend",
    "get_backend",
    "HardwareMode",
]
