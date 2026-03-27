
"""
CAP Audit Logger
=================
Writes user-facing events to the audit_log database table.
This is separate from the application logger (logging_setup.py) —
audit events are visible to clinic administrators and are part
of the compliance trail.

Usage:
    from cap.layer5_data.audit import AuditLogger
    audit = AuditLogger(db)
    audit.log("scan_started", tech_id=1, details="Slide 42, patient Buddy")
"""

from __future__ import annotations

from typing import Optional

from cap.layer5_data.db_manager import DatabaseManager
from cap.common.logging_setup import get_logger

logger = get_logger("data.audit")


class EventType:
    """Standard audit event types."""
    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Scan events
    SCAN_STARTED = "scan_started"
    SCAN_PAUSED = "scan_paused"
    SCAN_RESUMED = "scan_resumed"
    SCAN_COMPLETED = "scan_completed"
    SCAN_FAILED = "scan_failed"

    # Inference events
    INFERENCE_STARTED = "inference_started"
    INFERENCE_COMPLETED = "inference_completed"
    INFERENCE_SKIPPED = "inference_skipped"

    # Review events
    RESULTS_CONFIRMED = "results_confirmed"
    DETECTION_CORRECTED = "detection_corrected"

    # Report events
    REPORT_GENERATED = "report_generated"
    REPORT_TRANSFERRED = "report_transferred"

    # System events
    OIL_WARNING = "oil_warning"
    THERMAL_WARNING = "thermal_warning"
    EMERGENCY_STOP = "emergency_stop"
    FOCUS_DRIFT_WARNING = "focus_drift_warning"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"

    # Configuration events
    CONFIG_CHANGED = "config_changed"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_ROLLBACK = "model_rollback"

    # Data events
    BACKUP_CREATED = "backup_created"
    RAW_FRAMES_CLEANED = "raw_frames_cleaned"
    DATA_EXPORTED = "data_exported"


class AuditLogger:
    """
    Writes audit events to the database.
    Each event is timestamped automatically by SQLite.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def log(
        self,
        event_type: str,
        user_id: int = None,
        details: str = None,
    ) -> int:
        """
        Record an audit event.

        Parameters
        ----------
        event_type : str
            One of the EventType constants (or any string).
        user_id : int, optional
            Technician ID, or None for system events.
        details : str, optional
            Free-text description of the event.

        Returns
        -------
        int
            The log_id of the inserted record.
        """
        cursor = self._db.execute(
            """INSERT INTO audit_log (event_type, user_id, details)
               VALUES (?, ?, ?)""",
            (event_type, user_id, details),
        )
        log_id = cursor.lastrowid
        logger.debug("Audit: [%s] user=%s — %s", event_type, user_id, details or "")
        return log_id

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Get the most recent audit events."""
        rows = self._db.fetchall(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_by_event_type(self, event_type: str, limit: int = 100) -> list[dict]:
        """Get audit events filtered by type."""
        rows = self._db.fetchall(
            """SELECT * FROM audit_log
               WHERE event_type = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (event_type, limit),
        )
        return [dict(r) for r in rows]

    def get_by_user(self, user_id: int, limit: int = 100) -> list[dict]:
        """Get audit events for a specific technician."""
        rows = self._db.fetchall(
            """SELECT * FROM audit_log
               WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit),
        )
        return [dict(r) for r in rows]

    def get_between_dates(
        self,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Get audit events between two dates.

        Parameters
        ----------
        start_date : str
            ISO format date string, e.g. "2026-03-01".
        end_date : str
            ISO format date string, e.g. "2026-03-31".
        """
        rows = self._db.fetchall(
            """SELECT * FROM audit_log
               WHERE timestamp BETWEEN ? AND ?
               ORDER BY timestamp""",
            (start_date, end_date),
        )
        return [dict(r) for r in rows]
