"""
CAP Backup and Data Retention
================================
Handles database backups, raw frame cleanup, and portable
archive exports.

Usage:
    from cap.layer5_data.backup import BackupManager
    bm = BackupManager(config, db)
    bm.create_backup()                  # Copy DB to backup location
    bm.cleanup_old_raw_frames()         # Delete raw frames older than retention period
    bm.create_portable_archive(path)    # Zip DB + stacked images
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import zipfile

from cap.layer5_data.db_manager import DatabaseManager
from cap.common.logging_setup import get_logger

logger = get_logger("data.backup")


class BackupManager:
    """Manages backups, data retention, and portable archives."""

    def __init__(self, config: object, db: DatabaseManager) -> None:
        if hasattr(config, "storage"):
            self._db_path = config.storage.db_path
            self._image_root = config.storage.image_root
            self._raw_retention_days = config.storage.raw_retention_days
            self._max_disk_gb = config.storage.max_disk_usage_gb
            self._backup_path = config.storage.backup_target_path
        else:
            storage = config.get("storage", {})
            self._db_path = storage.get("db_path", "./data/cap.db")
            self._image_root = storage.get("image_root", "./data/slides/")
            self._raw_retention_days = storage.get("raw_retention_days", 7)
            self._max_disk_gb = storage.get("max_disk_usage_gb", 400)
            self._backup_path = storage.get("backup_target_path")

        self._db = db
        logger.info(
            "BackupManager initialized: retention=%d days, backup=%s",
            self._raw_retention_days, self._backup_path or "(not configured)",
        )

    # ----- Database backup -----

    def create_backup(self, target_dir: str = None) -> Optional[str]:
        """
        Create a backup copy of the database.

        Parameters
        ----------
        target_dir : str, optional
            Directory to write the backup. Uses configured backup_target_path
            if not specified.

        Returns
        -------
        str or None
            Path to the backup file, or None if backup_target_path is not configured.
        """
        target = target_dir or self._backup_path
        if not target:
            logger.warning("No backup target configured — skipping backup")
            return None

        os.makedirs(target, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"cap_backup_{timestamp}.db"
        backup_path = os.path.join(target, backup_filename)

        # Use SQLite's backup API for a consistent copy
        import sqlite3
        source_conn = self._db.get_connection()
        dest_conn = sqlite3.connect(backup_path)
        source_conn.backup(dest_conn)
        dest_conn.close()

        logger.info("Database backup created: %s", backup_path)
        return backup_path

    # ----- Raw frame cleanup -----

    def cleanup_old_raw_frames(self) -> dict:
        """
        Delete raw Bayer frames older than raw_retention_days.
        Stacked composites are kept indefinitely.

        Returns
        -------
        dict
            Summary: {"directories_deleted": int, "bytes_freed": int}
        """
        cutoff = datetime.now() - timedelta(days=self._raw_retention_days)
        cutoff_str = cutoff.isoformat()

        # Find slides with raw frames older than retention period
        rows = self._db.fetchall(
            """SELECT slide_id, date FROM slides
               WHERE date < ? AND status IN ('scan_complete', 'complete')""",
            (cutoff_str,),
        )

        dirs_deleted = 0
        bytes_freed = 0

        for row in rows:
            slide_id = row["slide_id"]
            fields = self._db.fetchall(
                """SELECT field_id, image_path_raw FROM fields
                   WHERE slide_id = ? AND image_path_raw IS NOT NULL""",
                (slide_id,),
            )

            for field in fields:
                raw_path = field["image_path_raw"]
                if raw_path and os.path.isdir(raw_path):
                    dir_size = _get_dir_size(raw_path)
                    shutil.rmtree(raw_path, ignore_errors=True)
                    bytes_freed += dir_size
                    dirs_deleted += 1

                    # Clear the path in the database
                    self._db.execute(
                        "UPDATE fields SET image_path_raw = NULL WHERE field_id = ?",
                        (field["field_id"],),
                    )

        if dirs_deleted > 0:
            logger.info(
                "Raw frame cleanup: deleted %d directories, freed %.1f MB",
                dirs_deleted, bytes_freed / (1024 * 1024),
            )
        else:
            logger.debug("Raw frame cleanup: nothing to clean (retention=%d days)", self._raw_retention_days)

        return {"directories_deleted": dirs_deleted, "bytes_freed": bytes_freed}

    # ----- Portable archive -----

    def create_portable_archive(
        self,
        output_path: str,
        include_raw: bool = False,
    ) -> str:
        """
        Create a portable zip archive containing the database
        and stacked composite images (no raw frames by default).

        Parameters
        ----------
        output_path : str
            Path for the output zip file.
        include_raw : bool
            If True, include raw Bayer frames (much larger archive).

        Returns
        -------
        str
            Path to the created archive.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add database
            db_path = os.path.abspath(self._db_path)
            if os.path.isfile(db_path):
                zf.write(db_path, "cap.db")

            # Add stacked images (and optionally raw)
            image_root = Path(self._image_root)
            if image_root.is_dir():
                for file_path in image_root.rglob("*"):
                    if not file_path.is_file():
                        continue

                    rel_path = file_path.relative_to(image_root)

                    # Skip raw frames unless requested
                    if not include_raw and "raw" in rel_path.parts:
                        continue

                    zf.write(str(file_path), f"slides/{rel_path}")

        archive_size = os.path.getsize(output_path)
        logger.info(
            "Portable archive created: %s (%.1f MB)",
            output_path, archive_size / (1024 * 1024),
        )
        return output_path

    # ----- Disk usage -----

    def get_disk_usage(self) -> dict:
        """
        Get current disk usage of the data directory.

        Returns
        -------
        dict
            {"total_bytes": int, "total_gb": float, "over_limit": bool}
        """
        image_root = Path(self._image_root)
        total = _get_dir_size(str(image_root)) if image_root.is_dir() else 0

        db_size = os.path.getsize(self._db_path) if os.path.isfile(self._db_path) else 0
        total += db_size

        total_gb = total / (1024 ** 3)
        over_limit = total_gb > self._max_disk_gb

        if over_limit:
            logger.warning(
                "Disk usage %.1f GB exceeds limit of %d GB",
                total_gb, self._max_disk_gb,
            )

        return {
            "total_bytes": total,
            "total_gb": round(total_gb, 2),
            "over_limit": over_limit,
            "limit_gb": self._max_disk_gb,
        }


def _get_dir_size(path: str) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total
