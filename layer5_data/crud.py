"""
CAP CRUD Operations
====================
Create, Read, Update, Delete functions for all database tables.
All functions take a DatabaseManager instance and return typed results.

Usage:
    from cap.layer5_data.crud import insert_patient, get_slide, get_all_technicians
    patient_id = insert_patient(db, name="Buddy", species="canine", owner_name="Smith")
    slide = get_slide(db, slide_id=1)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from cap.layer5_data.db_manager import DatabaseManager
from cap.common.logging_setup import get_logger

logger = get_logger("data.crud")


# ============================================================================
# Patients
# ============================================================================

def insert_patient(
    db: DatabaseManager,
    name: str,
    species: str = "canine",
    breed: str = None,
    owner_name: str = None,
    owner_contact: str = None,
    notes: str = None,
) -> int:
    """Insert a patient record. Returns the patient_id."""
    cursor = db.execute(
        """INSERT INTO patients (name, species, breed, owner_name, owner_contact, notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (name, species, breed, owner_name, owner_contact, notes),
    )
    patient_id = cursor.lastrowid
    logger.debug("Inserted patient: id=%d, name=%s", patient_id, name)
    return patient_id


def get_patient(db: DatabaseManager, patient_id: int) -> Optional[dict]:
    """Get a patient by ID. Returns dict or None."""
    row = db.fetchone("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    return dict(row) if row else None


def search_patients(db: DatabaseManager, query: str) -> list[dict]:
    """Search patients by name or owner name (case-insensitive partial match)."""
    rows = db.fetchall(
        """SELECT * FROM patients
           WHERE name LIKE ? OR owner_name LIKE ?
           ORDER BY name""",
        (f"%{query}%", f"%{query}%"),
    )
    return [dict(r) for r in rows]


def get_all_patients(db: DatabaseManager) -> list[dict]:
    """Get all patients ordered by name."""
    rows = db.fetchall("SELECT * FROM patients ORDER BY name")
    return [dict(r) for r in rows]


# ============================================================================
# Technicians
# ============================================================================

def insert_technician(
    db: DatabaseManager,
    name: str,
    login: str,
    pin_hash: str = None,
) -> int:
    """Insert a technician. Returns the tech_id."""
    cursor = db.execute(
        "INSERT INTO technicians (name, login, pin_hash) VALUES (?, ?, ?)",
        (name, login, pin_hash),
    )
    tech_id = cursor.lastrowid
    logger.debug("Inserted technician: id=%d, name=%s", tech_id, name)
    return tech_id


def get_technician(db: DatabaseManager, tech_id: int) -> Optional[dict]:
    """Get a technician by ID."""
    row = db.fetchone("SELECT * FROM technicians WHERE tech_id = ?", (tech_id,))
    return dict(row) if row else None


def get_all_technicians(db: DatabaseManager, active_only: bool = True) -> list[dict]:
    """Get all technicians. Optionally filter to active only."""
    if active_only:
        rows = db.fetchall("SELECT * FROM technicians WHERE is_active = 1 ORDER BY name")
    else:
        rows = db.fetchall("SELECT * FROM technicians ORDER BY name")
    return [dict(r) for r in rows]


# ============================================================================
# Sessions
# ============================================================================

def start_session(db: DatabaseManager, tech_id: int) -> int:
    """Start a new login session. Returns session_id."""
    cursor = db.execute(
        "INSERT INTO sessions (tech_id) VALUES (?)",
        (tech_id,),
    )
    session_id = cursor.lastrowid
    logger.info("Session started: id=%d, tech_id=%d", session_id, tech_id)
    return session_id


def end_session(db: DatabaseManager, session_id: int) -> None:
    """End a session by setting end_time."""
    db.execute(
        """UPDATE sessions SET end_time = CURRENT_TIMESTAMP,
           slides_processed = (SELECT COUNT(*) FROM slides WHERE session_id = ?)
           WHERE session_id = ?""",
        (session_id, session_id),
    )
    logger.info("Session ended: id=%d", session_id)


# ============================================================================
# Model Versions
# ============================================================================

def insert_model_version(
    db: DatabaseManager,
    version_tag: str,
    training_date: str = None,
    dataset_size: int = None,
    validation_metrics: dict = None,
    notes: str = None,
) -> int:
    """Register a new model version. Returns version_id."""
    metrics_json = json.dumps(validation_metrics) if validation_metrics else None
    cursor = db.execute(
        """INSERT INTO model_versions
           (version_tag, training_date, dataset_size, validation_metrics_json, notes)
           VALUES (?, ?, ?, ?, ?)""",
        (version_tag, training_date, dataset_size, metrics_json, notes),
    )
    version_id = cursor.lastrowid
    logger.info("Registered model version: %s (id=%d)", version_tag, version_id)
    return version_id


def get_model_version(db: DatabaseManager, version_tag: str) -> Optional[dict]:
    """Get a model version by tag."""
    row = db.fetchone(
        "SELECT * FROM model_versions WHERE version_tag = ?", (version_tag,)
    )
    return dict(row) if row else None


# ============================================================================
# Slides
# ============================================================================

def insert_slide(
    db: DatabaseManager,
    patient_id: int = None,
    session_id: int = None,
    technician_id: int = None,
    model_version: str = None,
    scan_region_json: str = None,
    scan_region_field_count: int = None,
    notes: str = None,
) -> int:
    """Create a new slide record in pending status. Returns slide_id."""
    cursor = db.execute(
        """INSERT INTO slides
           (patient_id, session_id, technician_id, model_version,
            scan_region_json, scan_region_field_count, notes, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (patient_id, session_id, technician_id, model_version,
         scan_region_json, scan_region_field_count, notes),
    )
    slide_id = cursor.lastrowid
    logger.info("Created slide: id=%d, patient=%s", slide_id, patient_id)
    return slide_id


def update_slide_status(db: DatabaseManager, slide_id: int, status: str) -> None:
    """Update a slide's status."""
    db.execute(
        "UPDATE slides SET status = ? WHERE slide_id = ?",
        (status, slide_id),
    )
    logger.debug("Slide %d status → %s", slide_id, status)


def update_slide_scan_complete(
    db: DatabaseManager,
    slide_id: int,
    scan_duration: float,
    focus_map_json: str = None,
    focus_map_grid_size: str = None,
) -> None:
    """Mark a slide scan as complete with timing and focus data."""
    db.execute(
        """UPDATE slides SET
           status = 'scan_complete',
           scan_duration = ?,
           focus_map_json = ?,
           focus_map_grid_size = ?
           WHERE slide_id = ?""",
        (scan_duration, focus_map_json, focus_map_grid_size, slide_id),
    )
    logger.info("Slide %d scan complete: %.1f seconds", slide_id, scan_duration)


def get_slide(db: DatabaseManager, slide_id: int) -> Optional[dict]:
    """Get a slide by ID."""
    row = db.fetchone("SELECT * FROM slides WHERE slide_id = ?", (slide_id,))
    return dict(row) if row else None


def get_slides_for_patient(db: DatabaseManager, patient_id: int) -> list[dict]:
    """Get all slides for a patient, ordered by date (newest first)."""
    rows = db.fetchall(
        "SELECT * FROM slides WHERE patient_id = ? ORDER BY date DESC",
        (patient_id,),
    )
    return [dict(r) for r in rows]


def get_recent_slides(db: DatabaseManager, limit: int = 20) -> list[dict]:
    """Get the most recent slides across all patients."""
    rows = db.fetchall(
        "SELECT * FROM slides ORDER BY date DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


# ============================================================================
# Fields
# ============================================================================

def insert_field(
    db: DatabaseManager,
    slide_id: int,
    x: int,
    y: int,
    status: str = "pending",
    predicted_z: float = None,
) -> int:
    """Create a field record. Returns field_id."""
    cursor = db.execute(
        """INSERT INTO fields (slide_id, x, y, status, predicted_z)
           VALUES (?, ?, ?, ?, ?)""",
        (slide_id, x, y, status, predicted_z),
    )
    return cursor.lastrowid


def insert_fields_batch(
    db: DatabaseManager,
    slide_id: int,
    field_positions: list[tuple[int, int]],
    predicted_z_values: list[float] = None,
) -> list[int]:
    """
    Insert multiple fields for a slide in a single transaction.
    Returns list of field_ids.
    """
    conn = db.get_connection()
    field_ids = []

    for i, (x, y) in enumerate(field_positions):
        pz = predicted_z_values[i] if predicted_z_values else None
        cursor = conn.execute(
            """INSERT INTO fields (slide_id, x, y, status, predicted_z)
               VALUES (?, ?, ?, 'pending', ?)""",
            (slide_id, x, y, pz),
        )
        field_ids.append(cursor.lastrowid)

    conn.commit()
    logger.debug("Inserted %d fields for slide %d", len(field_ids), slide_id)
    return field_ids


def update_field_status(
    db: DatabaseManager,
    field_id: int,
    status: str,
    **kwargs,
) -> None:
    """
    Update a field's status and optional additional columns.

    Supported kwargs: image_path_raw, image_path_stacked, focus_score,
    actual_z, z_drift_flagged.
    """
    updates = ["status = ?"]
    params = [status]

    allowed_fields = {
        "image_path_raw", "image_path_stacked", "focus_score",
        "actual_z", "z_drift_flagged",
    }

    for key, value in kwargs.items():
        if key in allowed_fields:
            updates.append(f"{key} = ?")
            params.append(value)

    params.append(field_id)
    db.execute(
        f"UPDATE fields SET {', '.join(updates)} WHERE field_id = ?",
        tuple(params),
    )


def get_field(db: DatabaseManager, field_id: int) -> Optional[dict]:
    """Get a field by ID."""
    row = db.fetchone("SELECT * FROM fields WHERE field_id = ?", (field_id,))
    return dict(row) if row else None


def get_fields_for_slide(db: DatabaseManager, slide_id: int) -> list[dict]:
    """Get all fields for a slide, ordered by position."""
    rows = db.fetchall(
        "SELECT * FROM fields WHERE slide_id = ? ORDER BY y, x",
        (slide_id,),
    )
    return [dict(r) for r in rows]


def get_field_count_by_status(db: DatabaseManager, slide_id: int) -> dict[str, int]:
    """Get count of fields per status for a slide."""
    rows = db.fetchall(
        """SELECT status, COUNT(*) as cnt FROM fields
           WHERE slide_id = ? GROUP BY status""",
        (slide_id,),
    )
    return {r["status"]: r["cnt"] for r in rows}


# ============================================================================
# Focus Stacking Metadata
# ============================================================================

def insert_stacking_meta(
    db: DatabaseManager,
    field_id: int,
    block_size: int,
    z_depths_captured: int,
    avg_sharpness: float,
    min_sharpness: float,
    z_depth_distribution: dict,
    stacking_duration_ms: float,
    registration_shifts: list[tuple[float, float]] = None,
) -> int:
    """Insert focus stacking metadata for a field. Returns stack_id."""
    cursor = db.execute(
        """INSERT OR REPLACE INTO focus_stacking_meta
           (field_id, block_size, z_depths_captured, avg_sharpness,
            min_sharpness, z_depth_distribution_json, stacking_duration_ms,
            registration_shifts_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (field_id, block_size, z_depths_captured, avg_sharpness,
         min_sharpness, json.dumps(z_depth_distribution),
         stacking_duration_ms,
         json.dumps(registration_shifts) if registration_shifts else None),
    )
    return cursor.lastrowid


# ============================================================================
# Detections
# ============================================================================

def insert_detection(
    db: DatabaseManager,
    field_id: int,
    class_name: str,
    confidence: float,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    model_version: str = None,
) -> int:
    """Insert a single detection. Returns detection_id."""
    cursor = db.execute(
        """INSERT INTO detections
           (field_id, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (field_id, class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h, model_version),
    )
    return cursor.lastrowid


def insert_detections_batch(
    db: DatabaseManager,
    detections: list[dict],
) -> int:
    """
    Insert multiple detections in a single transaction.
    Each dict must have: field_id, class_name, confidence,
    bbox_x, bbox_y, bbox_w, bbox_h, model_version.
    Returns number of rows inserted.
    """
    db.executemany(
        """INSERT INTO detections
           (field_id, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, model_version)
           VALUES (:field_id, :class_name, :confidence,
                   :bbox_x, :bbox_y, :bbox_w, :bbox_h, :model_version)""",
        detections,
    )
    count = len(detections)
    logger.debug("Inserted %d detections in batch", count)
    return count


def get_detections_for_field(db: DatabaseManager, field_id: int) -> list[dict]:
    """Get all detections for a field."""
    rows = db.fetchall(
        "SELECT * FROM detections WHERE field_id = ? ORDER BY confidence DESC",
        (field_id,),
    )
    return [dict(r) for r in rows]


def get_detections_for_slide(db: DatabaseManager, slide_id: int) -> list[dict]:
    """Get all detections for all fields in a slide."""
    rows = db.fetchall(
        """SELECT d.*, f.x, f.y FROM detections d
           JOIN fields f ON d.field_id = f.field_id
           WHERE f.slide_id = ?
           ORDER BY d.class, d.confidence DESC""",
        (slide_id,),
    )
    return [dict(r) for r in rows]


def get_organism_counts(db: DatabaseManager, slide_id: int) -> dict[str, int]:
    """Get per-class detection counts for a slide."""
    rows = db.fetchall(
        """SELECT class, COUNT(*) as cnt FROM detections d
           JOIN fields f ON d.field_id = f.field_id
           WHERE f.slide_id = ?
           GROUP BY class""",
        (slide_id,),
    )
    return {r["class"]: r["cnt"] for r in rows}


# ============================================================================
# Results (slide-level aggregated)
# ============================================================================

def insert_results(
    db: DatabaseManager,
    slide_id: int,
    organism_counts: dict,
    severity_score: str,
    severity_grades: dict = None,
    flagged_field_ids: list[int] = None,
    model_version: str = None,
    plain_english_summary: str = None,
) -> int:
    """Insert or replace slide-level results. Returns result_id."""
    cursor = db.execute(
        """INSERT OR REPLACE INTO results
           (slide_id, organism_counts_json, severity_score, severity_grades_json,
            flagged_fields_json, model_version, plain_english_summary)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (slide_id, json.dumps(organism_counts), severity_score,
         json.dumps(severity_grades) if severity_grades else None,
         json.dumps(flagged_field_ids) if flagged_field_ids else None,
         model_version, plain_english_summary),
    )
    logger.info("Results saved for slide %d: severity=%s", slide_id, severity_score)
    return cursor.lastrowid


def get_results(db: DatabaseManager, slide_id: int) -> Optional[dict]:
    """Get results for a slide."""
    row = db.fetchone("SELECT * FROM results WHERE slide_id = ?", (slide_id,))
    if row:
        result = dict(row)
        # Parse JSON fields for convenience
        if result.get("organism_counts_json"):
            result["organism_counts"] = json.loads(result["organism_counts_json"])
        if result.get("severity_grades_json"):
            result["severity_grades"] = json.loads(result["severity_grades_json"])
        if result.get("flagged_fields_json"):
            result["flagged_field_ids"] = json.loads(result["flagged_fields_json"])
        return result
    return None


# ============================================================================
# Corrections
# ============================================================================

def insert_correction(
    db: DatabaseManager,
    detection_id: int,
    tech_id: int,
    original_class: str,
    corrected_class: str,
) -> int:
    """Log a technician correction. Returns correction_id."""
    cursor = db.execute(
        """INSERT INTO corrections
           (detection_id, tech_id, original_class, corrected_class)
           VALUES (?, ?, ?, ?)""",
        (detection_id, tech_id, original_class, corrected_class),
    )
    correction_id = cursor.lastrowid
    logger.info(
        "Correction logged: detection=%d, %s → %s (tech=%d)",
        detection_id, original_class, corrected_class, tech_id,
    )
    return correction_id


def get_unreviewed_corrections(db: DatabaseManager) -> list[dict]:
    """Get all corrections not yet reviewed by Noah."""
    rows = db.fetchall(
        """SELECT c.*, d.field_id, d.confidence, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
           FROM corrections c
           JOIN detections d ON c.detection_id = d.detection_id
           WHERE c.reviewed = 0
           ORDER BY c.timestamp""",
    )
    return [dict(r) for r in rows]


def mark_correction_reviewed(
    db: DatabaseManager,
    correction_id: int,
    reviewer_notes: str = None,
) -> None:
    """Mark a correction as reviewed."""
    db.execute(
        """UPDATE corrections SET reviewed = 1, reviewer_notes = ?
           WHERE correction_id = ?""",
        (reviewer_notes, correction_id),
    )
