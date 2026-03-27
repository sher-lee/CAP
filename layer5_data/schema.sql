
-- =============================================================================
-- CAP Database Schema
-- =============================================================================
-- All tables for the Cytology Analysis Project.
-- SQLite dialect. Foreign keys must be enabled per-connection:
--     PRAGMA foreign_keys = ON;
--
-- This file is executed by db_manager.py on first run.
-- Schema version is tracked in the _schema_version table.
-- =============================================================================

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_version (
    version     INTEGER NOT NULL,
    applied_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- Patients and technicians (must exist before slides)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    species         TEXT NOT NULL DEFAULT 'canine',
    breed           TEXT,
    owner_name      TEXT,
    owner_contact   TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS technicians (
    tech_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    login           TEXT NOT NULL UNIQUE,
    pin_hash        TEXT,           -- NULL for prototype (no PIN); bcrypt hash for production
    is_active       BOOLEAN DEFAULT 1,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- Sessions
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tech_id         INTEGER NOT NULL REFERENCES technicians(tech_id),
    start_time      DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time        DATETIME,
    slides_processed INTEGER DEFAULT 0
);


-- ---------------------------------------------------------------------------
-- Model versions (must exist before slides and detections)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_versions (
    version_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag             TEXT NOT NULL UNIQUE,
    training_date           DATETIME,
    dataset_size            INTEGER,
    validation_metrics_json TEXT,
    notes                   TEXT,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- Slides
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS slides (
    slide_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id              INTEGER REFERENCES patients(patient_id),
    session_id              INTEGER REFERENCES sessions(session_id),
    technician_id           INTEGER REFERENCES technicians(tech_id),
    date                    DATETIME DEFAULT CURRENT_TIMESTAMP,
    scan_duration           REAL,               -- seconds
    status                  TEXT NOT NULL DEFAULT 'pending',
        -- pending / scanning / scan_complete / inferring / complete / failed
    model_version           TEXT,
    in_oil                  BOOLEAN DEFAULT 1,
    scan_region_json        TEXT,               -- polygon vertices in motor coords
    scan_region_field_count INTEGER,
    focus_map_json          TEXT,               -- fitted surface coefficients + sample points
    focus_map_grid_size     TEXT,               -- e.g. "3x3"
    notes                   TEXT
);


-- ---------------------------------------------------------------------------
-- Fields (one per X/Y position on a slide)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fields (
    field_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id            INTEGER NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    x                   INTEGER NOT NULL,       -- grid coordinate
    y                   INTEGER NOT NULL,       -- grid coordinate
    image_path_raw      TEXT,                   -- path to raw Bayer frames directory
    image_path_stacked  TEXT,                   -- path to focus-stacked composite
    focus_score         REAL,                   -- overall Laplacian variance
    status              TEXT NOT NULL DEFAULT 'pending',
        -- pending / capturing / captured / processing / processed / stacked / failed / skipped
    predicted_z         REAL,                   -- from focus map interpolation
    actual_z            REAL,                   -- from per-field autofocus
    z_drift_flagged     BOOLEAN DEFAULT 0,
    UNIQUE(slide_id, x, y)
);


-- ---------------------------------------------------------------------------
-- Focus stacking metadata (one per field, block-level detail in files)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS focus_stacking_meta (
    stack_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id                INTEGER NOT NULL REFERENCES fields(field_id) ON DELETE CASCADE,
    block_size              INTEGER,            -- pixels (e.g. 16 or 32)
    z_depths_captured       INTEGER,
    avg_sharpness           REAL,
    min_sharpness           REAL,
    z_depth_distribution_json TEXT,             -- {z_index: block_count}
    stacking_duration_ms    REAL,
    registration_shifts_json TEXT,              -- [(dx, dy), ...] for frames 1-5
    UNIQUE(field_id)
);


-- ---------------------------------------------------------------------------
-- Detections (individual organism detections from AI)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS detections (
    detection_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id        INTEGER NOT NULL REFERENCES fields(field_id) ON DELETE CASCADE,
    class           TEXT NOT NULL,              -- organism class label
    confidence      REAL NOT NULL,             -- 0.0–1.0
    bbox_x          REAL NOT NULL,
    bbox_y          REAL NOT NULL,
    bbox_w          REAL NOT NULL,
    bbox_h          REAL NOT NULL,
    model_version   TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_field ON detections(field_id);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class);


-- ---------------------------------------------------------------------------
-- Results (slide-level aggregated results)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id            INTEGER NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    organism_counts_json TEXT,                  -- {"cocci_small": 42, "yeast": 17}
    severity_score      TEXT,                  -- highest severity grade (e.g. "3+")
    severity_grades_json TEXT,                 -- {"cocci_small": "2+", "yeast": "3+"}
    flagged_fields_json TEXT,                  -- [field_id, ...]
    model_version       TEXT,
    plain_english_summary TEXT,                -- auto-generated
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(slide_id)
);


-- ---------------------------------------------------------------------------
-- Corrections (human-in-the-loop feedback)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS corrections (
    correction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id    INTEGER NOT NULL REFERENCES detections(detection_id) ON DELETE CASCADE,
    tech_id         INTEGER NOT NULL REFERENCES technicians(tech_id),
    original_class  TEXT NOT NULL,
    corrected_class TEXT NOT NULL,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    reviewed        BOOLEAN DEFAULT 0,         -- reviewed by Noah before retraining
    reviewer_notes  TEXT
);

CREATE INDEX IF NOT EXISTS idx_corrections_reviewed ON corrections(reviewed);


-- ---------------------------------------------------------------------------
-- Audit log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type  TEXT NOT NULL,
    user_id     INTEGER,                       -- tech_id or NULL for system events
    details     TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);
