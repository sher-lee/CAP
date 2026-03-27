"""
CAP Database Manager
=====================
Handles SQLite connection lifecycle, schema creation, and migrations.
All database access flows through this class.

Usage:
    db = DatabaseManager("./data/cap.db")
    db.initialize()                    # Creates tables if needed
    conn = db.get_connection()         # Get a connection for queries
    db.close()                         # Clean shutdown
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from cap.common.logging_setup import get_logger

logger = get_logger("data.db")

# Current schema version — increment when schema.sql changes
SCHEMA_VERSION = 1

# Path to schema.sql relative to this file
_SCHEMA_SQL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


class DatabaseManager:
    """
    Manages the SQLite database lifecycle.

    Responsibilities:
    - Create database file and directories if they don't exist
    - Execute schema creation on first run
    - Track schema version for future migrations
    - Provide connections with foreign keys enabled
    """

    def __init__(self, db_path: str) -> None:
        """
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file (e.g. "./data/cap.db").
            Parent directories are created automatically.
        """
        self._db_path = os.path.abspath(db_path)
        self._connection: Optional[sqlite3.Connection] = None
        logger.debug("DatabaseManager created for: %s", self._db_path)

    @property
    def db_path(self) -> str:
        return self._db_path

    def initialize(self) -> None:
        """
        Initialize the database. Creates the file and all tables
        if they don't exist. Runs migrations if the schema version
        has changed.

        Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        is_new = not os.path.exists(self._db_path)
        conn = self.get_connection()

        if is_new:
            logger.info("Creating new database: %s", self._db_path)
            self._execute_schema(conn)
            self._set_schema_version(conn, SCHEMA_VERSION)
            logger.info("Schema version %d applied", SCHEMA_VERSION)
        else:
            current_version = self._get_schema_version(conn)
            if current_version < SCHEMA_VERSION:
                logger.info(
                    "Database schema outdated (v%d → v%d), running migrations...",
                    current_version, SCHEMA_VERSION,
                )
                self._run_migrations(conn, current_version, SCHEMA_VERSION)
                self._set_schema_version(conn, SCHEMA_VERSION)
                logger.info("Migrations complete, now at schema v%d", SCHEMA_VERSION)
            else:
                logger.debug("Database schema is current (v%d)", current_version)

        conn.commit()

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with foreign keys enabled
        and Row factory set for dict-like access.

        Returns the existing connection if one is open,
        or creates a new one.

        Returns
        -------
        sqlite3.Connection
        """
        if self._connection is None:
            self._connection = sqlite3.connect(self._db_path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            logger.debug("Database connection opened: %s", self._db_path)
        return self._connection

    def execute(
        self,
        sql: str,
        params: tuple | dict = (),
        commit: bool = True,
    ) -> sqlite3.Cursor:
        """
        Execute a single SQL statement.

        Parameters
        ----------
        sql : str
            SQL statement with ? or :named placeholders.
        params : tuple or dict
            Parameter values.
        commit : bool
            Whether to auto-commit after execution.

        Returns
        -------
        sqlite3.Cursor
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params)
        if commit:
            conn.commit()
        return cursor

    def executemany(
        self,
        sql: str,
        params_list: list[tuple | dict],
        commit: bool = True,
    ) -> sqlite3.Cursor:
        """
        Execute a SQL statement for each parameter set.

        Parameters
        ----------
        sql : str
            SQL statement.
        params_list : list
            List of parameter tuples/dicts.
        commit : bool
            Whether to auto-commit.

        Returns
        -------
        sqlite3.Cursor
        """
        conn = self.get_connection()
        cursor = conn.executemany(sql, params_list)
        if commit:
            conn.commit()
        return cursor

    def fetchone(self, sql: str, params: tuple | dict = ()) -> Optional[sqlite3.Row]:
        """Execute a query and return one row, or None."""
        cursor = self.execute(sql, params, commit=False)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        cursor = self.execute(sql, params, commit=False)
        return cursor.fetchall()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        row = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def row_count(self, table_name: str) -> int:
        """Return the number of rows in a table."""
        row = self.fetchone(f"SELECT COUNT(*) as cnt FROM {table_name}")
        return row["cnt"] if row else 0

    # ----- Internal -----

    def _execute_schema(self, conn: sqlite3.Connection) -> None:
        """Execute the schema.sql file to create all tables."""
        if not os.path.isfile(_SCHEMA_SQL_PATH):
            raise FileNotFoundError(
                f"Schema file not found: {_SCHEMA_SQL_PATH}\n"
                f"Expected schema.sql in the layer5_data/ directory."
            )

        with open(_SCHEMA_SQL_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn.executescript(schema_sql)
        logger.debug("Schema SQL executed from: %s", _SCHEMA_SQL_PATH)

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get the current schema version from the database."""
        try:
            cursor = conn.execute(
                "SELECT MAX(version) as v FROM _schema_version"
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else 0
        except sqlite3.OperationalError:
            # _schema_version table doesn't exist yet
            return 0

    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        """Record a schema version in the tracking table."""
        conn.execute(
            "INSERT INTO _schema_version (version) VALUES (?)",
            (version,),
        )

    def _run_migrations(
        self,
        conn: sqlite3.Connection,
        from_version: int,
        to_version: int,
    ) -> None:
        """
        Run schema migrations from from_version to to_version.

        For now, re-executes the full schema (all CREATE IF NOT EXISTS).
        Future migrations can add ALTER TABLE statements here.
        """
        logger.info("Running migrations from v%d to v%d", from_version, to_version)

        # For the initial version, just re-run the schema
        # (CREATE IF NOT EXISTS is safe to run multiple times)
        self._execute_schema(conn)

        # Future migration example:
        # if from_version < 2:
        #     conn.execute("ALTER TABLE slides ADD COLUMN new_column TEXT")
        #     logger.info("Migration v1→v2: added slides.new_column")
