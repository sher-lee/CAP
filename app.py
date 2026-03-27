"""
CAP Application Entry Point
=============================
Initializes configuration, logging, database, and hardware backend.
This is the spine of the application — everything flows from here.

Run from the project root:
    python -m cap.app

Or import and call main() programmatically.
"""

from __future__ import annotations

import sys
import os

from cap.config import load_config, CAPConfig
from cap.common.logging_setup import configure_logging, get_logger
from cap.common.backend import get_backend, Backend


def main(config_path: str | None = None) -> None:
    """
    Application startup sequence.

    Follows the order defined in the spec:
    1. Load configuration
    2. Configure logging
    3. Initialize database
    4. Initialize hardware backend (real or simulation)
    5. Log startup summary
    6. (Future) Launch UI

    Parameters
    ----------
    config_path : str, optional
        Path to cap_config.yaml. If None, uses the default location.
    """
    # Step 1: Load configuration
    config = load_config(config_path)

    # Step 2: Configure logging
    configure_logging(
        log_dir=config.logging.log_dir,
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        log_filename=config.logging.log_filename,
    )

    logger = get_logger("app")
    logger.info("=" * 60)
    logger.info("CAP — Cytology Analysis Project")
    logger.info("=" * 60)
    logger.info("Configuration loaded from: %s", config.config_path)

    # Step 3: Initialize database
    from cap.layer5_data.db_manager import DatabaseManager

    db = DatabaseManager(config.storage.db_path)
    db.initialize()
    logger.info("Database initialized: %s", config.storage.db_path)

    # Step 4: Ensure image storage directories exist
    os.makedirs(config.storage.image_root, exist_ok=True)
    logger.info("Image storage root: %s", config.storage.image_root)

    # Step 5: Initialize hardware backend
    backend = get_backend(config.to_dict())
    logger.info("Hardware backend: %s", backend.mode)

    # Step 6: Initialize hardware modules
    motor = backend.motor_controller_class(config)
    safety = backend.safety_system_class(config)
    oil_monitor = backend.oil_safety_class(config)
    camera = backend.camera_interface_class(config)
    focus = backend.preliminary_focus_class(config)
    autofocus = backend.per_field_autofocus_class(config)

    logger.info("All hardware modules initialized")

    # Step 7: Startup summary
    _log_startup_summary(logger, config, backend, db)

    # Step 8: (Future) Launch UI
    # from cap.layer6_ui.main_window import launch_ui
    # launch_ui(config, backend, db)

    logger.info("CAP initialized successfully in %s mode", config.hardware_mode)
    logger.info("Ready for UI launch (not yet implemented)")

    # Return references for interactive testing
    return AppContext(
        config=config,
        backend=backend,
        db=db,
        motor=motor,
        safety=safety,
        oil_monitor=oil_monitor,
        camera=camera,
        focus=focus,
        autofocus=autofocus,
    )


class AppContext:
    """
    Container for all initialized application components.
    Returned by main() for interactive testing and passed to the UI.
    """

    __slots__ = (
        "config", "backend", "db", "motor", "safety",
        "oil_monitor", "camera", "focus", "autofocus",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def shutdown(self):
        """Clean shutdown of all components."""
        logger = get_logger("app")
        logger.info("Shutting down CAP...")

        if hasattr(self, "camera") and hasattr(self.camera, "release"):
            self.camera.release()
        if hasattr(self, "oil_monitor") and hasattr(self.oil_monitor, "shutdown"):
            self.oil_monitor.shutdown()
        if hasattr(self, "safety") and hasattr(self.safety, "shutdown"):
            self.safety.shutdown()
        if hasattr(self, "db") and hasattr(self.db, "close"):
            self.db.close()

        logger.info("CAP shut down cleanly")


def _log_startup_summary(
    logger,
    config: CAPConfig,
    backend: Backend,
    db,
) -> None:
    """Log a summary of the startup configuration."""
    logger.info("--- Startup Summary ---")
    logger.info("  Hardware mode:     %s", config.hardware_mode)
    logger.info("  Database:          %s", config.storage.db_path)
    logger.info("  Image root:        %s", config.storage.image_root)
    logger.info("  AI enabled:        %s", config.inference.enabled)
    logger.info("  Model path:        %s", config.inference.model_path)
    logger.info("  Scan speed:        %d fields/sec", config.scan.fields_per_second)
    logger.info("  Z depths/field:    %d", config.focus.z_depths_per_field)
    logger.info("  Focus block size:  %d px", config.focus.block_size)
    logger.info("  Settle delay:      %d ms", config.motor.settle_delay_ms)
    logger.info("  Stacked format:    %s (quality %d)", config.storage.stacked_format, config.storage.stacked_jpeg_quality)
    logger.info("-----------------------")


if __name__ == "__main__":
    # Allow passing config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    ctx = main(config_path)
