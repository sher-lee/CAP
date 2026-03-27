"""
CAP Configuration Package
==========================
Loads and provides typed access to all configurable constants.

Usage:
    from cap.config import load_config, save_config
    config = load_config()
    print(config.motor.microsteps)  # 256
"""

from cap.config.config_loader import (
    CAPConfig,
    load_config,
    save_config,
    # Section types (for type hints in other modules)
    CameraConfig,
    FocusConfig,
    InferenceConfig,
    LoggingConfig,
    MetricsConfig,
    MotorConfig,
    ProcessingConfig,
    ScanConfig,
    ScanRegionConfig,
    SimConfig,
    SlideConfig,
    StorageConfig,
    ThermalConfig,
    TransferConfig,
    UIConfig,
    VisualizationConfig,
)

__all__ = [
    "CAPConfig",
    "load_config",
    "save_config",
    "CameraConfig",
    "FocusConfig",
    "InferenceConfig",
    "LoggingConfig",
    "MetricsConfig",
    "MotorConfig",
    "ProcessingConfig",
    "ScanConfig",
    "ScanRegionConfig",
    "SimConfig",
    "SlideConfig",
    "StorageConfig",
    "ThermalConfig",
    "TransferConfig",
    "UIConfig",
    "VisualizationConfig",
]
