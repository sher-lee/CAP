"""
CAP Configuration Loader
=========================
Loads cap_config.yaml and returns a typed CAPConfig object.
Every configurable constant in the system is accessed through this
object — no module should read the YAML file directly.

Usage:
    from cap.config.config_loader import load_config
    config = load_config()                         # default path
    config = load_config("path/to/config.yaml")    # custom path

    # Access values with attribute syntax:
    config.motor.microsteps          # 256
    config.focus.block_size          # 16
    config.inference.enabled         # True
    config.storage.db_path           # "./data/cap.db"

    # Get the raw dict for passing to get_backend():
    config.to_dict()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Default config file location
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cap_config.yaml"
)


# ---------------------------------------------------------------------------
# Typed configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    camera_test_image_dir: str = "./tests/test_images/"
    motor_delay_ms: int = 10
    generate_synthetic: bool = True
    synthetic_image_width: int = 3840
    synthetic_image_height: int = 2160


@dataclass
class WhiteBalanceRGB:
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0


@dataclass
class MotorConfig:
    protocol: str = "gpio"
    microsteps: int = 256
    motor_current_ma: int = 700
    speed: int = 1000
    x_min: int = 0
    x_max: int = 100000
    y_min: int = 0
    y_max: int = 100000
    z_min: int = 0
    z_max: int = 10000
    z_full_range_degrees: float = 1.0
    z_step_size_microns: float = 0.5
    settle_delay_ms: int = 100
    homing_speed: int = 500
    stallguard_threshold: int = 50
    x_steps_per_mm: float = 1000.0
    y_steps_per_mm: float = 1000.0
    x_origin: int = 0
    y_origin: int = 0


@dataclass
class CameraConfig:
    exposure: int = 10000
    gain: float = 0.0
    white_balance_rgb: WhiteBalanceRGB = field(default_factory=WhiteBalanceRGB)
    trigger_mode: str = "hardware"
    bayer_pattern: str = "RG"
    bit_depth: int = 10
    fov_width_mm: float = 0.007
    fov_height_mm: float = 0.004


@dataclass
class FocusMapGrid:
    rows: int = 3
    cols: int = 3


@dataclass
class FocusConfig:
    z_depths_per_field: int = 6
    block_size: int = 16
    blend_sigma: float = 4.0
    max_registration_shift: int = 50
    sharpness_metric: str = "laplacian"
    coarse_sweep_steps: int = 20
    fine_sweep_steps: int = 10
    focus_map_grid: FocusMapGrid = field(default_factory=FocusMapGrid)
    drift_threshold: int = 200
    surface_fit_order: int = 2


@dataclass
class ScanRegionConfig:
    default_preset: Optional[str] = None
    oil_safety_brightness_threshold: float = 0.3


@dataclass
class SlideConfig:
    width_mm: float = 75.0
    height_mm: float = 25.0


@dataclass
class ScanConfig:
    fields_per_second: int = 2
    max_capture_retries: int = 3
    serpentine_enabled: bool = True


@dataclass
class ProcessingConfig:
    noise_filter_type: str = "gaussian"
    noise_filter_kernel: int = 3
    model_input_width: int = 640
    model_input_height: int = 640


@dataclass
class InferenceConfig:
    enabled: bool = True
    model_path: str = "./models/current.pt"
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45
    batch_size: int = 16
    classes: list[str] = field(default_factory=lambda: [
        "cocci_small", "cocci_large", "yeast", "rods", "ear_mites", "empty_artifact"
    ])
    severity_thresholds: dict[str, list[int]] = field(default_factory=lambda: {
        "default": [1, 5, 15, 30],
        "cocci_small": [1, 5, 15, 30],
        "cocci_large": [1, 5, 15, 30],
        "yeast": [1, 5, 15, 30],
        "rods": [1, 3, 10, 20],
        "ear_mites": [1, 2, 5, 10],
    })


@dataclass
class StorageConfig:
    db_path: str = "./data/cap.db"
    image_root: str = "./data/slides/"
    raw_format: str = "tiff"
    stacked_format: str = "jpeg"
    stacked_jpeg_quality: int = 95
    raw_retention_days: int = 7
    max_disk_usage_gb: int = 400
    backup_target_path: Optional[str] = None


@dataclass
class VisualizationConfig:
    stitch_overlap_px: int = 50
    tile_size: int = 256
    annotation_colors: dict[str, str] = field(default_factory=lambda: {
        "cocci_small": "#E8593C",
        "cocci_large": "#D85A30",
        "yeast": "#1D9E75",
        "rods": "#378ADD",
        "ear_mites": "#D4537E",
        "empty_artifact": "#888780",
    })


@dataclass
class TransferConfig:
    protocol: str = "smb_share"
    target_path: Optional[str] = None
    fallback_local_path: str = "./data/exports/reports/"


@dataclass
class MetricsConfig:
    export_format: str = "pdf"
    chart_library: str = "matplotlib"


@dataclass
class ThermalConfig:
    warn_temp_c: int = 85
    pause_temp_c: int = 95
    check_interval_sec: int = 5


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    console_level: str = "DEBUG"
    file_level: str = "DEBUG"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    log_filename: str = "cap.log"


@dataclass
class PreviewResolution:
    width: int = 960
    height: int = 540


@dataclass
class UIConfig:
    framework: str = "PySide6"
    preview_resolution: PreviewResolution = field(default_factory=PreviewResolution)
    theme: str = "system"
    language: str = "en"


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

@dataclass
class CAPConfig:
    """Root configuration object containing all sections."""
    hardware_mode: str = "simulation"
    sim: SimConfig = field(default_factory=SimConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    scan_region: ScanRegionConfig = field(default_factory=ScanRegionConfig)
    slide: SlideConfig = field(default_factory=SlideConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def to_dict(self) -> dict:
        """Convert back to a plain dict (e.g. for passing to get_backend)."""
        return _to_dict(self)

    @property
    def config_path(self) -> Optional[str]:
        """Path the config was loaded from, if available."""
        return getattr(self, "_config_path", None)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> CAPConfig:
    """
    Load configuration from a YAML file and return a typed CAPConfig.

    Parameters
    ----------
    path : str, optional
        Path to the YAML config file. If None, uses the default
        cap_config.yaml in the config/ directory.

    Returns
    -------
    CAPConfig
        Fully populated configuration object. Missing keys use defaults.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH

    path = os.path.abspath(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected cap_config.yaml in the config/ directory."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    config = _build_config(raw)

    # Stash the source path for diagnostics
    object.__setattr__(config, "_config_path", path)

    return config


def save_config(config: CAPConfig, path: Optional[str] = None) -> None:
    """
    Write the current configuration back to a YAML file.
    Used by the Settings screen to persist changes.

    Parameters
    ----------
    config : CAPConfig
        The configuration object to save.
    path : str, optional
        Output path. If None, overwrites the file the config was loaded from.
    """
    if path is None:
        path = config.config_path or _DEFAULT_CONFIG_PATH

    data = config.to_dict()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_config(raw: dict) -> CAPConfig:
    """Recursively build a CAPConfig from a raw dict, using defaults for missing keys."""
    return CAPConfig(
        hardware_mode=raw.get("hardware_mode", "simulation"),
        sim=_build_section(SimConfig, raw.get("sim", {})),
        motor=_build_section(MotorConfig, raw.get("motor", {})),
        camera=_build_camera(raw.get("camera", {})),
        focus=_build_focus(raw.get("focus", {})),
        scan_region=_build_section(ScanRegionConfig, raw.get("scan_region", {})),
        slide=_build_section(SlideConfig, raw.get("slide", {})),
        scan=_build_section(ScanConfig, raw.get("scan", {})),
        processing=_build_section(ProcessingConfig, raw.get("processing", {})),
        inference=_build_section(InferenceConfig, raw.get("inference", {})),
        storage=_build_section(StorageConfig, raw.get("storage", {})),
        visualization=_build_section(VisualizationConfig, raw.get("visualization", {})),
        transfer=_build_section(TransferConfig, raw.get("transfer", {})),
        metrics=_build_section(MetricsConfig, raw.get("metrics", {})),
        thermal=_build_section(ThermalConfig, raw.get("thermal", {})),
        logging=_build_section(LoggingConfig, raw.get("logging", {})),
        ui=_build_ui(raw.get("ui", {})),
    )


def _build_section(cls: type, data: Any) -> Any:
    """Build a flat dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return cls()
    # Filter to only keys the dataclass accepts
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def _build_camera(data: dict) -> CameraConfig:
    """Build CameraConfig with nested WhiteBalanceRGB."""
    if not isinstance(data, dict):
        return CameraConfig()
    wb_raw = data.pop("white_balance_rgb", {})
    wb = _build_section(WhiteBalanceRGB, wb_raw) if isinstance(wb_raw, dict) else WhiteBalanceRGB()
    cam = _build_section(CameraConfig, data)
    cam.white_balance_rgb = wb
    return cam


def _build_focus(data: dict) -> FocusConfig:
    """Build FocusConfig with nested FocusMapGrid."""
    if not isinstance(data, dict):
        return FocusConfig()
    grid_raw = data.pop("focus_map_grid", {})
    grid = _build_section(FocusMapGrid, grid_raw) if isinstance(grid_raw, dict) else FocusMapGrid()
    focus = _build_section(FocusConfig, data)
    focus.focus_map_grid = grid
    return focus


def _build_ui(data: dict) -> UIConfig:
    """Build UIConfig with nested PreviewResolution."""
    if not isinstance(data, dict):
        return UIConfig()
    res_raw = data.pop("preview_resolution", {})
    res = _build_section(PreviewResolution, res_raw) if isinstance(res_raw, dict) else PreviewResolution()
    ui = _build_section(UIConfig, data)
    ui.preview_resolution = res
    return ui


def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or primitive) to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for key in obj.__dataclass_fields__:
            if key.startswith("_"):
                continue
            result[key] = _to_dict(getattr(obj, key))
        return result
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    else:
        return obj
