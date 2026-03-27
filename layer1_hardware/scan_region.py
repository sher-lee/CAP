"""
Scan Region Manager
=====================
Manages the technician-defined scan region polygon. Converts polygon
vertices to a field grid using serpentine raster ordering, performs
point-in-polygon filtering, and provides scan estimates.

This module is pure software — no hardware dependency. It works
identically in simulation and real modes.

Usage:
    from cap.layer1_hardware.scan_region import ScanRegionManager
    srm = ScanRegionManager(config)
    srm.set_polygon(motor_vertices)
    fields = srm.get_field_positions()
    print(srm.get_estimates())
"""

from __future__ import annotations

import json
from typing import Optional

from cap.common.dataclasses import ScanRegion
from cap.common.logging_setup import get_logger

logger = get_logger("scan_region")


class ScanRegionManager:
    """
    Generates a serpentine field grid within a technician-drawn polygon.
    All coordinates are in motor microsteps.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "camera"):
            self._fov_w_mm = config.camera.fov_width_mm
            self._fov_h_mm = config.camera.fov_height_mm
            self._x_steps_mm = config.motor.x_steps_per_mm
            self._y_steps_mm = config.motor.y_steps_per_mm
            self._fields_per_sec = config.scan.fields_per_second
            self._stacked_format = config.storage.stacked_format
        else:
            cam = config.get("camera", {})
            motor = config.get("motor", {})
            scan = config.get("scan", {})
            storage = config.get("storage", {})
            self._fov_w_mm = cam.get("fov_width_mm", 7.68)
            self._fov_h_mm = cam.get("fov_height_mm", 4.32)
            self._x_steps_mm = motor.get("x_steps_per_mm", 1000.0)
            self._y_steps_mm = motor.get("y_steps_per_mm", 1000.0)
            self._fields_per_sec = scan.get("fields_per_second", 2)
            self._stacked_format = storage.get("stacked_format", "jpeg")

        # Field size in motor steps
        self._field_w_steps = int(self._fov_w_mm * self._x_steps_mm)
        self._field_h_steps = int(self._fov_h_mm * self._y_steps_mm)

        # Current region
        self._region: Optional[ScanRegion] = None

        logger.info(
            "ScanRegionManager initialized: field_size=%dx%d steps (%.2fx%.2f mm)",
            self._field_w_steps, self._field_h_steps,
            self._fov_w_mm, self._fov_h_mm,
        )

    @property
    def field_width_steps(self) -> int:
        return self._field_w_steps

    @property
    def field_height_steps(self) -> int:
        return self._field_h_steps

    def set_polygon(self, motor_vertices: list[tuple[int, int]]) -> ScanRegion:
        """
        Define the scan region from polygon vertices in motor coordinates.
        Generates the field grid immediately.

        Parameters
        ----------
        motor_vertices : list of (x, y)
            Polygon vertices in motor microstep coordinates.

        Returns
        -------
        ScanRegion
            The populated scan region with field positions and estimates.
        """
        if len(motor_vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        field_positions = self._generate_field_grid(motor_vertices)

        field_count = len(field_positions)
        scan_time = field_count / self._fields_per_sec
        disk_mb = self._estimate_disk_mb(field_count)

        self._region = ScanRegion(
            polygon_vertices=motor_vertices,
            field_positions=field_positions,
            field_count=field_count,
            estimated_scan_time_sec=scan_time,
            estimated_disk_usage_mb=disk_mb,
        )

        logger.info(
            "Scan region set: %d vertices -> %d fields, ~%.1f sec, ~%.0f MB",
            len(motor_vertices), field_count, scan_time, disk_mb,
        )

        return self._region

    def set_preset(self, preset: str, slide_w_mm: float = 75.0, slide_h_mm: float = 25.0) -> ScanRegion:
        """
        Set a preset scan region.

        Parameters
        ----------
        preset : str
            "full_slide" or "center_half".
        """
        w_steps = int(slide_w_mm * self._x_steps_mm)
        h_steps = int(slide_h_mm * self._y_steps_mm)
        margin = int(min(w_steps, h_steps) * 0.02)

        if preset == "full_slide":
            vertices = [
                (margin, margin),
                (w_steps - margin, margin),
                (w_steps - margin, h_steps - margin),
                (margin, h_steps - margin),
            ]
        elif preset == "center_half":
            cx, cy = w_steps // 2, h_steps // 2
            hw = int(w_steps * 0.35)
            hh = int(h_steps * 0.35)
            vertices = [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ]
        else:
            raise ValueError(f"Unknown preset: {preset}")

        region = self.set_polygon(vertices)
        region.preset_name = preset
        return region

    def get_region(self) -> Optional[ScanRegion]:
        """Get the current scan region, or None if not set."""
        return self._region

    def get_field_positions(self) -> list[tuple[int, int]]:
        """Get the serpentine-ordered field positions."""
        if self._region is None:
            return []
        return self._region.field_positions

    def get_estimates(self) -> dict:
        """Get scan estimates as a dict."""
        if self._region is None:
            return {"field_count": 0, "scan_time_sec": 0, "disk_mb": 0}
        return {
            "field_count": self._region.field_count,
            "scan_time_sec": self._region.estimated_scan_time_sec,
            "disk_mb": self._region.estimated_disk_usage_mb,
        }

    def to_json(self) -> str:
        """Serialize the current region to JSON for database storage."""
        if self._region is None:
            return "{}"
        return json.dumps({
            "polygon_vertices": self._region.polygon_vertices,
            "field_count": self._region.field_count,
            "preset_name": self._region.preset_name,
        })

    def _generate_field_grid(self, polygon: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Generate a serpentine raster grid of field positions within
        the polygon, filtering to only fields whose center falls inside.
        """
        xs = [v[0] for v in polygon]
        ys = [v[1] for v in polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        half_w = self._field_w_steps // 2
        half_h = self._field_h_steps // 2

        rows = []
        y = min_y + half_h
        row_idx = 0

        while y <= max_y - half_h:
            row = []
            x = min_x + half_w

            while x <= max_x - half_w:
                if _point_in_polygon(x, y, polygon):
                    row.append((x, y))
                x += self._field_w_steps

            if row_idx % 2 == 1:
                row.reverse()

            rows.append(row)
            y += self._field_h_steps
            row_idx += 1

        fields = []
        for row in rows:
            fields.extend(row)

        return fields

    def _estimate_disk_mb(self, field_count: int) -> float:
        """Estimate disk usage in MB."""
        mb_per_field = 4.0 if self._stacked_format == "jpeg" else 24.0
        raw_per_field = 6 * 8.0  # 6 Z-depths x ~8MB raw Bayer
        return field_count * (mb_per_field + raw_per_field)


def _point_in_polygon(x: int, y: int, polygon: list[tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
