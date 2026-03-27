
"""
Coordinate Mapper
==================
Converts between three coordinate systems:
1. UI pixels (what the technician sees on the polygon drawing widget)
2. Slide millimeters (physical slide dimensions)
3. Motor microsteps (what the motors understand)

Calibration values (steps_per_mm, origin offsets) are loaded from
config and updated during the hardware calibration routine.

Usage:
    from cap.layer1_hardware.coordinate_mapper import CoordinateMapper
    mapper = CoordinateMapper(config)
    motor_x, motor_y = mapper.mm_to_motor(37.5, 12.5)
    mm_x, mm_y = mapper.motor_to_mm(37500, 12500)
    frac_x, frac_y = mapper.motor_to_fractional(37500, 12500)
"""

from __future__ import annotations

from cap.common.logging_setup import get_logger

logger = get_logger("coordinate_mapper")


class CoordinateMapper:
    """
    Bidirectional coordinate conversion between UI, slide, and motor spaces.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "motor"):
            self._x_steps_per_mm = config.motor.x_steps_per_mm
            self._y_steps_per_mm = config.motor.y_steps_per_mm
            self._x_origin = config.motor.x_origin
            self._y_origin = config.motor.y_origin
            self._slide_w_mm = config.slide.width_mm
            self._slide_h_mm = config.slide.height_mm
        else:
            motor = config.get("motor", {})
            slide = config.get("slide", {})
            self._x_steps_per_mm = motor.get("x_steps_per_mm", 1000.0)
            self._y_steps_per_mm = motor.get("y_steps_per_mm", 1000.0)
            self._x_origin = motor.get("x_origin", 0)
            self._y_origin = motor.get("y_origin", 0)
            self._slide_w_mm = slide.get("width_mm", 75.0)
            self._slide_h_mm = slide.get("height_mm", 25.0)

        logger.info(
            "CoordinateMapper initialized: %.1f steps/mm (X), %.1f steps/mm (Y), "
            "slide=%.1fx%.1f mm, origin=(%d, %d)",
            self._x_steps_per_mm, self._y_steps_per_mm,
            self._slide_w_mm, self._slide_h_mm,
            self._x_origin, self._y_origin,
        )

    # ----- Slide mm <-> Motor steps -----

    def mm_to_motor(self, mm_x: float, mm_y: float) -> tuple[int, int]:
        """
        Convert slide millimeter coordinates to motor microstep coordinates.

        Parameters
        ----------
        mm_x, mm_y : float
            Position in millimeters relative to the slide origin.

        Returns
        -------
        tuple[int, int]
            Motor position in microsteps.
        """
        motor_x = int(mm_x * self._x_steps_per_mm) + self._x_origin
        motor_y = int(mm_y * self._y_steps_per_mm) + self._y_origin
        return (motor_x, motor_y)

    def motor_to_mm(self, motor_x: int, motor_y: int) -> tuple[float, float]:
        """
        Convert motor microstep coordinates to slide millimeters.

        Parameters
        ----------
        motor_x, motor_y : int
            Motor position in microsteps.

        Returns
        -------
        tuple[float, float]
            Position in millimeters relative to the slide origin.
        """
        mm_x = (motor_x - self._x_origin) / self._x_steps_per_mm
        mm_y = (motor_y - self._y_origin) / self._y_steps_per_mm
        return (mm_x, mm_y)

    # ----- Fractional (0-1) <-> Motor steps -----

    def fractional_to_motor(self, frac_x: float, frac_y: float) -> tuple[int, int]:
        """
        Convert fractional slide coordinates (0-1 range) to motor steps.
        Used by the UI polygon tool which works in fractional coordinates.

        Parameters
        ----------
        frac_x, frac_y : float
            Position as fraction of slide dimensions (0.0 to 1.0).

        Returns
        -------
        tuple[int, int]
            Motor position in microsteps.
        """
        mm_x = frac_x * self._slide_w_mm
        mm_y = frac_y * self._slide_h_mm
        return self.mm_to_motor(mm_x, mm_y)

    def motor_to_fractional(self, motor_x: int, motor_y: int) -> tuple[float, float]:
        """
        Convert motor steps to fractional slide coordinates (0-1 range).

        Parameters
        ----------
        motor_x, motor_y : int
            Motor position in microsteps.

        Returns
        -------
        tuple[float, float]
            Position as fraction of slide dimensions.
        """
        mm_x, mm_y = self.motor_to_mm(motor_x, motor_y)
        frac_x = mm_x / self._slide_w_mm if self._slide_w_mm > 0 else 0.0
        frac_y = mm_y / self._slide_h_mm if self._slide_h_mm > 0 else 0.0
        return (frac_x, frac_y)

    # ----- Fractional <-> mm -----

    def fractional_to_mm(self, frac_x: float, frac_y: float) -> tuple[float, float]:
        """Convert fractional (0-1) to millimeters."""
        return (frac_x * self._slide_w_mm, frac_y * self._slide_h_mm)

    def mm_to_fractional(self, mm_x: float, mm_y: float) -> tuple[float, float]:
        """Convert millimeters to fractional (0-1)."""
        frac_x = mm_x / self._slide_w_mm if self._slide_w_mm > 0 else 0.0
        frac_y = mm_y / self._slide_h_mm if self._slide_h_mm > 0 else 0.0
        return (frac_x, frac_y)

    # ----- Bulk conversion -----

    def fractional_polygon_to_motor(
        self, fractional_vertices: list[tuple[float, float]]
    ) -> list[tuple[int, int]]:
        """Convert a list of fractional polygon vertices to motor coordinates."""
        return [self.fractional_to_motor(fx, fy) for fx, fy in fractional_vertices]

    def motor_polygon_to_fractional(
        self, motor_vertices: list[tuple[int, int]]
    ) -> list[tuple[float, float]]:
        """Convert a list of motor polygon vertices to fractional coordinates."""
        return [self.motor_to_fractional(mx, my) for mx, my in motor_vertices]

    # ----- Field of view -----

    def get_fov_motor_steps(self) -> tuple[int, int]:
        """Get the field of view size in motor steps."""
        w = int(self._fov_w_mm * self._x_steps_per_mm) if hasattr(self, "_fov_w_mm") else self._field_w_steps
        h = int(self._fov_h_mm * self._y_steps_per_mm) if hasattr(self, "_fov_h_mm") else self._field_h_steps
        return (w, h)

    # ----- Calibration -----

    def update_calibration(
        self,
        x_steps_per_mm: float = None,
        y_steps_per_mm: float = None,
        x_origin: int = None,
        y_origin: int = None,
    ) -> None:
        """
        Update calibration values. Called after running the hardware
        calibration routine with a stage micrometer.

        Parameters
        ----------
        x_steps_per_mm : float, optional
        y_steps_per_mm : float, optional
        x_origin : int, optional
        y_origin : int, optional
        """
        if x_steps_per_mm is not None:
            self._x_steps_per_mm = x_steps_per_mm
        if y_steps_per_mm is not None:
            self._y_steps_per_mm = y_steps_per_mm
        if x_origin is not None:
            self._x_origin = x_origin
        if y_origin is not None:
            self._y_origin = y_origin

        logger.info(
            "Calibration updated: %.1f steps/mm (X), %.1f steps/mm (Y), "
            "origin=(%d, %d)",
            self._x_steps_per_mm, self._y_steps_per_mm,
            self._x_origin, self._y_origin,
        )

    @property
    def slide_dimensions_mm(self) -> tuple[float, float]:
        """Slide dimensions as (width_mm, height_mm)."""
        return (self._slide_w_mm, self._slide_h_mm)

    @property
    def steps_per_mm(self) -> tuple[float, float]:
        """Calibration values as (x_steps_per_mm, y_steps_per_mm)."""
        return (self._x_steps_per_mm, self._y_steps_per_mm)
