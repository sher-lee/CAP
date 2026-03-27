"""
Polygon Drawing Widget
========================
Custom QWidget that displays a schematic slide view and lets the
technician draw a polygon to define the scan region.

Features:
- Click to add vertices
- Double-click to close the polygon
- Drag vertices to adjust
- Right-click to undo last vertex
- Shows field grid preview inside the polygon
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPolygonF, QMouseEvent,
    QPainterPath, QFont,
)

from cap.common.logging_setup import get_logger

logger = get_logger("ui.polygon")

# Colors
SLIDE_BG = QColor(245, 240, 235)        # Light beige (slide background)
SLIDE_BORDER = QColor(180, 175, 170)     # Gray border
POLYGON_FILL = QColor(29, 158, 117, 50)  # Teal transparent
POLYGON_STROKE = QColor(29, 158, 117)    # Teal solid
VERTEX_COLOR = QColor(29, 158, 117)      # Teal vertices
VERTEX_HOVER = QColor(15, 110, 86)       # Darker teal on hover
GRID_COLOR = QColor(55, 138, 221, 80)    # Blue transparent (field grid)
CROSSHAIR = QColor(200, 200, 200)        # Light crosshair


class PolygonDrawWidget(QWidget):
    """
    Interactive polygon drawing tool on a schematic slide view.

    Emits polygon_changed whenever the polygon is modified,
    and polygon_closed when the polygon is finalized.
    """

    polygon_changed = Signal(list)   # list of (x_frac, y_frac) tuples (0-1 range)
    polygon_closed = Signal(list)    # emitted when polygon is finalized

    # Vertex grab radius in pixels
    VERTEX_RADIUS = 7
    GRAB_RADIUS = 14

    def __init__(
        self,
        slide_width_mm: float = 75.0,
        slide_height_mm: float = 25.0,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent)

        self._slide_w_mm = slide_width_mm
        self._slide_h_mm = slide_height_mm

        # Polygon state
        self._vertices: list[QPointF] = []  # In widget pixel coordinates
        self._is_closed = False
        self._dragging_idx: Optional[int] = None
        self._hover_idx: Optional[int] = None

        # Field grid preview
        self._field_grid: list[QPointF] = []
        self._field_size_px: float = 0

        # Mouse tracking for crosshair
        self._mouse_pos: Optional[QPointF] = None
        self.setMouseTracking(True)

        # Minimum size
        self.setMinimumSize(600, 220)

        logger.debug("PolygonDrawWidget initialized: %.0fx%.0f mm slide", slide_width_mm, slide_height_mm)

    # ----- Public API -----

    def clear(self) -> None:
        """Clear the polygon and reset state."""
        self._vertices.clear()
        self._is_closed = False
        self._dragging_idx = None
        self._hover_idx = None
        self._field_grid.clear()
        self.polygon_changed.emit([])
        self.update()

    def set_preset(self, preset: str) -> None:
        """
        Set a preset polygon shape.

        Parameters
        ----------
        preset : str
            "full_slide" or "center_half"
        """
        r = self._slide_rect()
        margin = 10

        if preset == "full_slide":
            self._vertices = [
                QPointF(r.left() + margin, r.top() + margin),
                QPointF(r.right() - margin, r.top() + margin),
                QPointF(r.right() - margin, r.bottom() - margin),
                QPointF(r.left() + margin, r.bottom() - margin),
            ]
        elif preset == "center_half":
            cx, cy = r.center().x(), r.center().y()
            hw = r.width() * 0.35
            hh = r.height() * 0.35
            self._vertices = [
                QPointF(cx - hw, cy - hh),
                QPointF(cx + hw, cy - hh),
                QPointF(cx + hw, cy + hh),
                QPointF(cx - hw, cy + hh),
            ]
        else:
            logger.warning("Unknown preset: %s", preset)
            return

        self._is_closed = True
        self._emit_polygon()
        self.update()

    def get_polygon_fractional(self) -> list[tuple[float, float]]:
        """
        Get polygon vertices as fractional coordinates (0-1 range)
        relative to the slide dimensions.
        """
        r = self._slide_rect()
        if r.width() == 0 or r.height() == 0:
            return []

        return [
            (
                (v.x() - r.left()) / r.width(),
                (v.y() - r.top()) / r.height(),
            )
            for v in self._vertices
        ]

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    def set_field_grid(self, field_centers: list[tuple[float, float]], field_size_frac: float) -> None:
        """
        Display a field grid preview inside the polygon.

        Parameters
        ----------
        field_centers : list of (x_frac, y_frac)
            Field center positions in fractional slide coordinates.
        field_size_frac : float
            Field size as a fraction of slide width.
        """
        r = self._slide_rect()
        self._field_grid = [
            QPointF(r.left() + x * r.width(), r.top() + y * r.height())
            for x, y in field_centers
        ]
        self._field_size_px = field_size_frac * r.width()
        self.update()

    def undo_last_vertex(self) -> None:
        """Remove the last added vertex."""
        if self._vertices and not self._is_closed:
            self._vertices.pop()
            self._emit_polygon()
            self.update()
        elif self._is_closed:
            # Reopen the polygon for editing
            self._is_closed = False
            self._vertices.pop()
            self._emit_polygon()
            self.update()

    # ----- Painting -----

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        slide_rect = self._slide_rect()

        # Draw slide background
        painter.setPen(QPen(SLIDE_BORDER, 2))
        painter.setBrush(QBrush(SLIDE_BG))
        painter.drawRoundedRect(slide_rect, 4, 4)

        # Draw scale markers
        self._draw_scale_markers(painter, slide_rect)

        # Draw crosshair at mouse position
        if self._mouse_pos and not self._is_closed:
            painter.setPen(QPen(CROSSHAIR, 1, Qt.PenStyle.DashLine))
            painter.drawLine(
                int(slide_rect.left()), int(self._mouse_pos.y()),
                int(slide_rect.right()), int(self._mouse_pos.y()),
            )
            painter.drawLine(
                int(self._mouse_pos.x()), int(slide_rect.top()),
                int(self._mouse_pos.x()), int(slide_rect.bottom()),
            )

        # Draw field grid preview
        if self._field_grid and self._field_size_px > 0:
            half = self._field_size_px / 2
            painter.setPen(QPen(GRID_COLOR, 1))
            painter.setBrush(QBrush(QColor(55, 138, 221, 30)))
            for pt in self._field_grid:
                painter.drawRect(QRectF(pt.x() - half, pt.y() - half, self._field_size_px, self._field_size_px))

        # Draw polygon
        if len(self._vertices) >= 2:
            # Fill
            if self._is_closed and len(self._vertices) >= 3:
                poly = QPolygonF(self._vertices)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(POLYGON_FILL))
                painter.drawPolygon(poly)

            # Stroke
            painter.setPen(QPen(POLYGON_STROKE, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._vertices) - 1):
                painter.drawLine(self._vertices[i], self._vertices[i + 1])
            if self._is_closed:
                painter.drawLine(self._vertices[-1], self._vertices[0])

        # Draw vertices
        for i, v in enumerate(self._vertices):
            color = VERTEX_HOVER if i == self._hover_idx else VERTEX_COLOR
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(v, self.VERTEX_RADIUS, self.VERTEX_RADIUS)

        # Draw instructions
        if not self._vertices:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(
                slide_rect, Qt.AlignmentFlag.AlignCenter,
                "Click to draw scan region\nDouble-click to close shape"
            )

        painter.end()

    def _draw_scale_markers(self, painter: QPainter, rect: QRectF) -> None:
        """Draw mm scale markers along the slide edges."""
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(QFont("Arial", 8))

        # Horizontal markers every 10mm
        px_per_mm_x = rect.width() / self._slide_w_mm
        for mm in range(0, int(self._slide_w_mm) + 1, 10):
            x = rect.left() + mm * px_per_mm_x
            painter.drawLine(int(x), int(rect.bottom()), int(x), int(rect.bottom() + 5))
            if mm % 20 == 0:
                painter.drawText(int(x - 10), int(rect.bottom() + 18), f"{mm}mm")

    # ----- Mouse events -----

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if not self._slide_rect().contains(pos):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on an existing vertex (for dragging)
            hit = self._hit_test_vertex(pos)
            if hit is not None:
                self._dragging_idx = hit
                return

            # Add new vertex (if polygon is not closed)
            if not self._is_closed:
                self._vertices.append(QPointF(pos))
                self._emit_polygon()
                self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            self.undo_last_vertex()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click to close the polygon."""
        if event.button() == Qt.MouseButton.LeftButton:
            if len(self._vertices) >= 3 and not self._is_closed:
                self._is_closed = True
                self._emit_polygon()
                self.polygon_closed.emit(self.get_polygon_fractional())
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        self._mouse_pos = pos

        if self._dragging_idx is not None:
            # Constrain to slide rect
            r = self._slide_rect()
            x = max(r.left(), min(r.right(), pos.x()))
            y = max(r.top(), min(r.bottom(), pos.y()))
            self._vertices[self._dragging_idx] = QPointF(x, y)
            self._emit_polygon()
        else:
            # Update hover state
            self._hover_idx = self._hit_test_vertex(pos)

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_idx = None

    # ----- Helpers -----

    def _slide_rect(self) -> QRectF:
        """Calculate the slide rectangle centered in the widget with aspect ratio preserved."""
        w = self.width()
        h = self.height()
        margin = 30

        avail_w = w - 2 * margin
        avail_h = h - 2 * margin

        slide_aspect = self._slide_w_mm / self._slide_h_mm
        avail_aspect = avail_w / max(avail_h, 1)

        if avail_aspect > slide_aspect:
            # Height-constrained
            rect_h = avail_h
            rect_w = rect_h * slide_aspect
        else:
            # Width-constrained
            rect_w = avail_w
            rect_h = rect_w / slide_aspect

        x = (w - rect_w) / 2
        y = (h - rect_h) / 2

        return QRectF(x, y, rect_w, rect_h)

    def _hit_test_vertex(self, pos: QPointF) -> Optional[int]:
        """Return index of vertex under the cursor, or None."""
        for i, v in enumerate(self._vertices):
            dx = pos.x() - v.x()
            dy = pos.y() - v.y()
            if (dx * dx + dy * dy) <= self.GRAB_RADIUS ** 2:
                return i
        return None

    def _emit_polygon(self) -> None:
        """Emit the current polygon as fractional coordinates."""
        self.polygon_changed.emit(self.get_polygon_fractional())
