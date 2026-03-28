"""
Tile Pyramid Builder
=====================
Generates Deep Zoom Image (DZI) format tile pyramids from
a stitched whole-slide composite. The zoomable viewer loads
tiles on demand for smooth pan/zoom without loading the full
composite into memory.

Output structure:
    tiles/
    ├── slide.dzi           (XML descriptor)
    └── slide_files/
        ├── 0/              (most zoomed out — single tile)
        │   └── 0_0.jpg
        ├── 1/
        │   ├── 0_0.jpg
        │   └── 1_0.jpg
        ├── ...
        └── N/              (full resolution)
            ├── 0_0.jpg
            ├── 1_0.jpg
            └── ...
"""

from __future__ import annotations

import math
import os
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from cap.common.logging_setup import get_logger

logger = get_logger("visualization.tiles")


class TilePyramidBuilder:
    """
    Generates DZI tile pyramids from a stitched composite.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "visualization"):
            self._tile_size = config.visualization.tile_size
        else:
            vis = config.get("visualization", {})
            self._tile_size = vis.get("tile_size", 256)

        if hasattr(config, "storage"):
            self._jpeg_quality = config.storage.stacked_jpeg_quality
        else:
            storage = config.get("storage", {})
            self._jpeg_quality = storage.get("stacked_jpeg_quality", 95)

        self._overlap = 1  # DZI standard overlap

        logger.info(
            "TilePyramidBuilder initialized: tile_size=%d, overlap=%d",
            self._tile_size, self._overlap,
        )

    def build(
        self,
        composite: np.ndarray,
        output_dir: str,
        name: str = "slide",
    ) -> str:
        """
        Generate a full DZI tile pyramid from a composite image.

        Parameters
        ----------
        composite : np.ndarray
            Stitched composite, shape (H, W, 3), dtype uint8.
        output_dir : str
            Directory to write the tiles (e.g. /slides/{id}/tiles/).
        name : str
            Base name for the DZI file and tile directory.

        Returns
        -------
        str
            Path to the .dzi descriptor file.
        """
        h, w = composite.shape[:2]

        # Calculate number of levels
        max_dim = max(w, h)
        max_level = int(math.ceil(math.log2(max_dim))) + 1

        tiles_dir = os.path.join(output_dir, f"{name}_files")
        os.makedirs(tiles_dir, exist_ok=True)

        total_tiles = 0

        # Generate tiles at each level (from full resolution down to 1x1)
        for level in range(max_level, -1, -1):
            # Scale factor for this level
            scale = 2 ** (max_level - level)
            level_w = max(1, int(math.ceil(w / scale)))
            level_h = max(1, int(math.ceil(h / scale)))

            # Resize composite to this level's resolution
            if level == max_level:
                level_image = composite
            else:
                level_image = cv2.resize(
                    composite, (level_w, level_h),
                    interpolation=cv2.INTER_AREA,
                )

            # Create level directory
            level_dir = os.path.join(tiles_dir, str(level))
            os.makedirs(level_dir, exist_ok=True)

            # Slice into tiles
            tile_size = self._tile_size
            overlap = self._overlap

            n_tiles_x = int(math.ceil(level_w / tile_size))
            n_tiles_y = int(math.ceil(level_h / tile_size))

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    # Calculate tile bounds with overlap
                    x0 = max(0, tx * tile_size - overlap)
                    y0 = max(0, ty * tile_size - overlap)
                    x1 = min(level_w, (tx + 1) * tile_size + overlap)
                    y1 = min(level_h, (ty + 1) * tile_size + overlap)

                    tile = level_image[y0:y1, x0:x1]

                    if tile.size == 0:
                        continue

                    tile_path = os.path.join(level_dir, f"{tx}_{ty}.jpg")
                    cv2.imwrite(
                        tile_path, tile,
                        [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                    )
                    total_tiles += 1

        # Write DZI descriptor XML
        dzi_path = os.path.join(output_dir, f"{name}.dzi")
        self._write_dzi_descriptor(dzi_path, w, h, name)

        logger.info(
            "Tile pyramid built: %d levels, %d tiles, %dx%d px → %s",
            max_level + 1, total_tiles, w, h, dzi_path,
        )

        return dzi_path

    def _write_dzi_descriptor(
        self, path: str, width: int, height: int, name: str
    ) -> None:
        """Write the DZI XML descriptor file."""
        root = ET.Element("Image")
        root.set("xmlns", "http://schemas.microsoft.com/deepzoom/2008")
        root.set("Format", "jpg")
        root.set("Overlap", str(self._overlap))
        root.set("TileSize", str(self._tile_size))

        size = ET.SubElement(root, "Size")
        size.set("Width", str(width))
        size.set("Height", str(height))

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def get_tile_path(
        self,
        tiles_dir: str,
        level: int,
        tile_x: int,
        tile_y: int,
        name: str = "slide",
    ) -> str:
        """
        Get the file path for a specific tile.
        Used by the zoomable viewer to load tiles on demand.

        Parameters
        ----------
        tiles_dir : str
            Root tiles directory.
        level : int
            Zoom level.
        tile_x, tile_y : int
            Tile grid coordinates.
        name : str
            DZI base name.

        Returns
        -------
        str
            Path to the tile JPEG file.
        """
        return os.path.join(tiles_dir, f"{name}_files", str(level), f"{tile_x}_{tile_y}.jpg")
