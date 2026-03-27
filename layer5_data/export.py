
"""
CAP Data Export
================
Export scan data in various formats for clinic records,
AI retraining, and metrics analysis.

Formats:
- CSV: Slide results for clinic records
- CVAT XML: Annotations for the retraining pipeline
- Pandas DataFrame: For metrics engine (Layer 8)
"""

from __future__ import annotations

import csv
import json
import os
from typing import Optional
from xml.etree import ElementTree as ET

from cap.layer5_data.db_manager import DatabaseManager
from cap.layer5_data import crud
from cap.common.logging_setup import get_logger

logger = get_logger("data.export")


# ============================================================================
# CSV Export
# ============================================================================

def export_slide_csv(
    db: DatabaseManager,
    slide_id: int,
    output_path: str,
) -> str:
    """
    Export a slide's detection results as CSV.

    Columns: field_x, field_y, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h

    Parameters
    ----------
    db : DatabaseManager
    slide_id : int
    output_path : str
        Path to write the CSV file.

    Returns
    -------
    str
        The output path written.
    """
    detections = crud.get_detections_for_slide(db, slide_id)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "field_x", "field_y", "class", "confidence",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "model_version",
        ])
        for d in detections:
            writer.writerow([
                d.get("x"), d.get("y"), d["class"], d["confidence"],
                d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"],
                d.get("model_version", ""),
            ])

    logger.info("CSV exported: slide %d → %s (%d detections)", slide_id, output_path, len(detections))
    return output_path


def export_summary_csv(
    db: DatabaseManager,
    slide_id: int,
    output_path: str,
) -> str:
    """
    Export a slide summary as CSV (one row per organism class).

    Columns: class, count, severity_grade
    """
    results = crud.get_results(db, slide_id)
    if not results:
        logger.warning("No results found for slide %d", slide_id)
        return output_path

    counts = results.get("organism_counts", {})
    grades = results.get("severity_grades", {})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count", "severity_grade"])
        for cls, count in sorted(counts.items()):
            grade = grades.get(cls, "0")
            writer.writerow([cls, count, grade])

    logger.info("Summary CSV exported: slide %d → %s", slide_id, output_path)
    return output_path


# ============================================================================
# CVAT XML Export (for retraining pipeline)
# ============================================================================

def export_cvat_xml(
    db: DatabaseManager,
    slide_id: int,
    output_path: str,
    image_root: str = None,
) -> str:
    """
    Export detections in CVAT XML format for annotation review
    and model retraining.

    Parameters
    ----------
    db : DatabaseManager
    slide_id : int
    output_path : str
        Path to write the XML file.
    image_root : str, optional
        Root path to resolve image file paths.

    Returns
    -------
    str
        The output path written.
    """
    slide = crud.get_slide(db, slide_id)
    if not slide:
        raise ValueError(f"Slide {slide_id} not found")

    fields = crud.get_fields_for_slide(db, slide_id)

    # Build CVAT XML structure
    root = ET.Element("annotations")

    # Version info
    version = ET.SubElement(root, "version")
    version.text = "1.1"

    # Meta
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = f"slide_{slide_id}"
    ET.SubElement(task, "size").text = str(len(fields))

    # Labels
    labels_elem = ET.SubElement(task, "labels")
    classes_seen = set()

    for field in fields:
        detections = crud.get_detections_for_field(db, field["field_id"])

        # Image element
        image_elem = ET.SubElement(root, "image")
        image_elem.set("id", str(field["field_id"]))
        image_elem.set("name", f"{field['x']}_{field['y']}.jpg")
        if field.get("image_path_stacked"):
            image_elem.set("name", os.path.basename(field["image_path_stacked"]))

        for det in detections:
            box = ET.SubElement(image_elem, "box")
            box.set("label", det["class"])
            box.set("xtl", str(det["bbox_x"]))
            box.set("ytl", str(det["bbox_y"]))
            box.set("xbr", str(det["bbox_x"] + det["bbox_w"]))
            box.set("ybr", str(det["bbox_y"] + det["bbox_h"]))
            box.set("occluded", "0")

            # Track unique classes for the labels section
            classes_seen.add(det["class"])

    # Add label definitions
    for cls_name in sorted(classes_seen):
        label = ET.SubElement(labels_elem, "label")
        ET.SubElement(label, "name").text = cls_name

    # Write XML
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    logger.info("CVAT XML exported: slide %d → %s (%d fields)", slide_id, output_path, len(fields))
    return output_path


# ============================================================================
# Pandas DataFrame Export (for metrics engine)
# ============================================================================

def get_slides_dataframe(db: DatabaseManager):
    """
    Get all slides as a Pandas DataFrame.
    Requires pandas to be installed.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame exports: pip install pandas")

    rows = db.fetchall("SELECT * FROM slides ORDER BY date DESC")
    return pd.DataFrame([dict(r) for r in rows])


def get_detections_dataframe(db: DatabaseManager, slide_id: int = None):
    """
    Get detections as a Pandas DataFrame.
    If slide_id is provided, filters to that slide.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame exports: pip install pandas")

    if slide_id:
        detections = crud.get_detections_for_slide(db, slide_id)
    else:
        rows = db.fetchall(
            """SELECT d.*, f.slide_id, f.x, f.y FROM detections d
               JOIN fields f ON d.field_id = f.field_id
               ORDER BY f.slide_id, d.class"""
        )
        detections = [dict(r) for r in rows]

    return pd.DataFrame(detections)


def get_corrections_dataframe(db: DatabaseManager):
    """
    Get all corrections as a Pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame exports: pip install pandas")

    rows = db.fetchall(
        """SELECT c.*, d.class as ai_class, d.confidence, f.slide_id
           FROM corrections c
           JOIN detections d ON c.detection_id = d.detection_id
           JOIN fields f ON d.field_id = f.field_id
           ORDER BY c.timestamp"""
    )
    return pd.DataFrame([dict(r) for r in rows])
