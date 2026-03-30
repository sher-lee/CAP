"""
CVAT Annotation Exporter
==========================
Exports corrected annotations in CVAT XML format for import into
annotation tools (CVAT, Label Studio) or direct use in YOLOv11
retraining.

Also supports YOLO-format export (one .txt file per image with
class_id cx cy w h normalized coordinates) for direct training.

Usage:
    exporter = CVATExporter(db, config)
    exporter.export_cvat(slide_id, output_path)
    exporter.export_yolo_batch(slide_ids, output_dir)
"""

from __future__ import annotations

import os
import shutil
from typing import Optional
from xml.etree import ElementTree as ET

from cap.layer5_data.db_manager import DatabaseManager
from cap.layer5_data import crud
from cap.layer5_data.export import export_cvat_xml
from cap.layer9_retraining.corrections import CorrectionManager
from cap.common.logging_setup import get_logger

logger = get_logger("retraining.cvat_export")


class CVATExporter:
    """
    Exports annotations for retraining, with corrections applied.
    """

    def __init__(self, db: DatabaseManager, config: object = None) -> None:
        self._db = db
        self._config = config
        self._correction_mgr = CorrectionManager(db, config)

        # Get configured classes for YOLO export
        if config and hasattr(config, "inference"):
            self._classes = list(config.inference.classes)
        else:
            self._classes = [
                "cocci_small", "cocci_large", "yeast",
                "rods", "ear_mites", "empty_artifact",
            ]

    def export_cvat(
        self,
        slide_id: int,
        output_path: str,
        apply_corrections: bool = True,
    ) -> str:
        """
        Export annotations for a single slide in CVAT XML format.

        Parameters
        ----------
        slide_id : int
        output_path : str
            Path to write the XML file.
        apply_corrections : bool
            If True, applies technician corrections before export.
            If False, exports raw AI predictions.

        Returns
        -------
        str
            The output path written.
        """
        if not apply_corrections:
            # Use the existing Layer 5 export directly
            return export_cvat_xml(self._db, slide_id, output_path)

        # Get corrected annotations
        corrected = self._correction_mgr.get_corrected_annotations(slide_id)
        fields = crud.get_fields_for_slide(self._db, slide_id)

        # Build CVAT XML
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"

        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "name").text = f"slide_{slide_id}_corrected"
        ET.SubElement(task, "size").text = str(len(fields))

        labels_elem = ET.SubElement(task, "labels")
        classes_seen = set()

        # Group detections by field
        by_field: dict[int, list[dict]] = {}
        for det in corrected:
            fid = det["field_id"]
            if fid not in by_field:
                by_field[fid] = []
            by_field[fid].append(det)

        for field in fields:
            fid = field["field_id"]
            image_elem = ET.SubElement(root, "image")
            image_elem.set("id", str(fid))
            image_elem.set("name", f"{field['x']}_{field['y']}.jpg")

            for det in by_field.get(fid, []):
                box = ET.SubElement(image_elem, "box")
                box.set("label", det["class"])
                box.set("xtl", str(det["bbox_x"]))
                box.set("ytl", str(det["bbox_y"]))
                box.set("xbr", str(det["bbox_x"] + det["bbox_w"]))
                box.set("ybr", str(det["bbox_y"] + det["bbox_h"]))
                box.set("occluded", "0")
                if det.get("was_corrected"):
                    box.set("attribute", "corrected=true")
                classes_seen.add(det["class"])

        for cls_name in sorted(classes_seen):
            label = ET.SubElement(labels_elem, "label")
            ET.SubElement(label, "name").text = cls_name

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(
            "Corrected CVAT XML exported: slide %d → %s (%d annotations)",
            slide_id, output_path, len(corrected),
        )
        return output_path

    def export_yolo_format(
        self,
        slide_id: int,
        output_dir: str,
        image_width: int = 640,
        image_height: int = 640,
        apply_corrections: bool = True,
    ) -> str:
        """
        Export annotations in YOLO format (one .txt per image).

        Each line: class_id center_x center_y width height
        (all normalized to 0-1).

        Parameters
        ----------
        slide_id : int
        output_dir : str
            Directory to write label files.
        image_width, image_height : int
            Image dimensions for coordinate normalization.
        apply_corrections : bool
            If True, applies corrections before export.

        Returns
        -------
        str
            The output directory.
        """
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        if apply_corrections:
            detections = self._correction_mgr.get_corrected_annotations(slide_id)
        else:
            detections = crud.get_detections_for_slide(self._db, slide_id)

        # Group by field position
        by_field: dict[str, list[dict]] = {}
        for det in detections:
            key = f"{det.get('x', 0)}_{det.get('y', 0)}"
            if key not in by_field:
                by_field[key] = []
            by_field[key].append(det)

        # Write one .txt per field
        class_to_id = {cls: idx for idx, cls in enumerate(self._classes)}

        for field_key, dets in by_field.items():
            label_path = os.path.join(labels_dir, f"{field_key}.txt")
            with open(label_path, "w") as f:
                for det in dets:
                    cls_name = det["class"]
                    cls_id = class_to_id.get(cls_name)
                    if cls_id is None:
                        continue  # Skip unknown classes

                    # Convert bbox to YOLO format (normalized center x, y, w, h)
                    bx = det["bbox_x"]
                    by_ = det["bbox_y"]
                    bw = det["bbox_w"]
                    bh = det["bbox_h"]

                    cx = (bx + bw / 2) / image_width
                    cy = (by_ + bh / 2) / image_height
                    nw = bw / image_width
                    nh = bh / image_height

                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # Write classes.txt
        classes_path = os.path.join(output_dir, "classes.txt")
        with open(classes_path, "w") as f:
            for cls in self._classes:
                f.write(f"{cls}\n")

        logger.info(
            "YOLO format exported: slide %d → %s (%d fields)",
            slide_id, output_dir, len(by_field),
        )
        return output_dir

    def export_yolo_batch(
        self,
        slide_ids: list[int],
        output_dir: str,
        image_width: int = 640,
        image_height: int = 640,
        copy_images: bool = False,
    ) -> str:
        """
        Export multiple slides in YOLO format for batch retraining.

        Creates a dataset directory structure:
            output_dir/
                classes.txt
                labels/
                    slide_{id}_{x}_{y}.txt
                images/     (if copy_images=True)
                    slide_{id}_{x}_{y}.jpg

        Parameters
        ----------
        slide_ids : list of int
            Slides to include in the dataset.
        output_dir : str
            Root output directory.
        image_width, image_height : int
            Image dimensions for normalization.
        copy_images : bool
            If True, copies stacked composite images alongside labels.

        Returns
        -------
        str
            The output directory.
        """
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        if copy_images:
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

        class_to_id = {cls: idx for idx, cls in enumerate(self._classes)}
        total_annotations = 0

        for slide_id in slide_ids:
            corrected = self._correction_mgr.get_corrected_annotations(slide_id)
            fields = crud.get_fields_for_slide(self._db, slide_id)

            by_field_id: dict[int, list[dict]] = {}
            for det in corrected:
                fid = det["field_id"]
                if fid not in by_field_id:
                    by_field_id[fid] = []
                by_field_id[fid].append(det)

            for field in fields:
                fid = field["field_id"]
                fx, fy = field["x"], field["y"]
                filename = f"slide_{slide_id}_{fx}_{fy}"

                # Write label file
                label_path = os.path.join(labels_dir, f"{filename}.txt")
                field_dets = by_field_id.get(fid, [])
                with open(label_path, "w") as f:
                    for det in field_dets:
                        cls_id = class_to_id.get(det["class"])
                        if cls_id is None:
                            continue
                        bx, by_, bw, bh = det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]
                        cx = (bx + bw / 2) / image_width
                        cy = (by_ + bh / 2) / image_height
                        nw = bw / image_width
                        nh = bh / image_height
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                        total_annotations += 1

                # Copy image if requested
                if copy_images and field.get("image_path_stacked"):
                    src = field["image_path_stacked"]
                    if os.path.isfile(src):
                        dst = os.path.join(images_dir, f"{filename}.jpg")
                        shutil.copy2(src, dst)

        # Write classes.txt
        classes_path = os.path.join(output_dir, "classes.txt")
        with open(classes_path, "w") as f:
            for cls in self._classes:
                f.write(f"{cls}\n")

        # Write dataset.yaml (YOLOv11 training config)
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"# CAP Retraining Dataset\n")
            f.write(f"# Generated: {os.path.basename(output_dir)}\n")
            f.write(f"# Slides: {len(slide_ids)}\n")
            f.write(f"# Annotations: {total_annotations}\n\n")
            f.write(f"path: {os.path.abspath(output_dir)}\n")
            f.write(f"train: images\n")
            f.write(f"val: images\n\n")
            f.write(f"names:\n")
            for idx, cls in enumerate(self._classes):
                f.write(f"  {idx}: {cls}\n")

        logger.info(
            "YOLO batch exported: %d slides, %d annotations → %s",
            len(slide_ids), total_annotations, output_dir,
        )
        return output_dir
