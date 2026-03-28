"""
Severity Calculator
=====================
Maps organism detection counts to the semi-quantitative veterinary
severity scale (0, 1+, 2+, 3+, 4+) and generates plain-English
summaries for the PDF report.

Thresholds are configurable per organism class in cap_config.yaml.
"""

from __future__ import annotations

from cap.common.dataclasses import SeverityGrade
from cap.common.logging_setup import get_logger

logger = get_logger("visualization.severity")

# Grade display names
_GRADE_NAMES = {
    SeverityGrade.NONE: "none seen",
    SeverityGrade.RARE: "rare/few",
    SeverityGrade.MODERATE: "moderate",
    SeverityGrade.MANY: "many",
    SeverityGrade.PACKED: "packed/numerous",
}

# Friendly organism names for plain-English summaries
_ORGANISM_NAMES = {
    "cocci_small": "small cocci bacteria",
    "cocci_large": "large cocci bacteria",
    "yeast": "yeast organisms",
    "rods": "rod-shaped bacteria",
    "ear_mites": "ear mites",
    "empty_artifact": "artifacts",
}


def compute_severity(
    count: int,
    thresholds: list[int] = None,
) -> SeverityGrade:
    """
    Map a detection count to a severity grade.

    Parameters
    ----------
    count : int
        Number of detections for this organism class.
    thresholds : list of int
        Four threshold values [1+, 2+, 3+, 4+].
        Count >= thresholds[3] → 4+
        Count >= thresholds[2] → 3+
        Count >= thresholds[1] → 2+
        Count >= thresholds[0] → 1+
        Count == 0 → 0

    Returns
    -------
    SeverityGrade
    """
    if thresholds is None:
        thresholds = [1, 5, 15, 30]

    if count <= 0:
        return SeverityGrade.NONE
    elif count >= thresholds[3]:
        return SeverityGrade.PACKED
    elif count >= thresholds[2]:
        return SeverityGrade.MANY
    elif count >= thresholds[1]:
        return SeverityGrade.MODERATE
    elif count >= thresholds[0]:
        return SeverityGrade.RARE
    else:
        return SeverityGrade.NONE


def compute_all_severities(
    organism_counts: dict[str, int],
    severity_thresholds: dict[str, list[int]] = None,
) -> dict[str, SeverityGrade]:
    """
    Compute severity grades for all organism classes.

    Parameters
    ----------
    organism_counts : dict
        {class_name: count}.
    severity_thresholds : dict
        {class_name: [t1, t2, t3, t4]} thresholds per class.
        Falls back to "default" key if class not found.

    Returns
    -------
    dict
        {class_name: SeverityGrade}
    """
    if severity_thresholds is None:
        severity_thresholds = {"default": [1, 5, 15, 30]}

    default_thresholds = severity_thresholds.get("default", [1, 5, 15, 30])

    result = {}
    for cls, count in organism_counts.items():
        thresholds = severity_thresholds.get(cls, default_thresholds)
        result[cls] = compute_severity(count, thresholds)

    return result


def get_overall_severity(grades: dict[str, SeverityGrade]) -> SeverityGrade:
    """
    Get the highest severity grade across all organism classes.
    Ignores empty_artifact.
    """
    grade_order = [
        SeverityGrade.NONE,
        SeverityGrade.RARE,
        SeverityGrade.MODERATE,
        SeverityGrade.MANY,
        SeverityGrade.PACKED,
    ]

    highest = SeverityGrade.NONE
    for cls, grade in grades.items():
        if cls == "empty_artifact":
            continue
        if grade_order.index(grade) > grade_order.index(highest):
            highest = grade

    return highest


def generate_summary(
    organism_counts: dict[str, int],
    severity_grades: dict[str, SeverityGrade],
    overall_severity: SeverityGrade,
) -> str:
    """
    Generate a plain-English summary of the findings for the PDF report.

    Examples:
        "Moderate yeast infection (2+) with occasional cocci bacteria (1+)."
        "No organisms detected."
        "Packed yeast organisms (4+) with many rod-shaped bacteria (3+).
         Ear mites detected (1+) — recommend further evaluation."

    Parameters
    ----------
    organism_counts : dict
        {class_name: count}
    severity_grades : dict
        {class_name: SeverityGrade}
    overall_severity : SeverityGrade

    Returns
    -------
    str
        Plain-English summary paragraph.
    """
    if not organism_counts or overall_severity == SeverityGrade.NONE:
        return "No organisms detected on this slide."

    # Filter to organisms actually found (exclude artifacts and zero counts)
    found = {
        cls: (count, severity_grades.get(cls, SeverityGrade.NONE))
        for cls, count in organism_counts.items()
        if count > 0 and cls != "empty_artifact"
    }

    if not found:
        return "No organisms detected on this slide."

    # Sort by severity (highest first)
    grade_order = [
        SeverityGrade.NONE, SeverityGrade.RARE, SeverityGrade.MODERATE,
        SeverityGrade.MANY, SeverityGrade.PACKED,
    ]
    sorted_findings = sorted(
        found.items(),
        key=lambda x: grade_order.index(x[1][1]),
        reverse=True,
    )

    # Build sentence fragments
    fragments = []
    for cls, (count, grade) in sorted_findings:
        name = _ORGANISM_NAMES.get(cls, cls.replace("_", " "))
        grade_desc = _GRADE_NAMES.get(grade, str(grade.value))

        if grade == SeverityGrade.PACKED:
            fragments.append(f"packed {name} ({grade.value})")
        elif grade == SeverityGrade.MANY:
            fragments.append(f"many {name} ({grade.value})")
        elif grade == SeverityGrade.MODERATE:
            fragments.append(f"moderate {name} ({grade.value})")
        elif grade == SeverityGrade.RARE:
            fragments.append(f"occasional {name} ({grade.value})")

    if not fragments:
        return "No significant organisms detected on this slide."

    # Construct the summary sentence
    if len(fragments) == 1:
        summary = f"{fragments[0].capitalize()} detected."
    elif len(fragments) == 2:
        summary = f"{fragments[0].capitalize()} with {fragments[1]} detected."
    else:
        all_but_last = ", ".join(fragments[:-1])
        summary = f"{all_but_last.capitalize()}, and {fragments[-1]} detected."

    # Add ear mites warning if present
    if "ear_mites" in found and found["ear_mites"][1] != SeverityGrade.NONE:
        summary += " Ear mites detected — recommend further evaluation."

    return summary
