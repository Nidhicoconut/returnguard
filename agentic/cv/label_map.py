"""Map raw detector labels (Roboflow ``defect-clothes`` style) onto the shared ``DefectClass`` enum.

ASSUMPTION (unverified for defect-clothes-v2): the dataset's classes are ``hole, stain, clothes, cut``.
Roboflow blocked automated access to the dataset page, so confirm the class list with whoever trains
the model and edit ``RAW_TO_DEFECT`` / ``GARMENT_LABELS`` if it differs.

* hole, cut   -> tear_hole
* stain       -> stain_discoloration
* clothes     -> NOT a defect: a garment locator box (used for part location / product visibility)
"""
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from agentic.schemas import BBox, DefectClass, Detection

RAW_TO_DEFECT: dict[str, DefectClass] = {
    "hole": DefectClass.tear_hole,
    "cut": DefectClass.tear_hole,
    "stain": DefectClass.stain_discoloration,
}
GARMENT_LABELS = frozenset({"clothes"})

# What a model trained on this dataset can possibly detect. A claim outside this set on apparel is
# UNTESTABLE (-> insufficient evidence), never "contradicted".
DATASET_DETECTABLE = frozenset(RAW_TO_DEFECT.values())


def normalize_label(raw: str) -> str:
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def map_label(raw: str) -> DefectClass | None:
    """DefectClass for a raw label, or None if it is a garment label or unknown."""
    return RAW_TO_DEFECT.get(normalize_label(raw))


def is_garment_label(raw: str) -> bool:
    return normalize_label(raw) in GARMENT_LABELS


def is_detectable(defect: DefectClass) -> bool:
    return defect in DATASET_DETECTABLE


@dataclass
class MappedPredictions:
    detections: list[Detection] = field(default_factory=list)
    garment_boxes: list[BBox] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)   # raw labels we did not recognise
    dropped: int = 0                                     # boxes discarded as degenerate / malformed


def _to_bbox(cx: float, cy: float, w: float, h: float, img_w: float, img_h: float) -> BBox | None:
    """Pixel centre-format box -> clamped, normalised BBox (None if degenerate)."""
    x0 = max(0.0, (cx - w / 2) / img_w)
    y0 = max(0.0, (cy - h / 2) / img_h)
    x1 = min(1.0, (cx + w / 2) / img_w)
    y1 = min(1.0, (cy + h / 2) / img_h)
    if x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6:
        return None
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def roboflow_to_detections(predictions: Iterable[Mapping], width: int, height: int) -> MappedPredictions:
    """Convert Roboflow object-detection predictions to schema objects.

    Each prediction is ``{"class": str, "confidence": float, "x": cx_px, "y": cy_px, "width": px, "height": px}``
    (Roboflow's centre-format pixel boxes). Malformed entries are counted in ``dropped``, never raised.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    out = MappedPredictions()
    for p in predictions:
        try:
            label = p["class"]
            box = _to_bbox(float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"]), width, height)
            score = min(1.0, max(0.0, float(p["confidence"])))
        except (KeyError, TypeError, ValueError):
            out.dropped += 1
            continue
        if box is None:
            out.dropped += 1
            continue
        if is_garment_label(label):
            out.garment_boxes.append(box)
            continue
        defect = map_label(label)
        if defect is None:
            out.unmapped.append(str(label))
            continue
        out.detections.append(Detection(defect_class=defect, bbox=box, score=score, source="detector"))
    return out
