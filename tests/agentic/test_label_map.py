import pytest

from agentic.cv.label_map import (
    DATASET_DETECTABLE, is_detectable, is_garment_label, map_label, normalize_label, roboflow_to_detections,
)
from agentic.schemas import DefectClass


@pytest.mark.parametrize("raw,expected", [
    ("hole", DefectClass.tear_hole), ("Cut", DefectClass.tear_hole), (" STAIN ", DefectClass.stain_discoloration),
    ("clothes", None), ("button", None),
])
def test_map_label(raw, expected):
    assert map_label(raw) == expected


def test_normalize_and_garment():
    assert normalize_label(" Torn-Hole ") == "torn_hole"
    assert is_garment_label("Clothes") and not is_garment_label("hole")


def test_dataset_cannot_detect_crack_dent_scratch_missing():
    assert DATASET_DETECTABLE == {DefectClass.tear_hole, DefectClass.stain_discoloration}
    for d in (DefectClass.crack_break, DefectClass.dent_deformation, DefectClass.scratch_scuff, DefectClass.missing_part):
        assert not is_detectable(d)


def test_roboflow_conversion_normalises_and_splits():
    preds = [
        {"class": "stain", "confidence": 0.9, "x": 100, "y": 50, "width": 40, "height": 20},
        {"class": "cut", "confidence": 0.7, "x": 10, "y": 10, "width": 40, "height": 40},   # spills outside -> clamped
        {"class": "clothes", "confidence": 0.99, "x": 100, "y": 100, "width": 200, "height": 200},
        {"class": "zipper", "confidence": 0.8, "x": 50, "y": 50, "width": 10, "height": 10},
        {"class": "hole", "confidence": 0.8, "x": 50, "y": 50, "width": 0, "height": 10},   # degenerate
        {"class": "hole", "confidence": 0.8},                                                 # malformed
    ]
    out = roboflow_to_detections(preds, 200, 100)
    assert [d.defect_class for d in out.detections] == [DefectClass.stain_discoloration, DefectClass.tear_hole]
    stain = out.detections[0]
    assert stain.bbox.x == pytest.approx(0.4) and stain.bbox.y == pytest.approx(0.4)
    assert stain.bbox.w == pytest.approx(0.2) and stain.bbox.h == pytest.approx(0.2)
    clamped = out.detections[1].bbox
    assert clamped.x == 0 and clamped.y == 0 and clamped.x + clamped.w <= 1
    assert len(out.garment_boxes) == 1
    assert out.unmapped == ["zipper"] and out.dropped == 2


def test_confidence_is_clamped_and_bad_image_size_rejected():
    out = roboflow_to_detections([{"class": "hole", "confidence": 1.4, "x": 5, "y": 5, "width": 4, "height": 4}], 10, 10)
    assert out.detections[0].score == 1.0
    with pytest.raises(ValueError):
        roboflow_to_detections([], 0, 10)
