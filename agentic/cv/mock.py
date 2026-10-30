"""Deterministic mock CV tools. No model, no image decoding, no network, no randomness.

Which scenario an image gets is decided by (in order):
  1. an explicit ``{image_id: MockScenario}`` mapping,
  2. a scenario name as a whole token in the file name  (``case7__hole.jpg`` -> HOLE),
  3. the default scenario (CLEAN).

SEMANTICS the mocks assume where ``schemas.py`` is silent (please confirm with the real model):
  * ``QualityResult.blur``     : blur amount in 0-1, HIGHER = BLURRIER
  * ``QualityResult.exposure`` : 0-1, 0.5 = well exposed
  * ``Segmentation.area_pct``  : percent of the IMAGE covered by the region (0-100)
"""
import re
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence

from agentic.cv.interfaces import CVToolbox
from agentic.models import ImageInput
from agentic.schemas import (
    BBox,
    DefectClass,
    Detection,
    IntegrityResult,
    ProductMatch,
    QualityResult,
    RegionCheck,
    Segmentation,
)

OTHER_SKU = "SKU-OTHER-1"


class MockScenario(str, Enum):
    CLEAN = "clean"
    STAIN_ON_SLEEVE = "stain_on_sleeve"
    HOLE = "hole"
    BLURRY = "blurry"
    WRONG_PRODUCT = "wrong_product"
    DUPLICATE_IMAGE = "duplicate_image"
    DETECTOR_CRASH = "detector_crash"


# scenario -> (defect, bbox, detector score, part location)
_DEFECTS = {
    MockScenario.STAIN_ON_SLEEVE: (DefectClass.stain_discoloration, BBox(x=0.12, y=0.25, w=0.18, h=0.15), 0.87, "sleeve"),
    MockScenario.HOLE: (DefectClass.tear_hole, BBox(x=0.40, y=0.55, w=0.08, h=0.08), 0.91, "front panel"),
    MockScenario.BLURRY: (DefectClass.stain_discoloration, BBox(x=0.12, y=0.25, w=0.18, h=0.15), 0.31, "sleeve"),
    MockScenario.WRONG_PRODUCT: (DefectClass.tear_hole, BBox(x=0.40, y=0.55, w=0.08, h=0.08), 0.91, "front panel"),
    MockScenario.DUPLICATE_IMAGE: (DefectClass.stain_discoloration, BBox(x=0.12, y=0.25, w=0.18, h=0.15), 0.87, "sleeve"),
}


class ScenarioResolver:
    def __init__(self, by_image: Optional[Mapping[str, MockScenario]] = None, default: MockScenario = MockScenario.CLEAN):
        self.by_image = dict(by_image or {})
        self.default = default

    def resolve(self, image: ImageInput) -> MockScenario:
        if image.image_id in self.by_image:
            return MockScenario(self.by_image[image.image_id])
        stem = Path(image.path).stem.lower()
        for sc in sorted(MockScenario, key=lambda s: -len(s.value)):  # longest name first
            if re.search(rf"(?<![a-z0-9]){re.escape(sc.value)}(?![a-z0-9])", stem):
                return sc
        return self.default


class MockQualityChecker:
    def __init__(self, r: ScenarioResolver):
        self._r = r

    def check(self, image: ImageInput) -> QualityResult:
        if self._r.resolve(image) is MockScenario.BLURRY:
            return QualityResult(image_id=image.image_id, blur=0.82, exposure=0.48, resolution_ok=True,
                                 product_visible=True, ok=False)
        return QualityResult(image_id=image.image_id, blur=0.08, exposure=0.50, resolution_ok=True,
                             product_visible=True, ok=True)


class MockIntegrityChecker:
    def __init__(self, r: ScenarioResolver):
        self._r = r

    def check(self, image: ImageInput, case_images: Sequence[ImageInput]) -> IntegrityResult:
        if self._r.resolve(image) is MockScenario.DUPLICATE_IMAGE:
            return IntegrityResult(image_id=image.image_id, duplicate_of="earlier_case_image_001", similarity=0.98)
        return IntegrityResult(image_id=image.image_id, duplicate_of=None, similarity=0.05, exif_flags=[])


class MockProductMatcher:
    def __init__(self, r: ScenarioResolver):
        self._r = r

    def match(self, image: ImageInput, expected_sku: str) -> ProductMatch:
        if self._r.resolve(image) is MockScenario.WRONG_PRODUCT:
            return ProductMatch(image_id=image.image_id, top_k=[(OTHER_SKU, 0.88), (expected_sku, 0.31)], match=False)
        return ProductMatch(image_id=image.image_id, top_k=[(expected_sku, 0.91), (OTHER_SKU, 0.42)], match=True)


class MockDefectDetector:
    def __init__(self, r: ScenarioResolver):
        self._r = r

    def detect(self, image: ImageInput) -> list[Detection]:
        sc = self._r.resolve(image)
        if sc is MockScenario.DETECTOR_CRASH:
            raise RuntimeError("mock detector crashed")
        if sc not in _DEFECTS:
            return []
        defect, box, score, _ = _DEFECTS[sc]
        return [Detection(defect_class=defect, bbox=box, score=score, source="detector")]


class MockSegmenter:
    def __init__(self, r: ScenarioResolver):
        self._r = r

    def segment(self, image: ImageInput, bbox: BBox) -> Segmentation:
        part = _DEFECTS.get(self._r.resolve(image), (None, None, None, ""))[3]
        return Segmentation(area_pct=round(bbox.w * bbox.h * 100, 2), part_location=part)


class MockRegionVerifier:
    def __init__(self, r: ScenarioResolver):
        self._r = r

    def verify(self, image: ImageInput, bbox: BBox, defect_class: DefectClass) -> RegionCheck:
        sc = self._r.resolve(image)
        scenario_defect = _DEFECTS.get(sc, (None,))[0]
        if sc is MockScenario.BLURRY:
            return RegionCheck(image_id=image.image_id, bbox=bbox, description="Blind description: region too blurry to tell.",
                               confirms=False, confidence=0.40)
        if scenario_defect is not None and scenario_defect == defect_class:
            return RegionCheck(image_id=image.image_id, bbox=bbox,
                               description=f"Blind description: visible {defect_class.value.replace('_', ' ')} in the marked region.",
                               confirms=True, confidence=0.85)
        return RegionCheck(image_id=image.image_id, bbox=bbox, description="Blind description: fabric looks intact in the marked region.",
                           confirms=False, confidence=0.70)


def build_mock_toolbox(by_image: Optional[Mapping[str, MockScenario]] = None,
                       default: MockScenario = MockScenario.CLEAN) -> CVToolbox:
    r = ScenarioResolver(by_image, default)
    return CVToolbox(
        detector=MockDefectDetector(r), quality=MockQualityChecker(r), integrity=MockIntegrityChecker(r),
        product_matcher=MockProductMatcher(r), segmenter=MockSegmenter(r), region_verifier=MockRegionVerifier(r),
        name="mock",
    )
