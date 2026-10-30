"""Contracts the CV layer must satisfy. The workflow depends ONLY on these Protocols.

To plug in a real model, implement any of them (``DefectDetector`` is the minimum) and return
objects from ``agentic.schemas``. Conventions every implementation must follow:

* bounding boxes are normalised 0-1 (x, y, w, h) with the origin at the top-left
* a tool may raise; the ToolRunner records the failure and the workflow degrades safely
* returned ``image_id`` (when the schema has one) must equal the ``image.image_id`` passed in
* "no defect found" is an EMPTY list, not an error
"""
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

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


@runtime_checkable
class QualityChecker(Protocol):  # T1
    def check(self, image: ImageInput) -> QualityResult: ...


@runtime_checkable
class IntegrityChecker(Protocol):  # T7
    def check(self, image: ImageInput, case_images: Sequence[ImageInput]) -> IntegrityResult: ...


@runtime_checkable
class ProductMatcher(Protocol):  # T3
    def match(self, image: ImageInput, expected_sku: str) -> ProductMatch: ...


@runtime_checkable
class DefectDetector(Protocol):  # T4
    def detect(self, image: ImageInput) -> list[Detection]: ...


@runtime_checkable
class Segmenter(Protocol):  # T5
    def segment(self, image: ImageInput, bbox: BBox) -> Segmentation: ...


@runtime_checkable
class RegionVerifier(Protocol):  # T9
    def verify(self, image: ImageInput, bbox: BBox, defect_class: DefectClass) -> RegionCheck: ...


@dataclass
class CVToolbox:
    """Bundle of CV tools. Every slot is optional except the detector: a missing tool is recorded
    as "skipped" in the report (never silently assumed to have passed)."""
    detector: Optional[DefectDetector] = None
    quality: Optional[QualityChecker] = None
    integrity: Optional[IntegrityChecker] = None
    product_matcher: Optional[ProductMatcher] = None
    segmenter: Optional[Segmenter] = None
    region_verifier: Optional[RegionVerifier] = None
    name: str = "custom"

    def available(self) -> list[str]:
        return [k for k in ("detector", "quality", "integrity", "product_matcher", "segmenter", "region_verifier")
                if getattr(self, k) is not None]
