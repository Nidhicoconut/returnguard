"""Contracts shared by every part of the agent. Change these and everything downstream feels it.

Rules baked in:
* bounding boxes are normalised 0-1 (x, y, w, h) so any image size works
* the LLM may only produce findings and verdicts; the Decision enum is produced by the rule engine
* every finding carries an EvidenceRef so reports can cite it (FR-8)
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ----------------------------------------------------------------- enums
class Category(str, Enum):
    apparel_footwear = "apparel_footwear"
    electronics = "electronics"
    drinkware_kitchen = "drinkware_kitchen"


class DefectClass(str, Enum):
    crack_break = "crack_break"
    dent_deformation = "dent_deformation"
    scratch_scuff = "scratch_scuff"
    tear_hole = "tear_hole"
    stain_discoloration = "stain_discoloration"
    missing_part = "missing_part"


class Severity(str, Enum):
    minor = "minor"
    moderate = "moderate"
    severe = "severe"
    unknown = "unknown"


class Timing(str, Enum):
    on_arrival = "on_arrival"
    after_use = "after_use"
    unknown = "unknown"


class Resolution(str, Enum):
    refund = "refund"
    replacement = "replacement"
    unspecified = "unspecified"


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class Decision(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    REQUEST_MORE_EVIDENCE = "REQUEST_MORE_EVIDENCE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"  # note: there is deliberately NO reject value


# ----------------------------------------------------------------- geometry + evidence
class BBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside_image(self):
        if self.x + self.w > 1.0001 or self.y + self.h > 1.0001:
            raise ValueError("box extends outside the image")
        return self


class EvidenceRef(BaseModel):
    """Pointer a report sentence can cite: which image, which region, which tool call."""
    image_id: str
    tool_call_id: str
    bbox: Optional[BBox] = None


# ----------------------------------------------------------------- FR-2: parsed claim
class ParsedClaim(BaseModel):
    defect_type: Optional[DefectClass] = None  # None = customer did not say / unclear
    location: str = ""                          # e.g. "hinge of charging case"
    severity: Severity = Severity.unknown
    timing: Timing = Timing.unknown
    requested_resolution: Resolution = Resolution.unspecified


# ----------------------------------------------------------------- tool outputs (T0-T9)
class OrderDetails(BaseModel):  # T0
    order_id: str
    sku: str
    category: Category
    price_inr: float
    delivered_date: str  # ISO date
    window_ok: bool


class QualityResult(BaseModel):  # T1
    image_id: str
    blur: float
    exposure: float
    resolution_ok: bool
    product_visible: bool
    ok: bool


class ProductMatch(BaseModel):  # T3
    image_id: str
    top_k: list[tuple[str, float]]  # [(sku, cosine)]
    ocr_text: str = ""
    match: bool


class Detection(BaseModel):  # T4
    defect_class: DefectClass
    bbox: BBox
    score: float = Field(ge=0, le=1)
    source: str = "detector"  # detector | open_vocab


class Segmentation(BaseModel):  # T5
    area_pct: float
    part_location: str = ""


class IntegrityResult(BaseModel):  # T7
    image_id: str
    duplicate_of: Optional[str] = None
    similarity: float = 0.0
    exif_flags: list[str] = []


class RegionCheck(BaseModel):  # T9 (blind description, then targeted question)
    image_id: str
    bbox: BBox
    description: str
    confirms: bool
    confidence: float = Field(ge=0, le=1)


class PolicyClause(BaseModel):  # T8
    clause_id: str
    text: str
    covers_defects: list[DefectClass] = []


# ----------------------------------------------------------------- audit trail
class ToolCall(BaseModel):
    call_id: str
    step_no: int
    tool: str
    args: dict
    result: dict
    latency_ms: int = 0
    error: Optional[str] = None


# ----------------------------------------------------------------- findings, verdict, decision
class Finding(BaseModel):
    text: str
    refs: list[EvidenceRef] = Field(min_length=1)  # a finding with no evidence is not allowed


class ConsistencyResult(BaseModel):  # FR-5, produced by the LLM judge
    verdict: Verdict
    rationale: str
    findings: list[Finding] = []


class DecisionResult(BaseModel):  # FR-7, produced ONLY by the rule engine
    decision: Decision
    reason_codes: list[str]
    confidence: float = Field(ge=0, le=1)
    clause_ids: list[str] = []
