"""New contracts for the agentic workflow.

These models EXTEND ``agentic/schemas.py``; they never redefine or alter anything in it.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from agentic.schemas import (
    BBox,
    ConsistencyResult,
    DecisionResult,
    DefectClass,
    Detection,
    Decision,
    EvidenceRef,
    IntegrityResult,
    OrderDetails,
    ParsedClaim,
    PolicyClause,
    ProductMatch,
    QualityResult,
    RegionCheck,
    Segmentation,
    Severity,
    ToolCall,
)


# =============================================================== case input (Increment 1)
class ImageInput(BaseModel):
    """One customer photo. ``image_id`` is the stable id every EvidenceRef points at."""
    image_id: str
    path: str  # filesystem path; mock tools never open it, real tools do

    @field_validator("image_id", "path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class CaseInput(BaseModel):
    case_id: str
    order_id: str
    claim_text: str = ""
    images: list[ImageInput] = []

    @model_validator(mode="after")
    def _unique_image_ids(self):
        ids = [i.image_id for i in self.images]
        if len(ids) != len(set(ids)):
            raise ValueError("image_ids must be unique within a case")
        return self


# =============================================================== CV plan output (Increment 1)
class DetectionAnalysis(BaseModel):
    """One detection plus the follow-up tools run on its region, each with its call id."""
    detection: Detection
    detection_call_id: str
    segmentation: Optional[Segmentation] = None
    segmentation_call_id: Optional[str] = None
    region_check: Optional[RegionCheck] = None
    region_call_id: Optional[str] = None


class ImageCVReport(BaseModel):
    """Everything the CV tools said about one image. ``call_ids`` maps tool role -> call id."""
    image_id: str
    quality: Optional[QualityResult] = None
    integrity: Optional[IntegrityResult] = None
    product_match: Optional[ProductMatch] = None
    detections: list[DetectionAnalysis] = []
    call_ids: dict[str, str] = {}
    skipped: dict[str, str] = {}   # tool role -> reason it was not run (not an error)
    errors: dict[str, str] = {}    # tool role -> error message (the call is in the ledger)


class CVCaseReport(BaseModel):
    case_id: str
    images: list[ImageCVReport] = []
    ledger: list[ToolCall] = Field(default_factory=list)  # full ordered audit trail


# =============================================================== claim / order / policy (Increment 2)
class ClaimInfo(BaseModel):
    """Output of the claim parser: the schema's ParsedClaim plus every defect the customer mentioned."""
    parsed: ParsedClaim
    mentioned_defects: list[DefectClass] = []   # all defects named (negated ones excluded), in text order
    claim_text: str = ""                         # (possibly truncated) customer text, kept as DATA only
    truncated: bool = False


class OrderLookupResult(BaseModel):
    """T0 result. ``found`` carries a full OrderDetails; the other statuses are valid outcomes
    (not tool errors) that the policy engine will route to a human."""
    status: Literal["found", "not_found", "unsupported_category"]
    order_id: str = ""
    order: Optional[OrderDetails] = None
    raw_category: str = ""            # the store's own category label (kept for R4 handling)
    price_inr: Optional[float] = None
    delivered_date: str = ""
    days_since_delivery: Optional[int] = None


class PolicyBundle(BaseModel):  # T8
    clauses: list[PolicyClause]


# =============================================================== evidence aggregation (Increment 3)
DefectStatus = Literal["confirmed", "unverified", "disputed", "weak"]


class ObservedDefect(BaseModel):
    """One de-duplicated defect observation with full provenance."""
    image_id: str
    defect_class: DefectClass
    bbox: BBox
    score: float
    status: DefectStatus  # confirmed: T9 agrees | unverified: no T9 | disputed: T9 disagrees | weak: below min score
    part_location: str = ""
    area_pct: Optional[float] = None
    severity_estimate: Severity = Severity.unknown
    detection_call_id: str
    segmentation_call_id: Optional[str] = None
    region_call_id: Optional[str] = None
    region_confirms: Optional[bool] = None
    region_confidence: Optional[float] = None
    merged_from: int = 1  # number of overlapping raw boxes folded into this one

    def refs(self) -> list[EvidenceRef]:
        out = [EvidenceRef(image_id=self.image_id, tool_call_id=self.detection_call_id, bbox=self.bbox)]
        if self.region_call_id:
            out.append(EvidenceRef(image_id=self.image_id, tool_call_id=self.region_call_id, bbox=self.bbox))
        return out


class ImageEvidence(BaseModel):
    image_id: str
    usable: Optional[bool] = None          # None = quality unknown (tool missing / failed)
    quality_call_id: Optional[str] = None
    detector_ok: bool = False              # detector ran successfully on this image
    detector_call_id: Optional[str] = None
    product_match: Optional[bool] = None   # None = not checked
    product_call_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    exif_flags: list[str] = []
    integrity_call_id: Optional[str] = None
    skipped: dict[str, str] = {}


class ToolErrorInfo(BaseModel):
    call_id: str
    tool: str
    error: str
    image_id: Optional[str] = None


class EvidenceFeatures(BaseModel):
    """Deterministic facts derived from the evidence. The policy engine reads these, not LLM prose."""
    n_images: int = 0
    n_usable_images: int = 0
    n_scanned_usable_images: int = 0       # usable AND the detector ran on it
    quality_checked: bool = False
    product_checked: bool = False
    product_mismatch: bool = False
    integrity_flagged: bool = False
    duplicate_images: list[str] = []
    exif_flag_images: list[str] = []
    claimed_defect: Optional[DefectClass] = None
    claim_unspecified: bool = True         # customer did not name exactly one defect type
    claim_testable: bool = False           # model can actually detect this defect in this category
    claimed_defect_observed: bool = False  # confirmed/unverified detection of the claimed class
    claimed_defect_disputed: bool = False  # detected but T9 says it is not there (and nothing else supports it)
    claimed_defect_weak: bool = False      # only a low-score detection of the claimed class
    corroborating_images: int = 0          # distinct images showing the claimed defect
    location_match: Optional[bool] = None  # None = cannot compare
    other_defects: list[DefectClass] = []  # strong defects of a class other than the claimed one
    unmentioned_defects: list[DefectClass] = []  # strong defects the customer never mentioned (R6)
    max_defect_score: float = 0.0
    n_tool_errors: int = 0


class EvidenceBundle(BaseModel):
    case_id: str
    claim: ClaimInfo
    order: Optional[OrderLookupResult] = None
    clauses: list[PolicyClause] = []
    images: list[ImageEvidence] = []
    defects: list[ObservedDefect] = []
    features: EvidenceFeatures = Field(default_factory=EvidenceFeatures)
    tool_errors: list[ToolErrorInfo] = []
    ledger: list[ToolCall] = []


# =============================================================== consistency judge (Increment 4)
class JudgeDraft(BaseModel):
    """Raw output of a judge implementation, before validation."""
    result: ConsistencyResult
    warnings: list[str] = []


class JudgeReport(BaseModel):
    """Validated judge output. The verdict is advisory input to the policy engine, never a decision."""
    result: ConsistencyResult
    source: Literal["rules", "llm", "fallback", "none"]
    warnings: list[str] = []
    dropped_findings: int = 0
    downgraded: bool = False   # the judge's verdict was overridden to INSUFFICIENT by validation
    call_id: Optional[str] = None


# =============================================================== reporting (Increment 7)
StatementKind = Literal["claim", "order", "image", "finding", "verdict", "policy", "decision", "system"]
CITED_KINDS = ("image", "finding")  # statements about the photos MUST carry valid evidence refs


class ReportStatement(BaseModel):
    kind: StatementKind
    text: str
    refs: list[EvidenceRef] = []


class CaseReport(BaseModel):
    """Staff-facing explanation of a machine decision. Every statement about the photos is cited."""
    case_id: str
    generated_at: str
    decision: Decision
    reason_codes: list[str]
    clause_ids: list[str] = []
    evidence_round: int = 0
    judge_verdict: Optional[str] = None
    judge_source: Optional[str] = None
    statements: list[ReportStatement] = []
    customer_message: str = ""   # templated, never LLM-written, never reveals internal reasons
    warnings: list[str] = []


# =============================================================== human review (Increment 7)
class ReviewAction(str):
    """Allowed reviewer actions (plain strings so they serialise cleanly)."""
    APPROVE_REFUND = "approve_refund"
    APPROVE_REPLACEMENT = "approve_replacement"
    REQUEST_EVIDENCE = "request_evidence"
    DECLINE = "decline"  # only a human can decline; the system has no reject path
    ALL = ("approve_refund", "approve_replacement", "request_evidence", "decline")


class ReviewTask(BaseModel):
    task_id: str = ""            # assigned by the queue
    case_id: str
    created_at: str
    status: Literal["open", "resolved"] = "open"
    priority: Literal["high", "normal", "low"] = "normal"
    machine_decision: Decision
    reason_codes: list[str]
    clause_ids: list[str] = []
    would_have_auto_approved: bool = False   # True for audit-sample spot checks
    checklist: list[str] = []
    evidence_round: int = 0
    images: list[ImageInput] = []            # so reviewers can open the photos even if CV was skipped
    evidence: Optional[EvidenceBundle] = None
    judge: Optional[JudgeReport] = None
    report: Optional[CaseReport] = None


class ReviewOutcome(BaseModel):
    task_id: str
    case_id: str
    reviewer: str
    action: str
    note: str = ""
    decided_at: str
    machine_decision: Decision
    machine_reason_codes: list[str] = []
    overrides_machine: bool = False  # reviewer did not follow what the machine would have done (audit spot checks)


# =============================================================== workflow result (Increment 6)
class WorkflowResult(BaseModel):
    case_id: str
    decision: DecisionResult
    rounds_used: int = 0
    evidence: Optional[EvidenceBundle] = None
    judge: Optional[JudgeReport] = None
    report: Optional[CaseReport] = None
    review_task: Optional[ReviewTask] = None
    review_enqueued: bool = False
    system_error: Optional[str] = None
    warnings: list[str] = []
    ledger: list[ToolCall] = []   # every tool call from every evidence round, in order
