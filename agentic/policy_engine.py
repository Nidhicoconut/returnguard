"""Deterministic policy engine: the ONLY producer of a ``Decision``.

A pure function of (EvidenceBundle, JudgeReport, config, evidence_round). It reads structured
features and the judge's *verdict enum* only; judge prose, customer text and LLM output can never
change the outcome. Design rules:

* DEFAULT DENY: anything not explicitly cleared for approval escalates to a human.
* There is no reject. Out-of-window, high-value, contradicted, etc. all go to a human.
* All applicable hard-block reasons are collected (so the reviewer sees everything), but the
  decision is the same: ESCALATE_TO_HUMAN.
* REQUEST_MORE_EVIDENCE is used only when a better photo could actually help, and only for
  ``max_evidence_rounds`` rounds; after that the case escalates.
* The kill switch (auto_approve_enabled=False) blocks APPROVAL only; it never hides other outcomes.
* Audit sampling is a deterministic hash of the case id, so it is reproducible and unbiasable.
"""
import hashlib
from enum import Enum
from typing import Optional

from agentic.config import AgenticConfig
from agentic.models import EvidenceBundle, JudgeReport
from agentic.schemas import Decision, DecisionResult, Verdict


class Reason(str, Enum):
    # hard blocks -> ESCALATE
    TOOL_FAILURE = "TOOL_FAILURE"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    NON_RETURNABLE_CATEGORY = "NON_RETURNABLE_CATEGORY"
    OUT_OF_WINDOW = "OUT_OF_WINDOW"
    HIGH_VALUE = "HIGH_VALUE"
    INTEGRITY_FLAG = "INTEGRITY_FLAG"
    PRODUCT_MISMATCH = "PRODUCT_MISMATCH"
    CLAIM_UNCLEAR = "CLAIM_UNCLEAR"
    UNTESTABLE_CLAIM = "UNTESTABLE_CLAIM"
    QUALITY_UNKNOWN = "QUALITY_UNKNOWN"
    JUDGE_UNAVAILABLE = "JUDGE_UNAVAILABLE"
    JUDGE_DOWNGRADED = "JUDGE_DOWNGRADED"
    CLAIM_CONTRADICTED = "CLAIM_CONTRADICTED"
    LOCATION_MISMATCH = "LOCATION_MISMATCH"
    UNMENTIONED_DEFECT = "UNMENTIONED_DEFECT"
    # evidence gaps -> REQUEST_MORE_EVIDENCE (or ESCALATE when rounds are exhausted)
    NO_PHOTO = "NO_PHOTO"
    LOW_QUALITY = "LOW_QUALITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EVIDENCE_EXHAUSTED = "EVIDENCE_EXHAUSTED"
    # approval gate -> ESCALATE
    PRODUCT_UNVERIFIED = "PRODUCT_UNVERIFIED"
    REGION_UNCONFIRMED = "REGION_UNCONFIRMED"
    LOW_DETECTION_SCORE = "LOW_DETECTION_SCORE"
    AUTO_DISABLED = "AUTO_DISABLED"
    AUDIT_SAMPLE = "AUDIT_SAMPLE"
    # outcomes
    VERIFIED_DEFECT = "VERIFIED_DEFECT"
    DEFAULT_DENY = "DEFAULT_DENY"
    SYSTEM_ERROR = "SYSTEM_ERROR"


# which policy clauses (policy/return_policy.md) justify each reason
CLAUSES: dict[Reason, list[str]] = {
    Reason.TOOL_FAILURE: ["R7"], Reason.ORDER_NOT_FOUND: ["R7"], Reason.NON_RETURNABLE_CATEGORY: ["R4"],
    Reason.OUT_OF_WINDOW: ["R1"], Reason.HIGH_VALUE: ["R7"], Reason.INTEGRITY_FLAG: ["R5"],
    Reason.PRODUCT_MISMATCH: ["R5"], Reason.CLAIM_UNCLEAR: ["R5"], Reason.UNTESTABLE_CLAIM: ["R5"],
    Reason.QUALITY_UNKNOWN: ["R5"], Reason.JUDGE_UNAVAILABLE: ["R7"], Reason.JUDGE_DOWNGRADED: ["R7"],
    Reason.CLAIM_CONTRADICTED: ["R6"], Reason.LOCATION_MISMATCH: ["R6"], Reason.UNMENTIONED_DEFECT: ["R6"],
    Reason.NO_PHOTO: ["R5"], Reason.LOW_QUALITY: ["R5"], Reason.INSUFFICIENT_EVIDENCE: ["R5"],
    Reason.EVIDENCE_EXHAUSTED: ["R5"], Reason.PRODUCT_UNVERIFIED: ["R5"], Reason.REGION_UNCONFIRMED: ["R5"],
    Reason.LOW_DETECTION_SCORE: ["R5"], Reason.AUTO_DISABLED: ["R7"], Reason.AUDIT_SAMPLE: ["R7"],
    Reason.VERIFIED_DEFECT: ["R1", "R2"], Reason.DEFAULT_DENY: ["R7"], Reason.SYSTEM_ERROR: ["R7"],
}
_ORDER = list(Reason)  # declaration order doubles as display precedence


def audit_sampled(case_id: str, rate: float) -> bool:
    """Deterministic spot-check sampling: the same case id always gives the same answer."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    h = int.from_bytes(hashlib.sha256(f"audit:{case_id}".encode()).digest()[:8], "big") / 2**64
    return h < rate


def _result(decision: Decision, reasons: list[Reason], confidence: float, bundle: EvidenceBundle) -> DecisionResult:
    reasons = sorted(set(reasons), key=_ORDER.index)
    available = {c.clause_id for c in bundle.clauses}
    clause_ids = sorted({c for r in reasons for c in CLAUSES[r] if not available or c in available})
    return DecisionResult(decision=decision, reason_codes=[r.value for r in reasons],
                          confidence=max(0.0, min(1.0, confidence)), clause_ids=clause_ids)


def system_error_result(detail: str = "") -> DecisionResult:
    """Used when the workflow itself failed: always a human."""
    return DecisionResult(decision=Decision.ESCALATE_TO_HUMAN, reason_codes=[Reason.SYSTEM_ERROR.value],
                          confidence=0.0, clause_ids=["R7"])


def _hard_blocks(b: EvidenceBundle, judge: Optional[JudgeReport], cfg: AgenticConfig) -> set[Reason]:
    f, order, blocks = b.features, b.order, set()
    if f.n_tool_errors > 0:
        blocks.add(Reason.TOOL_FAILURE)

    order_failed = any(e.tool == "T0_order_lookup" for e in b.tool_errors)
    if order is None:
        if not order_failed:
            blocks.add(Reason.ORDER_NOT_FOUND)
    elif order.status == "not_found":
        blocks.add(Reason.ORDER_NOT_FOUND)
    elif order.status == "unsupported_category":
        blocks.add(Reason.NON_RETURNABLE_CATEGORY)
    elif order.order is not None and not order.order.window_ok:
        blocks.add(Reason.OUT_OF_WINDOW)
    if order is not None and order.price_inr is not None and order.price_inr > cfg.high_value_limit_inr:
        blocks.add(Reason.HIGH_VALUE)

    if judge is None or judge.source == "none":
        blocks.add(Reason.JUDGE_UNAVAILABLE)
    elif judge.downgraded:
        blocks.add(Reason.JUDGE_DOWNGRADED)
    if judge is not None and judge.result.verdict is Verdict.CONTRADICTED:
        blocks.add(Reason.CLAIM_CONTRADICTED)

    if f.integrity_flagged:
        blocks.add(Reason.INTEGRITY_FLAG)
    if f.product_mismatch:
        blocks.add(Reason.PRODUCT_MISMATCH)
    if f.claim_unspecified:
        blocks.add(Reason.CLAIM_UNCLEAR)
    elif order is not None and order.status == "found" and not f.claim_testable:
        blocks.add(Reason.UNTESTABLE_CLAIM)
    if f.n_images > 0 and not f.quality_checked and Reason.TOOL_FAILURE not in blocks:
        blocks.add(Reason.QUALITY_UNKNOWN)
    if f.location_match is False:
        blocks.add(Reason.LOCATION_MISMATCH)
    if f.unmentioned_defects:
        blocks.add(Reason.UNMENTIONED_DEFECT)
    return blocks


def decide(bundle: EvidenceBundle, judge: Optional[JudgeReport], *, config: Optional[AgenticConfig] = None,
           evidence_round: int = 0) -> DecisionResult:
    cfg = config or AgenticConfig()
    f = bundle.features

    # 1. hard blocks: a human must look, more photos will not fix it
    blocks = _hard_blocks(bundle, judge, cfg)
    if blocks:
        return _result(Decision.ESCALATE_TO_HUMAN, list(blocks), 1.0, bundle)

    # 2. evidence gaps: a better photo could help
    gaps: list[Reason] = []
    if f.n_images == 0:
        gaps.append(Reason.NO_PHOTO)
    elif f.n_usable_images == 0:
        gaps.append(Reason.LOW_QUALITY)
    if not gaps and judge.result.verdict is Verdict.INSUFFICIENT:
        gaps.append(Reason.INSUFFICIENT_EVIDENCE)
    if gaps:
        if evidence_round < cfg.max_evidence_rounds:
            return _result(Decision.REQUEST_MORE_EVIDENCE, gaps, 0.5, bundle)
        return _result(Decision.ESCALATE_TO_HUMAN, gaps + [Reason.EVIDENCE_EXHAUSTED], 1.0, bundle)

    # 3. approval gate: every condition must hold, else a human decides
    if judge.result.verdict is not Verdict.SUPPORTED or not (
            f.claimed_defect_observed and f.claim_testable and f.n_scanned_usable_images >= 1):
        return _result(Decision.ESCALATE_TO_HUMAN, [Reason.DEFAULT_DENY], 0.0, bundle)

    confirmed = [d for d in bundle.defects if d.defect_class == f.claimed_defect and d.status == "confirmed"]
    best = max(confirmed, key=lambda d: d.score, default=None)
    fails: list[Reason] = []
    if not f.product_checked:
        fails.append(Reason.PRODUCT_UNVERIFIED)
    if best is None:
        fails.append(Reason.REGION_UNCONFIRMED)
    elif best.score < cfg.auto_approve_min_score:
        fails.append(Reason.LOW_DETECTION_SCORE)
    if fails:
        return _result(Decision.ESCALATE_TO_HUMAN, fails, 1.0, bundle)
    if not cfg.auto_approve_enabled:
        return _result(Decision.ESCALATE_TO_HUMAN, [Reason.AUTO_DISABLED], 1.0, bundle)
    if audit_sampled(bundle.case_id, cfg.audit_sample_rate):
        return _result(Decision.ESCALATE_TO_HUMAN, [Reason.AUDIT_SAMPLE], 1.0, bundle)

    conf = min(best.score, best.region_confidence if best.region_confidence is not None else best.score)
    return _result(Decision.AUTO_APPROVE, [Reason.VERIFIED_DEFECT], conf, bundle)
