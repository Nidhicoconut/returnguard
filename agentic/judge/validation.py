"""Safety gate between any judge and the rest of the system.

Guarantees after ``run_judge``:
  * every finding cites at least one EvidenceRef that resolves to a real, successful, image-bound
    tool call in the case ledger (and, if it names a box, a box that tool actually saw)
  * SUPPORTED / CONTRADICTED verdicts are consistent with the deterministic features and carry >=1
    valid finding; otherwise they are downgraded to INSUFFICIENT
  * the judge can never fail the pipeline: errors fall back to the rules judge, then to INSUFFICIENT
  * decision words in judge text are redacted (only the policy engine produces decisions)
"""
import re
from typing import Optional

from agentic.cv.runner import ToolRunner
from agentic.evidence import bbox_iou
from agentic.judge.interface import ConsistencyJudge
from agentic.models import EvidenceBundle, JudgeDraft, JudgeReport
from agentic.schemas import BBox, ConsistencyResult, EvidenceRef, Finding, ToolCall, Verdict

JUDGE_TOOL = "J1_consistency_judge"
BBOX_IOU_MIN = 0.8
MAX_FINDINGS, MAX_RATIONALE, MAX_FINDING_TEXT = 10, 1200, 400
_DECISION_WORDS = re.compile(r"AUTO_APPROVE|REQUEST_MORE_EVIDENCE|ESCALATE_TO_HUMAN", re.I)
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean(text: str, limit: int, warnings: list[str]) -> str:
    text = _CTRL.sub(" ", str(text)).strip()
    if _DECISION_WORDS.search(text):
        warnings.append("judge text contained a decision keyword; redacted")
        text = _DECISION_WORDS.sub("[redacted]", text)
    return text[:limit]


def _call_boxes(call: ToolCall) -> list[BBox]:
    raw = []
    if isinstance(call.args.get("bbox"), dict):
        raw.append(call.args["bbox"])
    res = call.result or {}
    if isinstance(res.get("bbox"), dict):
        raw.append(res["bbox"])
    raw += [it["bbox"] for it in res.get("items", []) if isinstance(it, dict) and isinstance(it.get("bbox"), dict)]
    out = []
    for r in raw:
        try:
            out.append(BBox(**r))
        except Exception:  # noqa: BLE001
            pass
    return out


def ref_is_valid(ref: EvidenceRef, ledger: dict[str, ToolCall]) -> bool:
    call = ledger.get(ref.tool_call_id)
    if call is None or call.error is not None:
        return False
    if call.args.get("image_id") != ref.image_id:  # also rejects non-image calls (T0 / T8 / judge)
        return False
    if ref.bbox is not None:
        return any(bbox_iou(ref.bbox, b) >= BBOX_IOU_MIN for b in _call_boxes(call))
    return True


def _verdict_plausible(verdict: Verdict, bundle: EvidenceBundle) -> Optional[str]:
    """Return a reason if the verdict contradicts the deterministic features, else None."""
    f = bundle.features
    if verdict is Verdict.SUPPORTED:
        if not (f.claimed_defect_observed and f.claim_testable and f.n_scanned_usable_images >= 1):
            return "SUPPORTED without an observed, testable, claimed defect on a usable image"
    if verdict is Verdict.CONTRADICTED:
        if f.claimed_defect is None or not f.claim_testable or f.n_scanned_usable_images < 1 or f.claimed_defect_observed:
            return "CONTRADICTED without a usable scanned image, or although the claimed defect was observed"
    return None


def validate_draft(draft: JudgeDraft, bundle: EvidenceBundle) -> tuple[ConsistencyResult, list[str], int, bool]:
    warnings = list(draft.warnings)
    ledger = {c.call_id: c for c in bundle.ledger}
    kept: list[Finding] = []
    dropped = 0
    for fnd in draft.result.findings[:MAX_FINDINGS]:
        good = [r for r in fnd.refs if ref_is_valid(r, ledger)]
        if len(good) != len(fnd.refs):
            warnings.append(f"finding had {len(fnd.refs) - len(good)} unresolvable evidence ref(s)")
        if not good:
            dropped += 1
            continue
        kept.append(Finding(text=_clean(fnd.text, MAX_FINDING_TEXT, warnings), refs=good))
    dropped += max(0, len(draft.result.findings) - MAX_FINDINGS)

    verdict, downgraded = draft.result.verdict, False
    if verdict is not Verdict.INSUFFICIENT:
        reason = _verdict_plausible(verdict, bundle)
        if reason is None and not kept:
            reason = f"{verdict.value} had no valid cited finding"
        if reason:
            warnings.append(f"verdict downgraded to INSUFFICIENT: {reason}")
            verdict, downgraded = Verdict.INSUFFICIENT, True
    rationale = _clean(draft.result.rationale, MAX_RATIONALE, warnings)
    if downgraded:
        rationale = f"[verdict downgraded by validation] {rationale}"[:MAX_RATIONALE]
    return ConsistencyResult(verdict=verdict, rationale=rationale, findings=kept), warnings, dropped, downgraded


def run_judge(bundle: EvidenceBundle, primary: ConsistencyJudge, runner: ToolRunner,
              fallback: Optional[ConsistencyJudge] = None, *, timeout_s: Optional[float] = None,
              max_retries: Optional[int] = None) -> JudgeReport:
    """Run the judge through the ToolRunner (audited, timed, retried), validate, never raise."""
    warnings: list[str] = []

    def attempt(judge: ConsistencyJudge):
        return runner.call(JUDGE_TOOL, lambda: judge.judge(bundle),
                           args={"case_id": bundle.case_id, "judge": getattr(judge, "name", type(judge).__name__)},
                           expect=JudgeDraft, timeout_s=timeout_s, max_retries=max_retries)

    res = attempt(primary)
    source = getattr(primary, "name", "rules")
    if not res.ok and fallback is not None:
        warnings.append(f"primary judge failed ({res.call.error}); used fallback rules judge")
        res, source = attempt(fallback), "fallback"
    if not res.ok:
        warnings.append(f"judge unavailable ({res.call.error})")
        empty = ConsistencyResult(verdict=Verdict.INSUFFICIENT, rationale="The consistency judge was unavailable.")
        return JudgeReport(result=empty, source="none", warnings=warnings, call_id=res.call_id)
    result, w, dropped, downgraded = validate_draft(res.output, bundle)
    return JudgeReport(result=result, source=source, warnings=warnings + w, dropped_findings=dropped,
                       downgraded=downgraded, call_id=res.call_id)
