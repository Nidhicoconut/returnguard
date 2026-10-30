"""Case reports: a cited, deterministic explanation of a machine decision. No LLM writes any of it.

* Every statement about the photos (kinds "image" and "finding") must carry EvidenceRefs that resolve
  to real, successful, image-bound tool calls in the ledger. Statements that fail this are dropped and
  a warning is recorded, so a report can never assert something the evidence does not back.
* The customer message is templated from the decision only: it never reveals internal reasons, never
  accuses (no "duplicate", "fraud", "contradicted"), and never promises an outcome that a human owns.
"""
import json
import re
from typing import Optional, Sequence

from agentic.config import AgenticConfig
from agentic.judge.validation import _DECISION_WORDS, ref_is_valid
from agentic.models import (
    CITED_KINDS, CaseInput, CaseReport, EvidenceBundle, JudgeReport, ReportStatement,
)
from agentic.schemas import Decision, DecisionResult, EvidenceRef, ToolCall

# staff-facing explanation of each reason code
REASON_TEXT = {
    "TOOL_FAILURE": "One or more analysis tools failed, so the evidence is incomplete.",
    "ORDER_NOT_FOUND": "No matching order was found.",
    "NON_RETURNABLE_CATEGORY": "The product category is not eligible for automated defect decisions.",
    "OUT_OF_WINDOW": "The return window has passed; a person must decide on any exception.",
    "HIGH_VALUE": "The order value is above the automatic-approval limit.",
    "INTEGRITY_FLAG": "A photo was flagged as a duplicate or has suspicious metadata.",
    "PRODUCT_MISMATCH": "A photo does not appear to show the ordered product.",
    "CLAIM_UNCLEAR": "The claim does not name exactly one defect type.",
    "UNTESTABLE_CLAIM": "The vision model cannot test this defect type for this product category.",
    "QUALITY_UNKNOWN": "Photo quality could not be assessed.",
    "JUDGE_UNAVAILABLE": "The consistency check was unavailable.",
    "JUDGE_DOWNGRADED": "The consistency check produced an unreliable verdict that was overridden.",
    "CLAIM_CONTRADICTED": "The photos appear inconsistent with the claim.",
    "LOCATION_MISMATCH": "The defect was found, but not where the customer said.",
    "UNMENTIONED_DEFECT": "The photos show damage the customer did not mention.",
    "NO_PHOTO": "No photo was provided.",
    "LOW_QUALITY": "No photo was clear enough to analyse.",
    "INSUFFICIENT_EVIDENCE": "The evidence is not strong enough to decide.",
    "EVIDENCE_EXHAUSTED": "The maximum number of evidence requests was reached.",
    "PRODUCT_UNVERIFIED": "The product in the photo could not be verified.",
    "REGION_UNCONFIRMED": "The detected defect was not independently confirmed by a region check.",
    "LOW_DETECTION_SCORE": "The defect detection confidence is below the auto-approval threshold.",
    "AUTO_DISABLED": "Automatic approval is switched off; routed to a person.",
    "AUDIT_SAMPLE": "Randomly selected for a human quality spot check.",
    "VERIFIED_DEFECT": "The claimed defect was detected, confirmed, and all policy checks passed.",
    "DEFAULT_DENY": "The case did not meet every condition for automatic approval.",
    "SYSTEM_ERROR": "The workflow hit an internal error; a person must handle this case.",
}

_CUSTOMER = {
    "approved": "Good news: your return request has been approved. We will email you the next steps shortly.",
    "no_photo": "To process your request, please upload a clear photo of the item and of the problem you described.",
    "low_quality": ("Your photo was a little unclear. Please upload a sharp, well-lit photo of the whole item, "
                    "plus a close-up of the problem."),
    "insufficient": ("We could not clearly see the issue in your photos. Please upload a close-up of the problem "
                     "in good lighting, plus a photo of the whole item."),
    "review": "Thank you for your request. A member of our team will review it personally and get back to you soon.",
}
FORBIDDEN_CUSTOMER_TERMS = ("duplicate", "fraud", "contradict", "suspicious", "reject", "declin", "auto_approve",
                            "escalat", "mismatch", "integrity", "audit", "score", "detector", "policy")


def customer_message(decision: Decision, reason_codes: Sequence[str]) -> str:
    if decision is Decision.AUTO_APPROVE:
        return _CUSTOMER["approved"]
    if decision is Decision.REQUEST_MORE_EVIDENCE:
        codes = set(reason_codes)
        return _CUSTOMER["no_photo"] if "NO_PHOTO" in codes else _CUSTOMER["low_quality"] if "LOW_QUALITY" in codes \
            else _CUSTOMER["insufficient"]
    return _CUSTOMER["review"]


def _ref(image_id: str, call_id: Optional[str], bbox=None) -> list[EvidenceRef]:
    return [EvidenceRef(image_id=image_id, tool_call_id=call_id, bbox=bbox)] if call_id else []


def _excerpt(text: str, n: int = 200) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _build_statements(case: CaseInput, decision: DecisionResult, bundle: Optional[EvidenceBundle],
                      judge: Optional[JudgeReport], cfg: AgenticConfig, cv_skipped: bool) -> list[ReportStatement]:
    S = ReportStatement
    out: list[ReportStatement] = [S(kind="decision", text=f"Machine decision: {decision.decision.value} "
                                                          f"(confidence {decision.confidence:.2f}).")]
    if bundle is None:
        out.append(S(kind="system", text="No evidence bundle is available (the workflow failed before aggregation)."))
    else:
        p = bundle.claim.parsed
        what = p.defect_type.value if p.defect_type else "no single defect type"
        extras = ", ".join(x for x in (f"location: {p.location}" if p.location else "", f"severity: {p.severity.value}",
                                       f"timing: {p.timing.value}", f"wants: {p.requested_resolution.value}") if x)
        out.append(S(kind="claim", text=f"Customer claims {what} ({extras})."))
        if bundle.claim.claim_text:
            out.append(S(kind="claim", text=f"Customer's own words (unverified, quoted as data): “{_excerpt(bundle.claim.claim_text)}”"))

        o = bundle.order
        if o is None:
            out.append(S(kind="order", text="Order details are unavailable."))
        elif o.status == "not_found":
            out.append(S(kind="order", text=f"No order found for id '{o.order_id}'."))
        elif o.status == "unsupported_category":
            out.append(S(kind="order", text=f"Order {o.order_id}: category '{o.raw_category}' is not eligible for automated decisions "
                                            f"(price ₹{o.price_inr:,.0f})."))
        else:
            w = "within" if o.order.window_ok else "outside"
            out.append(S(kind="order", text=f"Order {o.order_id}: {o.order.category.value}, ₹{o.price_inr:,.0f}, delivered "
                                            f"{o.delivered_date} ({o.days_since_delivery} days ago), {w} the {cfg.return_window_days}-day window."))
        if cv_skipped:
            out.append(S(kind="system", text="Photo analysis was skipped because the order is ineligible for automatic processing; "
                                             "the original photos are attached to the review task."))

        defect_images = {d.image_id for d in bundle.defects}
        for im in bundle.images:
            if im.usable is None:
                out.append(S(kind="system", text=f"Image {im.image_id}: quality was not assessed."))
            else:
                out.append(S(kind="image", text=f"Image {im.image_id}: {'usable' if im.usable else 'NOT usable (blur, exposure, resolution or product not visible)'}.",
                             refs=_ref(im.image_id, im.quality_call_id)))
            if im.product_match is not None:
                out.append(S(kind="image", text=f"Image {im.image_id}: {'matches' if im.product_match else 'does NOT match'} the ordered product.",
                             refs=_ref(im.image_id, im.product_call_id)))
            if im.duplicate_of or im.exif_flags:
                bits = ([f"duplicate of {im.duplicate_of}"] if im.duplicate_of else []) + [f"metadata flag: {f}" for f in im.exif_flags]
                out.append(S(kind="image", text=f"Image {im.image_id}: " + "; ".join(bits) + ".", refs=_ref(im.image_id, im.integrity_call_id)))
            if im.detector_ok and im.image_id not in defect_images:
                out.append(S(kind="image", text=f"Image {im.image_id}: no defects detected.", refs=_ref(im.image_id, im.detector_call_id)))
            for role, why in sorted(im.skipped.items()):
                out.append(S(kind="system", text=f"Image {im.image_id}: {role} skipped ({why})."))
        for d in bundle.defects:
            where = f" on the {d.part_location}" if d.part_location else ""
            area = f", ~{d.area_pct}% of the image" if d.area_pct is not None else ""
            region = ""
            if d.region_confirms is not None:
                region = f"; region check {'confirms' if d.region_confirms else 'does not confirm'} ({d.region_confidence:.0%})"
            out.append(S(kind="image", text=f"Image {d.image_id}: {d.defect_class.value} detected{where}, score {d.score:.0%}, "
                                            f"status {d.status}{area}, severity ~{d.severity_estimate.value}{region}.", refs=d.refs()))
        for e in bundle.tool_errors:
            out.append(S(kind="system", text=f"Tool error in {e.tool} ({e.call_id}){' on ' + e.image_id if e.image_id else ''}: {_excerpt(e.error, 120)}"))

    if judge is not None:
        rationale = _DECISION_WORDS.sub("[redacted]", _excerpt(judge.result.rationale, 400))
        out.append(S(kind="verdict", text=f"Consistency check ({judge.source}): {judge.result.verdict.value}. {rationale}"))
        for f in judge.result.findings:
            out.append(S(kind="finding", text=_DECISION_WORDS.sub("[redacted]", f.text), refs=list(f.refs)))
        for w in judge.warnings:
            out.append(S(kind="system", text=f"Judge note: {_excerpt(w, 200)}"))
    for code in decision.reason_codes:
        out.append(S(kind="decision", text=f"{code}: {REASON_TEXT.get(code, 'No description.')}"))
    if bundle is not None:
        by_id = {c.clause_id: c for c in bundle.clauses}
        for cid in decision.clause_ids:
            if cid in by_id:
                out.append(S(kind="policy", text=_excerpt(f"{cid} — {by_id[cid].text}", 260)))
    return out


def validate_report(report: CaseReport, ledger: Sequence[ToolCall]) -> list[str]:
    """Problems found (empty list = valid). Cited statements need >=1 ref and every ref must resolve."""
    by_id = {c.call_id: c for c in ledger}
    problems = []
    for i, s in enumerate(report.statements):
        if s.kind in CITED_KINDS:
            if not s.refs:
                problems.append(f"statement {i} ({s.kind}) has no evidence refs")
            elif not all(ref_is_valid(r, by_id) for r in s.refs):
                problems.append(f"statement {i} ({s.kind}) has an unresolvable evidence ref")
    low = report.customer_message.lower()
    problems += [f"customer message contains forbidden term {t!r}" for t in FORBIDDEN_CUSTOMER_TERMS if t in low]
    return problems


def build_report(*, case: CaseInput, decision: DecisionResult, bundle: Optional[EvidenceBundle], judge: Optional[JudgeReport],
                 now: str, config: Optional[AgenticConfig] = None, evidence_round: int = 0, warnings: Sequence[str] = (),
                 cv_skipped: bool = False) -> CaseReport:
    cfg = config or AgenticConfig()
    ledger = bundle.ledger if bundle else []
    by_id = {c.call_id: c for c in ledger}
    kept, warns = [], list(warnings)
    for s in _build_statements(case, decision, bundle, judge, cfg, cv_skipped):
        if s.kind in CITED_KINDS and not (s.refs and all(ref_is_valid(r, by_id) for r in s.refs)):
            warns.append(f"dropped uncited {s.kind} statement: {_excerpt(s.text, 80)}")
            continue
        kept.append(s)
    report = CaseReport(
        case_id=case.case_id, generated_at=now, decision=decision.decision, reason_codes=list(decision.reason_codes),
        clause_ids=list(decision.clause_ids), evidence_round=evidence_round,
        judge_verdict=judge.result.verdict.value if judge else None, judge_source=judge.source if judge else None,
        statements=kept, customer_message=customer_message(decision.decision, decision.reason_codes), warnings=warns,
    )
    return report


# ----------------------------------------------------------------------------- rendering
def _cite(r: EvidenceRef) -> str:
    box = f"@({r.bbox.x:.2f},{r.bbox.y:.2f},{r.bbox.w:.2f},{r.bbox.h:.2f})" if r.bbox else ""
    return f"[{r.image_id}·{r.tool_call_id}{box}]"


_SECTIONS = [("decision", "Decision"), ("claim", "Claim"), ("order", "Order"), ("image", "Photo evidence"),
             ("finding", "Consistency findings"), ("verdict", "Consistency verdict"), ("policy", "Policy"), ("system", "System notes")]


def render_markdown(report: CaseReport) -> str:
    lines = [f"# Case {report.case_id}: {report.decision.value}", "",
             f"*Generated {report.generated_at} · evidence round {report.evidence_round}*", "",
             f"**Reason codes:** {', '.join(report.reason_codes) or '—'}  ", f"**Policy clauses:** {', '.join(report.clause_ids) or '—'}", ""]
    for kind, title in _SECTIONS:
        rows = [s for s in report.statements if s.kind == kind]
        if not rows:
            continue
        lines += [f"## {title}", ""]
        for s in rows:
            lines.append(f"- {s.text}" + (" " + " ".join(_cite(r) for r in s.refs) if s.refs else ""))
        lines.append("")
    if report.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in report.warnings] + [""]
    lines += ["## Message to customer", "", f"> {report.customer_message}", ""]
    return "\n".join(lines)


def render_json(report: CaseReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
