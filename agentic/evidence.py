"""Evidence aggregation: turns the raw tool outputs into one EvidenceBundle of deterministic facts.

Pure and offline. No LLM, no randomness. The policy engine reads ``EvidenceBundle.features`` (never
free text), and the judge gets a minimal ``judge_view``. Every observation keeps its tool-call ids so
reports can cite it.
"""
import re
from typing import Optional, Sequence

from agentic.config import AgenticConfig
from agentic.cv.label_map import is_detectable
from agentic.models import (
    CaseInput, ClaimInfo, CVCaseReport, EvidenceBundle, EvidenceFeatures, ImageCVReport, ImageEvidence,
    ObservedDefect, OrderLookupResult, ToolErrorInfo,
)
from agentic.schemas import BBox, DefectClass, PolicyClause, Severity, ToolCall

MERGE_IOU = 0.5
_STRONG = ("confirmed", "unverified", "disputed")      # score >= min (T9 may agree, be absent, or disagree)
_SUPPORTING = ("confirmed", "unverified")              # counts as "the defect is there"
_GENERIC_LOCATIONS = {"clothes", "garment", "item", "product", "apparel", "clothing"}
_LOC_STOP = {"left", "right", "upper", "lower", "inner", "outer", "top", "bottom", "front", "back", "the", "of", "a", "part", "panel", "area"}


# ----------------------------------------------------------------------------- helpers
def bbox_iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _raw_words(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (text or "").lower()) if t not in _LOC_STOP}


def _loc_tokens(text: str) -> set[str]:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in _raw_words(text)}


def location_match(claimed: str, observed: str) -> Optional[bool]:
    """True/False when the two places can be compared, None when either side is empty or generic
    (e.g. the segmenter only said "clothes")."""
    raw_obs = _raw_words(observed)
    if not _raw_words(claimed) or not raw_obs or raw_obs <= _GENERIC_LOCATIONS:
        return None
    return bool(_loc_tokens(claimed) & _loc_tokens(observed))


def _severity(area_pct: Optional[float], cfg: AgenticConfig) -> Severity:
    if area_pct is None:
        return Severity.unknown
    if area_pct <= cfg.severity_minor_max_area_pct:
        return Severity.minor
    if area_pct <= cfg.severity_moderate_max_area_pct:
        return Severity.moderate
    return Severity.severe


# ----------------------------------------------------------------------------- per-image
def _image_evidence(ir: ImageCVReport) -> ImageEvidence:
    q, ig, pm = ir.quality, ir.integrity, ir.product_match
    return ImageEvidence(
        image_id=ir.image_id,
        usable=None if q is None else bool(q.ok and q.product_visible and q.resolution_ok),
        quality_call_id=ir.call_ids.get("quality"),
        detector_ok="detector" in ir.call_ids and "detector" not in ir.errors,
        detector_call_id=ir.call_ids.get("detector"),
        product_match=None if pm is None else pm.match,
        product_call_id=ir.call_ids.get("product_match"),
        duplicate_of=ig.duplicate_of if ig else None,
        exif_flags=list(ig.exif_flags) if ig else [],
        integrity_call_id=ir.call_ids.get("integrity"),
        skipped=dict(ir.skipped),
    )


def _defects(cv: CVCaseReport, cfg: AgenticConfig) -> list[ObservedDefect]:
    raw: list[ObservedDefect] = []
    for ir in cv.images:
        for da in ir.detections:
            det, rc, seg = da.detection, da.region_check, da.segmentation
            if det.score < cfg.detection_min_score:
                status = "weak"
            elif rc is None:
                status = "unverified"
            elif rc.confirms and rc.confidence >= cfg.region_min_confidence:
                status = "confirmed"
            elif rc.confirms:
                status = "unverified"   # T9 agreed but was not sure enough to count as confirmation
            else:
                status = "disputed"
            area = seg.area_pct if seg else None
            raw.append(ObservedDefect(
                image_id=ir.image_id, defect_class=det.defect_class, bbox=det.bbox, score=det.score, status=status,
                part_location=seg.part_location if seg else "", area_pct=area, severity_estimate=_severity(area, cfg),
                detection_call_id=da.detection_call_id, segmentation_call_id=da.segmentation_call_id,
                region_call_id=da.region_call_id, region_confirms=rc.confirms if rc else None,
                region_confidence=rc.confidence if rc else None,
            ))
    # fold overlapping boxes of the same class on the same image into the best-scoring one
    kept: list[ObservedDefect] = []
    for d in sorted(raw, key=lambda x: -x.score):
        twin = next((k for k in kept if k.image_id == d.image_id and k.defect_class == d.defect_class
                     and bbox_iou(k.bbox, d.bbox) >= MERGE_IOU), None)
        if twin:
            twin.merged_from += 1
        else:
            kept.append(d)
    return kept


# ----------------------------------------------------------------------------- features
def _features(claim: ClaimInfo, order: Optional[OrderLookupResult], images: list[ImageEvidence],
              defects: list[ObservedDefect], errors: list[ToolErrorInfo], cfg: AgenticConfig) -> EvidenceFeatures:
    claimed = claim.parsed.defect_type
    category = order.order.category.value if order and order.status == "found" and order.order else None
    strong = [d for d in defects if d.status in _STRONG]
    support = [d for d in defects if d.status in _SUPPORTING]
    claimed_obs = [d for d in support if d.defect_class == claimed] if claimed else []
    claimed_disp = [d for d in strong if d.defect_class == claimed and d.status == "disputed"] if claimed else []
    claimed_weak = [d for d in defects if d.defect_class == claimed and d.status == "weak"] if claimed else []

    comparable = [location_match(claim.parsed.location, d.part_location) for d in claimed_obs]
    comparable = [c for c in comparable if c is not None]
    loc = None if not comparable else any(comparable)

    other = sorted({d.defect_class for d in support if d.defect_class != claimed}, key=lambda c: c.value)
    unmentioned = (sorted({d.defect_class for d in support if d.defect_class not in claim.mentioned_defects},
                          key=lambda c: c.value) if claim.mentioned_defects else [])

    return EvidenceFeatures(
        n_images=len(images),
        n_usable_images=sum(1 for i in images if i.usable),
        n_scanned_usable_images=sum(1 for i in images if i.usable and i.detector_ok),
        quality_checked=any(i.usable is not None for i in images),
        product_checked=any(i.product_match is not None for i in images),
        product_mismatch=any(i.product_match is False for i in images),
        integrity_flagged=any(i.duplicate_of or i.exif_flags for i in images),
        duplicate_images=[i.image_id for i in images if i.duplicate_of],
        exif_flag_images=[i.image_id for i in images if i.exif_flags],
        claimed_defect=claimed,
        claim_unspecified=claimed is None,
        claim_testable=bool(claimed and category in cfg.cv_supported_categories and is_detectable(claimed)),
        claimed_defect_observed=bool(claimed_obs),
        claimed_defect_disputed=bool(claimed_disp) and not claimed_obs,
        claimed_defect_weak=bool(claimed_weak) and not claimed_obs and not claimed_disp,
        corroborating_images=len({d.image_id for d in claimed_obs}),
        location_match=loc,
        other_defects=other,
        unmentioned_defects=unmentioned,
        max_defect_score=max((d.score for d in defects), default=0.0),
        n_tool_errors=len(errors),
    )


def _tool_errors(ledger: Sequence[ToolCall]) -> list[ToolErrorInfo]:
    return [ToolErrorInfo(call_id=c.call_id, tool=c.tool, error=c.error, image_id=c.args.get("image_id"))
            for c in ledger if c.error]


# ----------------------------------------------------------------------------- public API
def build_evidence(*, case: CaseInput, claim: ClaimInfo, order: Optional[OrderLookupResult],
                   clauses: Sequence[PolicyClause], cv: CVCaseReport, config: Optional[AgenticConfig] = None,
                   ledger: Optional[Sequence[ToolCall]] = None) -> EvidenceBundle:
    cfg = config or AgenticConfig()
    ledger = list(ledger if ledger is not None else cv.ledger)
    images = [_image_evidence(ir) for ir in cv.images]
    defects = _defects(cv, cfg)
    errors = _tool_errors(ledger)
    return EvidenceBundle(
        case_id=case.case_id, claim=claim, order=order, clauses=list(clauses), images=images, defects=defects,
        features=_features(claim, order, images, defects, errors, cfg), tool_errors=errors, ledger=ledger,
    )


def judge_view(bundle: EvidenceBundle) -> dict:
    """The ONLY evidence the LLM judge sees: structured, minimal, no price / window / policy outcome.
    The customer's free text is passed separately (as quoted data), never mixed into this dict."""
    f, p = bundle.features, bundle.claim.parsed
    return {
        "claim": {
            "defect_type": p.defect_type.value if p.defect_type else None,
            "location": p.location, "severity": p.severity.value, "timing": p.timing.value,
            "mentioned_defects": [d.value for d in bundle.claim.mentioned_defects],
        },
        "category": bundle.order.order.category.value if bundle.order and bundle.order.order else None,
        "claim_testable_by_detector": f.claim_testable,
        "images": [{"image_id": i.image_id, "usable": i.usable, "detector_ran": i.detector_ok,
                    "product_matches": i.product_match, "duplicate": bool(i.duplicate_of),
                    "exif_flags": i.exif_flags, "quality_ref": i.quality_call_id, "detector_ref": i.detector_call_id}
                   for i in bundle.images],
        "detections": [{
            "image_id": d.image_id, "defect_class": d.defect_class.value, "score": round(d.score, 3), "status": d.status,
            "part_location": d.part_location, "area_pct": d.area_pct, "severity_estimate": d.severity_estimate.value,
            "region_confirms": d.region_confirms, "region_confidence": d.region_confidence,
            "refs": [r.model_dump(mode="json") for r in d.refs()],
        } for d in bundle.defects],
        "derived": {
            "claimed_defect_observed": f.claimed_defect_observed, "claimed_defect_disputed": f.claimed_defect_disputed,
            "claimed_defect_weak": f.claimed_defect_weak, "location_match": f.location_match,
            "other_defects": [c.value for c in f.other_defects],
            "unmentioned_defects": [c.value for c in f.unmentioned_defects],
            "usable_images_scanned": f.n_scanned_usable_images,
        },
        "tool_errors": [{"tool": e.tool, "image_id": e.image_id} for e in bundle.tool_errors],
    }
