"""Deterministic CV tool plan for one case (no LLM involved).

Per image:  T1 quality -> T7 integrity -> T3 product match -> T4 detect
            -> for each detection (best first, score >= min_score, capped): T5 segment + T9 region check

Rules: a missing tool is recorded as *skipped*; a failing tool is recorded as an *error*; neither is
ever treated as a pass. If T1 ran and says the photo is unusable, T4/T5/T9 are skipped (policy R5:
ask for a better photo instead of reading evidence out of a bad one).
"""
from typing import Optional

from agentic.config import AgenticConfig
from agentic.cv.interfaces import CVToolbox
from agentic.cv.runner import ToolRunner
from agentic.models import CaseInput, CVCaseReport, DetectionAnalysis, ImageCVReport, ImageInput
from agentic.schemas import Detection, IntegrityResult, ProductMatch, QualityResult, RegionCheck, Segmentation

T1, T3, T4, T5, T7, T9 = "T1_quality", "T3_product_match", "T4_detect", "T5_segment", "T7_integrity", "T9_region_check"


def _run_image(image: ImageInput, case: CaseInput, expected_sku: Optional[str], tb: CVToolbox,
               runner: ToolRunner, cfg: AgenticConfig) -> ImageCVReport:
    rep = ImageCVReport(image_id=image.image_id)
    iid = image.image_id

    def note(role: str, res) -> None:
        rep.call_ids[role] = res.call_id
        if not res.ok:
            rep.errors[role] = res.call.error

    # T1 quality
    if tb.quality is None:
        rep.skipped["quality"] = "tool_not_available"
    else:
        res = runner.call(T1, lambda: tb.quality.check(image), args={"image_id": iid}, expect=QualityResult, image_id=iid)
        note("quality", res)
        rep.quality = res.output

    # T7 integrity
    if tb.integrity is None:
        rep.skipped["integrity"] = "tool_not_available"
    else:
        res = runner.call(T7, lambda: tb.integrity.check(image, case.images),
                          args={"image_id": iid, "case_image_ids": [i.image_id for i in case.images]},
                          expect=IntegrityResult, image_id=iid)
        note("integrity", res)
        rep.integrity = res.output

    # T3 product match
    if tb.product_matcher is None:
        rep.skipped["product_match"] = "tool_not_available"
    elif not expected_sku:
        rep.skipped["product_match"] = "no_expected_sku"
    else:
        res = runner.call(T3, lambda: tb.product_matcher.match(image, expected_sku),
                          args={"image_id": iid, "expected_sku": expected_sku}, expect=ProductMatch, image_id=iid)
        note("product_match", res)
        rep.product_match = res.output

    # gate: unusable photo -> no defect evidence (R5)
    if cfg.cv_skip_on_bad_quality and rep.quality is not None and not rep.quality.ok:
        for role in ("detector", "segmenter", "region_verifier"):
            rep.skipped[role] = "quality_not_ok"
        return rep

    # T4 detect
    if tb.detector is None:
        rep.skipped["detector"] = "tool_not_available"
        return rep
    res = runner.call(T4, lambda: tb.detector.detect(image), args={"image_id": iid}, expect=Detection,
                      many=True, image_id=iid)
    note("detector", res)
    detections = sorted(res.output or [], key=lambda d: -d.score)  # best first, stable
    det_call_id = res.call_id

    for idx, det in enumerate(detections):
        da = DetectionAnalysis(detection=det, detection_call_id=det_call_id)
        if det.score < cfg.detection_min_score:
            rep.skipped[f"followup[{idx}]"] = "below_min_score"
        elif idx >= cfg.cv_max_regions_per_image:
            rep.skipped[f"followup[{idx}]"] = "region_cap"
        else:
            if tb.segmenter is None:
                rep.skipped[f"segmenter[{idx}]"] = "tool_not_available"
            else:
                s = runner.call(T5, lambda det=det: tb.segmenter.segment(image, det.bbox),
                                args={"image_id": iid, "bbox": det.bbox}, expect=Segmentation, image_id=iid)
                da.segmentation, da.segmentation_call_id = s.output, s.call_id
                if not s.ok:
                    rep.errors[f"segmenter[{idx}]"] = s.call.error
            if tb.region_verifier is None:
                rep.skipped[f"region_verifier[{idx}]"] = "tool_not_available"
            else:
                v = runner.call(T9, lambda det=det: tb.region_verifier.verify(image, det.bbox, det.defect_class),
                                args={"image_id": iid, "bbox": det.bbox, "defect_class": det.defect_class.value},
                                expect=RegionCheck, image_id=iid)
                da.region_check, da.region_call_id = v.output, v.call_id
                if not v.ok:
                    rep.errors[f"region_verifier[{idx}]"] = v.call.error
        rep.detections.append(da)
    return rep


def run_cv_plan(case: CaseInput, expected_sku: Optional[str], toolbox: CVToolbox, runner: ToolRunner,
                config: Optional[AgenticConfig] = None) -> CVCaseReport:
    cfg = config or AgenticConfig()
    images = [_run_image(img, case, expected_sku, toolbox, runner, cfg) for img in case.images]
    return CVCaseReport(case_id=case.case_id, images=images, ledger=list(runner.ledger))
