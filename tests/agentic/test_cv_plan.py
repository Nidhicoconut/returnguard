import pytest

from agentic.config import AgenticConfig
from agentic.cv import CVToolbox, MockScenario, build_mock_toolbox, run_cv_plan
from agentic.cv.runner import ToolRunner
from agentic.models import CaseInput, ImageInput
from agentic.schemas import BBox, DefectClass, Detection

SKU = "SKU-TSHIRT-1"
CFG = AgenticConfig()


def case(*scenarios):
    imgs = [ImageInput(image_id=f"img{i+1}", path=f"/p/{s}.jpg") for i, s in enumerate(scenarios)]
    return CaseInput(case_id="CASE-1", order_id="ORD-1", claim_text="stain on sleeve", images=imgs)


def run(c, tb=None, cfg=CFG, sku=SKU):
    runner = ToolRunner(timeout_s=5)
    return run_cv_plan(c, sku, tb or build_mock_toolbox(), runner, cfg), runner


def test_defect_case_runs_full_deterministic_chain_in_order():
    rep, runner = run(case("stain_on_sleeve"))
    assert [c.tool for c in runner.ledger] == ["T1_quality", "T7_integrity", "T3_product_match", "T4_detect", "T5_segment", "T9_region_check"]
    (ir,) = rep.images
    (da,) = ir.detections
    assert ir.quality.ok and ir.product_match.match and not ir.errors and not ir.skipped
    assert da.detection.defect_class is DefectClass.stain_discoloration
    assert da.segmentation.part_location == "sleeve" and da.region_check.confirms
    # every id points at a real ledger entry (traceability)
    ids = {c.call_id for c in rep.ledger}
    assert set(ir.call_ids.values()) <= ids and {da.detection_call_id, da.segmentation_call_id, da.region_call_id} <= ids


def test_clean_image_has_no_detections_and_no_followups():
    rep, runner = run(case("clean"))
    assert rep.images[0].detections == [] and not rep.images[0].errors
    assert [c.tool for c in runner.ledger][-1] == "T4_detect"


def test_bad_quality_skips_defect_tools():
    rep, runner = run(case("blurry"))
    ir = rep.images[0]
    assert ir.quality.ok is False and ir.detections == []
    assert ir.skipped["detector"] == "quality_not_ok"
    assert "T4_detect" not in [c.tool for c in runner.ledger]


def test_bad_quality_gate_can_be_disabled_and_min_score_gates_followups():
    rep, runner = run(case("blurry"), cfg=AgenticConfig(cv_skip_on_bad_quality=False))
    ir = rep.images[0]
    assert len(ir.detections) == 1 and ir.skipped["followup[0]"] == "below_min_score"
    assert ir.detections[0].segmentation is None and ir.detections[0].region_check is None
    assert "T5_segment" not in [c.tool for c in runner.ledger]


def test_detector_crash_is_recorded_and_case_still_completes():
    rep, runner = run(case("detector_crash", "stain_on_sleeve"))
    bad, good = rep.images
    assert "detector" in bad.errors and bad.detections == []
    assert "RuntimeError" in bad.errors["detector"]
    assert good.detections and not good.errors            # one bad image does not poison the next
    assert any(c.error for c in rep.ledger)


def test_wrong_product_and_duplicate_are_surfaced_not_hidden():
    rep, _ = run(case("wrong_product", "duplicate_image"))
    assert rep.images[0].product_match.match is False
    assert rep.images[1].integrity.duplicate_of is not None


def test_missing_tools_are_skipped_never_assumed_to_pass():
    tb = CVToolbox(detector=build_mock_toolbox().detector, name="detector-only")
    rep, runner = run(case("hole"), tb=tb)
    ir = rep.images[0]
    assert ir.quality is None and ir.integrity is None and ir.product_match is None
    assert {ir.skipped[k] for k in ("quality", "integrity", "product_match")} == {"tool_not_available"}
    assert ir.skipped["segmenter[0]"] == ir.skipped["region_verifier[0]"] == "tool_not_available"
    assert len(ir.detections) == 1 and [c.tool for c in runner.ledger] == ["T4_detect"]


def test_no_expected_sku_skips_product_match():
    rep, _ = run(case("hole"), sku=None)
    assert rep.images[0].skipped["product_match"] == "no_expected_sku"


def test_region_cap_and_ordering_best_first():
    class Many:
        def detect(self, image):
            return [Detection(defect_class=DefectClass.tear_hole, bbox=BBox(x=0.1 * i, y=0.1, w=0.05, h=0.05), score=0.6 + 0.05 * i)
                    for i in range(4)]

    tb = build_mock_toolbox()
    tb.detector = Many()
    rep, runner = run(case("clean"), tb=tb, cfg=AgenticConfig(cv_max_regions_per_image=2))
    ir = rep.images[0]
    scores = [d.detection.score for d in ir.detections]
    assert scores == sorted(scores, reverse=True) and len(scores) == 4
    assert [d.segmentation is not None for d in ir.detections] == [True, True, False, False]
    assert ir.skipped["followup[2]"] == ir.skipped["followup[3]"] == "region_cap"


def test_followup_failure_is_isolated():
    class BadSeg:
        def segment(self, image, bbox):
            raise ValueError("seg failed")

    tb = build_mock_toolbox()
    tb.segmenter = BadSeg()
    rep, _ = run(case("hole"), tb=tb)
    ir = rep.images[0]
    assert "segmenter[0]" in ir.errors and ir.detections[0].segmentation is None
    assert ir.detections[0].region_check is not None       # the other follow-up still ran


def test_empty_case_and_report_is_json_serialisable():
    rep, _ = run(CaseInput(case_id="C", order_id="O"))
    assert rep.images == [] and rep.ledger == []
    rep2, _ = run(case("hole", "clean"))
    assert '"ledger"' in rep2.model_dump_json()


def test_plan_is_deterministic_across_runs():
    a, _ = run(case("hole", "stain_on_sleeve", "clean"))
    b, _ = run(case("hole", "stain_on_sleeve", "clean"))
    strip = lambda r: [c.model_copy(update={"latency_ms": 0}).model_dump() for c in r.ledger]
    assert strip(a) == strip(b)
