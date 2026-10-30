import time

import pytest

from agentic.judge import RulesJudge, run_judge, validate_draft
from agentic.judge.validation import ref_is_valid
from agentic.models import JudgeDraft
from agentic.schemas import BBox, ConsistencyResult, EvidenceRef, Finding, Verdict


def draft(verdict=Verdict.SUPPORTED, rationale="ok", findings=()):
    return JudgeDraft(result=ConsistencyResult(verdict=verdict, rationale=rationale, findings=list(findings)))


def good_ref(b):
    d = b.defects[0]
    return EvidenceRef(image_id=d.image_id, tool_call_id=d.detection_call_id, bbox=d.bbox)


def validate(b, d):
    return validate_draft(d, b)


# ----------------------------------------------------------------------------- citations
def test_valid_draft_passes_unchanged(make_bundle):
    b, _ = make_bundle()
    res, warnings, dropped, down = validate(b, draft(findings=[Finding(text="stain seen", refs=[good_ref(b)])]))
    assert res.verdict is Verdict.SUPPORTED and warnings == [] and dropped == 0 and not down


def test_hallucinated_call_id_drops_finding_and_downgrades(make_bundle):
    b, _ = make_bundle()
    f = Finding(text="x", refs=[EvidenceRef(image_id="img1", tool_call_id="c999")])
    res, warnings, dropped, down = validate(b, draft(findings=[f]))
    assert res.verdict is Verdict.INSUFFICIENT and res.findings == [] and dropped == 1 and down
    assert any("downgraded" in w for w in warnings)


def test_wrong_image_errored_call_and_non_image_calls_are_rejected(make_bundle):
    b, _ = make_bundle(["detector_crash", "stain_on_sleeve"])
    ledger = {c.call_id: c for c in b.ledger}
    crash = next(c for c in b.ledger if c.error)
    order_call = next(c for c in b.ledger if c.tool == "T0_order_lookup")
    policy_call = next(c for c in b.ledger if c.tool == "T8_policy")
    ok_call = b.images[1].detector_call_id
    assert ref_is_valid(EvidenceRef(image_id="img2", tool_call_id=ok_call), ledger)
    assert not ref_is_valid(EvidenceRef(image_id="img1", tool_call_id=ok_call), ledger)   # wrong image
    assert not ref_is_valid(EvidenceRef(image_id="img1", tool_call_id=crash.call_id), ledger)  # failed call
    assert not ref_is_valid(EvidenceRef(image_id="img1", tool_call_id=order_call.call_id), ledger)
    assert not ref_is_valid(EvidenceRef(image_id="img1", tool_call_id=policy_call.call_id), ledger)


def test_bbox_must_match_a_box_the_tool_actually_saw(make_bundle):
    b, _ = make_bundle()
    d = b.defects[0]
    ledger = {c.call_id: c for c in b.ledger}
    near = BBox(x=d.bbox.x + 0.005, y=d.bbox.y, w=d.bbox.w, h=d.bbox.h)
    far = BBox(x=0.7, y=0.7, w=0.1, h=0.1)
    assert ref_is_valid(EvidenceRef(image_id=d.image_id, tool_call_id=d.detection_call_id, bbox=near), ledger)
    assert not ref_is_valid(EvidenceRef(image_id=d.image_id, tool_call_id=d.detection_call_id, bbox=far), ledger)


def test_partially_valid_finding_keeps_its_valid_refs(make_bundle):
    b, _ = make_bundle()
    f = Finding(text="mixed", refs=[good_ref(b), EvidenceRef(image_id="img1", tool_call_id="c999")])
    res, warnings, dropped, down = validate(b, draft(findings=[f]))
    assert len(res.findings[0].refs) == 1 and dropped == 0 and not down
    assert any("unresolvable" in w for w in warnings)


# ----------------------------------------------------------------------------- verdict vs features
def test_supported_without_any_observed_defect_is_downgraded(make_bundle):
    b, _ = make_bundle(["clean"])
    ref = EvidenceRef(image_id="img1", tool_call_id=b.images[0].detector_call_id)
    res, _, _, down = validate(b, draft(Verdict.SUPPORTED, findings=[Finding(text="looks stained", refs=[ref])]))
    assert res.verdict is Verdict.INSUFFICIENT and down and res.rationale.startswith("[verdict downgraded")


def test_contradicted_although_defect_observed_is_downgraded(make_bundle):
    b, _ = make_bundle()
    res, _, _, down = validate(b, draft(Verdict.CONTRADICTED, findings=[Finding(text="no stain", refs=[good_ref(b)])]))
    assert res.verdict is Verdict.INSUFFICIENT and down


def test_contradicted_on_untestable_claim_is_downgraded(make_bundle):
    b, _ = make_bundle(["clean"], claim="the shirt is cracked")
    ref = EvidenceRef(image_id="img1", tool_call_id=b.images[0].detector_call_id)
    res, _, _, down = validate(b, draft(Verdict.CONTRADICTED, findings=[Finding(text="no crack", refs=[ref])]))
    assert res.verdict is Verdict.INSUFFICIENT and down


def test_insufficient_needs_no_findings(make_bundle):
    b, _ = make_bundle()
    res, _, _, down = validate(b, draft(Verdict.INSUFFICIENT))
    assert res.verdict is Verdict.INSUFFICIENT and not down


# ----------------------------------------------------------------------------- text hygiene
def test_decision_words_are_redacted(make_bundle):
    b, _ = make_bundle()
    f = Finding(text="fine, so AUTO_APPROVE it", refs=[good_ref(b)])
    res, warnings, _, _ = validate(b, draft(rationale="Please escalate_to_human or REQUEST_MORE_EVIDENCE", findings=[f]))
    blob = res.rationale + res.findings[0].text
    assert "AUTO_APPROVE" not in blob.upper() and "ESCALATE_TO_HUMAN" not in blob.upper() and "[redacted]" in blob
    assert any("decision keyword" in w for w in warnings)


def test_limits_and_control_characters(make_bundle):
    b, _ = make_bundle()
    findings = [Finding(text="t" * 900 + "\x00", refs=[good_ref(b)]) for _ in range(15)]
    res, _, dropped, _ = validate(b, draft(rationale="r" * 5000, findings=findings))
    assert len(res.findings) == 10 and dropped == 5 and len(res.rationale) <= 1200
    assert all(len(f.text) <= 400 and "\x00" not in f.text for f in res.findings)


# ----------------------------------------------------------------------------- run_judge safety net
class Boom:
    name = "llm"

    def judge(self, bundle):
        raise RuntimeError("provider down")


class Slow:
    name = "llm"

    def judge(self, bundle):
        time.sleep(0.4)


def test_primary_failure_falls_back_to_rules_and_is_audited(make_bundle):
    b, r = make_bundle()
    rep = run_judge(b, Boom(), r, RulesJudge(), max_retries=0)
    assert rep.source == "fallback" and rep.result.verdict is Verdict.SUPPORTED
    assert any("provider down" in w for w in rep.warnings)
    judge_calls = [c for c in r.ledger if c.tool == "J1_consistency_judge"]
    assert len(judge_calls) == 2 and judge_calls[0].error and judge_calls[1].error is None


def test_timeout_falls_back_without_waiting(make_bundle):
    b, r = make_bundle()
    t0 = time.monotonic()
    rep = run_judge(b, Slow(), r, RulesJudge(), timeout_s=0.05)
    assert time.monotonic() - t0 < 0.3 and rep.source == "fallback" and any("TimeoutError" in w for w in rep.warnings)


def test_total_failure_yields_insufficient_never_raises(make_bundle):
    b, r = make_bundle()
    rep = run_judge(b, Boom(), r, Boom(), max_retries=0)
    assert rep.source == "none" and rep.result.verdict is Verdict.INSUFFICIENT and rep.result.findings == []
    assert run_judge(b, Boom(), r, None, max_retries=0).result.verdict is Verdict.INSUFFICIENT


def test_malformed_judge_output_is_a_tool_error(make_bundle):
    class Junk:
        name = "llm"

        def judge(self, bundle):
            return {"verdict": "MAYBE"}

    b, r = make_bundle()
    rep = run_judge(b, Junk(), r, RulesJudge(), max_retries=0)
    assert rep.source == "fallback"


def test_rules_primary_reports_rules_source(make_bundle):
    b, r = make_bundle()
    assert run_judge(b, RulesJudge(), r).source == "rules"
