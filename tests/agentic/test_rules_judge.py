import pytest

from agentic.cv import CVToolbox, build_mock_toolbox
from agentic.judge import RulesJudge, run_judge
from agentic.judge.validation import ref_is_valid
from agentic.schemas import BBox, DefectClass, Detection, Verdict


def judge(bundle, runner):
    return run_judge(bundle, RulesJudge(), runner)


class Fixed:
    def __init__(self, *dets):
        self.dets = dets

    def detect(self, image):
        return list(self.dets)


def det(cls=DefectClass.stain_discoloration, score=0.9):
    return Detection(defect_class=cls, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2), score=score)


def test_supported_when_claimed_defect_is_confirmed(make_bundle):
    b, r = make_bundle()
    rep = judge(b, r)
    assert rep.result.verdict is Verdict.SUPPORTED and rep.source == "rules" and not rep.downgraded
    assert len(rep.result.findings) == 2 and rep.dropped_findings == 0
    assert "confirmed by a region check" in rep.result.rationale


def test_supported_but_flags_missing_region_confirmation(make_bundle):
    full = build_mock_toolbox()
    b, r = make_bundle(toolbox=CVToolbox(detector=full.detector, quality=full.quality))
    rep = judge(b, r)
    assert rep.result.verdict is Verdict.SUPPORTED and "not independently confirmed" in rep.result.rationale


def test_contradicted_when_a_different_defect_is_seen(make_bundle):
    b, r = make_bundle(["hole"], claim="a stain on the sleeve")
    rep = judge(b, r)
    assert rep.result.verdict is Verdict.CONTRADICTED and "tear hole" in rep.result.rationale


def test_contradicted_when_clear_photo_shows_nothing(make_bundle):
    b, r = make_bundle(["clean"], claim="a stain on the sleeve")
    rep = judge(b, r)
    assert rep.result.verdict is Verdict.CONTRADICTED and "limited" in rep.result.rationale
    assert rep.result.findings[0].refs[0].tool_call_id == b.images[0].detector_call_id


@pytest.mark.parametrize("kwargs,why", [
    (dict(scenarios=["blurry"]), "No clear photo"),
    (dict(scenarios=["detector_crash"]), "No clear photo"),
    (dict(claim="the shirt is cracked"), "cannot test"),
    (dict(claim="a stain", raw_category="Electronics"), "cannot test"),
    (dict(claim="it is not what I expected"), "exactly one defect"),
    (dict(claim="a stain on the collar"), "not where the customer said"),
])
def test_insufficient_cases(make_bundle, kwargs, why):
    b, r = make_bundle(**kwargs)
    rep = judge(b, r)
    assert rep.result.verdict is Verdict.INSUFFICIENT and why in rep.result.rationale


def test_weak_detection_is_insufficient_not_contradicted(make_bundle):
    full = build_mock_toolbox()
    full.detector = Fixed(det(score=0.3))
    b, r = make_bundle(toolbox=full)
    rep = judge(b, r)
    assert rep.result.verdict is Verdict.INSUFFICIENT and "low-confidence" in rep.result.rationale


def test_conflicting_region_check_is_insufficient(make_bundle):
    full = build_mock_toolbox()
    full.detector = Fixed(det(DefectClass.tear_hole))
    b, r = make_bundle(claim="a hole", toolbox=full)
    assert judge(b, r).result.verdict is Verdict.INSUFFICIENT


@pytest.mark.parametrize("scenario", ["clean", "stain_on_sleeve", "hole", "blurry", "wrong_product", "duplicate_image", "detector_crash"])
@pytest.mark.parametrize("claim", ["stain on the sleeve", "a hole in the shirt", "the screen is cracked", "not happy with it"])
def test_invariants_hold_for_every_scenario_and_claim(make_bundle, scenario, claim):
    b, r = make_bundle([scenario], claim=claim)
    rep = judge(b, r)
    f = b.features
    ledger = {c.call_id: c for c in b.ledger}
    assert rep.dropped_findings == 0 and not rep.downgraded           # the rules judge never needs correcting
    for fnd in rep.result.findings:
        assert fnd.refs and all(ref_is_valid(x, ledger) for x in fnd.refs)
    if rep.result.verdict is Verdict.SUPPORTED:
        assert f.claimed_defect_observed and f.claim_testable and f.n_scanned_usable_images >= 1
    if rep.result.verdict is Verdict.CONTRADICTED:
        assert f.claim_testable and f.n_scanned_usable_images >= 1 and not f.claimed_defect_observed
    assert not hasattr(rep, "decision") and not hasattr(rep.result, "decision")
