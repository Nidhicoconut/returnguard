import pytest
from pydantic import ValidationError
from agentic.schemas import *

def test_parsed_claim_defaults():
    c = ParsedClaim(defect_type="crack_break", location="hinge")
    assert c.severity == Severity.unknown and c.timing == Timing.unknown

def test_bad_defect_class_rejected():
    with pytest.raises(ValidationError):
        ParsedClaim(defect_type="exploded")

def test_bbox_must_be_inside_image():
    BBox(x=0.1, y=0.1, w=0.5, h=0.5)
    with pytest.raises(ValidationError):
        BBox(x=0.8, y=0.1, w=0.5, h=0.5)

def test_finding_needs_evidence():
    with pytest.raises(ValidationError):
        Finding(text="there is a crack", refs=[])
    f = Finding(text="crack near hinge", refs=[EvidenceRef(image_id="img2", tool_call_id="c4")])
    assert len(f.refs) == 1

def test_decision_has_no_reject():
    assert {d.value for d in Decision} == {"AUTO_APPROVE", "REQUEST_MORE_EVIDENCE", "ESCALATE_TO_HUMAN"}

def test_detection_score_range():
    with pytest.raises(ValidationError):
        Detection(defect_class="tear_hole", bbox=BBox(x=0,y=0,w=.1,h=.1), score=1.4)
