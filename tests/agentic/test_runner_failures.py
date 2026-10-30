import json
import time

import pytest

from agentic.cv.runner import ToolRunner
from agentic.schemas import BBox, DefectClass, Detection, QualityResult


def det(score=0.9):
    return Detection(defect_class=DefectClass.tear_hole, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2), score=score)


def q(image_id="i1"):
    return QualityResult(image_id=image_id, blur=0.1, exposure=0.5, resolution_ok=True, product_visible=True, ok=True)


def test_success_is_recorded_with_ids_and_json_safe_args():
    r = ToolRunner()
    res = r.call("T4", lambda: [det()], args={"image_id": "i1", "bbox": det().bbox}, expect=Detection, many=True, image_id="i1")
    assert res.ok and res.call_id == "c1" and res.call.step_no == 1
    assert res.call.result["items"][0]["defect_class"] == "tear_hole"
    assert res.call.args["bbox"] == {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    assert json.loads(r.ledger_json())[0]["tool"] == "T4"
    assert res.ref().tool_call_id == "c1" and res.ref().image_id == "i1"


def test_empty_detection_list_is_success_not_error():
    res = ToolRunner().call("T4", lambda: [], args={}, expect=Detection, many=True, image_id="i1")
    assert res.ok and res.output == []


def test_exception_is_captured_never_raised_and_retried_once():
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("model exploded")

    r = ToolRunner(max_retries=1)
    res = r.call("T4", boom, args={}, expect=Detection, many=True, image_id="i1")
    assert not res.ok and res.output is None and len(calls) == 2
    assert res.call.error == "RuntimeError: model exploded" and res.call.result == {}
    assert r.ledger == [res.call]  # failures are in the audit trail too


def test_transient_failure_recovers_on_retry():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("blip")
        return [det()]

    res = ToolRunner(max_retries=1).call("T4", flaky, args={}, expect=Detection, many=True, image_id="i1")
    assert res.ok and state["n"] == 2 and res.call.error is None


def test_timeout_is_an_error_and_not_retried():
    state = {"n": 0}

    def hang():
        state["n"] += 1
        time.sleep(0.5)
        return []

    r = ToolRunner(timeout_s=0.05, max_retries=3)
    t0 = time.monotonic()
    res = r.call("T4", hang, args={}, expect=Detection, many=True, image_id="i1")
    assert time.monotonic() - t0 < 0.4          # did not wait for the hung worker
    assert not res.ok and "TimeoutError" in res.call.error and state["n"] == 1


@pytest.mark.parametrize("bad", [None, "nope", {"defect_class": "tear_hole"}, [{"junk": 1}]])
def test_malformed_output_is_a_tool_error(bad):
    res = ToolRunner(max_retries=0).call("T4", lambda: bad, args={}, expect=Detection, many=True, image_id="i1")
    assert not res.ok and res.output is None


def test_single_output_none_is_an_error():
    res = ToolRunner(max_retries=0).call("T1", lambda: None, args={}, expect=QualityResult, image_id="i1")
    assert not res.ok and "got None" in res.call.error


def test_output_about_the_wrong_image_is_rejected():
    res = ToolRunner(max_retries=0).call("T1", lambda: q("other"), args={}, expect=QualityResult, image_id="i1")
    assert not res.ok and "image_id" in res.call.error


def test_dict_output_is_validated_into_schema():
    res = ToolRunner().call("T4", lambda: [det().model_dump()], args={}, expect=Detection, many=True, image_id="i1")
    assert res.ok and isinstance(res.output[0], Detection)


def test_step_numbers_increase_across_calls():
    r = ToolRunner()
    ids = [r.call("T1", lambda: q(), args={}, expect=QualityResult, image_id="i1").call_id for _ in range(3)]
    assert ids == ["c1", "c2", "c3"] and [c.step_no for c in r.ledger] == [1, 2, 3]


def test_ref_requires_image():
    res = ToolRunner().call("T0", lambda: q(), args={}, expect=QualityResult)
    with pytest.raises(ValueError):
        res.ref()
