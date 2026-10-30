import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentic.config import AgenticConfig
from agentic.judge import RulesJudge, run_judge
from agentic.models import CaseInput, ImageInput, ReviewAction, ReviewTask
from agentic.policy_engine import decide
from agentic.report import build_report
from agentic.review import InMemoryReviewQueue, JsonlReviewQueue, ReviewError, build_review_task
from agentic.schemas import Decision

NOW = "2026-10-30T12:00:00+00:00"


def task_for(make_bundle, *a, rate=0.0, **kw):
    cfg = AgenticConfig(audit_sample_rate=rate)
    bundle, runner = make_bundle(*a, **kw)
    rep = run_judge(bundle, RulesJudge(), runner)
    decision = decide(bundle, rep, config=cfg)
    case = CaseInput(case_id="CASE-1", order_id="ORD-1", claim_text=bundle.claim.claim_text,
                     images=[ImageInput(image_id=i.image_id, path="/p/x.jpg") for i in bundle.images])
    report = build_report(case=case, decision=decision, bundle=bundle, judge=rep, now=NOW, config=cfg)
    return build_review_task(case=case, decision=decision, bundle=bundle, judge=rep, report=report, now=NOW), decision


def bare(case_id="C", codes=("HIGH_VALUE",), priority="normal", spot=False):
    return ReviewTask(case_id=case_id, created_at=NOW, priority=priority, machine_decision=Decision.ESCALATE_TO_HUMAN,
                      reason_codes=list(codes), would_have_auto_approved=spot)


# ----------------------------------------------------------------------------- task building
def test_task_carries_evidence_checklist_images_and_priority(make_bundle):
    t, d = task_for(make_bundle, price=25000, days=45)
    assert t.machine_decision is Decision.ESCALATE_TO_HUMAN and t.priority == "high" and t.status == "open"
    assert t.reason_codes == d.reason_codes and t.clause_ids == d.clause_ids
    assert t.evidence is not None and t.report is not None and t.judge is not None and [i.image_id for i in t.images] == ["img1"]
    assert any("goodwill" in c for c in t.checklist) and any("order value" in c for c in t.checklist)


def test_audit_sample_task_is_a_low_priority_spot_check(make_bundle):
    t, d = task_for(make_bundle, rate=1.0)
    assert d.reason_codes == ["AUDIT_SAMPLE"] and t.would_have_auto_approved and t.priority == "low"


def test_normal_priority_for_ordinary_escalations(make_bundle):
    t, _ = task_for(make_bundle, scenarios=["clean"])
    assert t.priority == "normal" and not t.would_have_auto_approved


# ----------------------------------------------------------------------------- in-memory queue
def test_ids_are_sequential_and_open_tasks_sort_by_priority():
    q = InMemoryReviewQueue()
    a = q.enqueue(bare("A", priority="low"))
    b = q.enqueue(bare("B", priority="high"))
    c = q.enqueue(bare("C", priority="normal"))
    assert [a.task_id, b.task_id, c.task_id] == ["RT-00001", "RT-00002", "RT-00003"]
    assert [t.case_id for t in q.list_open()] == ["B", "C", "A"]
    assert [t.case_id for t in q.list_open("low")] == ["A"] and q.get("RT-00002").case_id == "B" and q.get("nope") is None


def test_enqueue_rejects_duplicates_and_non_open_tasks():
    q = InMemoryReviewQueue()
    t = q.enqueue(bare())
    with pytest.raises(ReviewError):
        q.enqueue(t)
    with pytest.raises(ReviewError):
        q.enqueue(bare().model_copy(update={"status": "resolved"}))


def test_resolve_records_an_audited_outcome():
    q = InMemoryReviewQueue()
    t = q.enqueue(bare())
    o = q.resolve(t.task_id, reviewer=" Asha ", action=ReviewAction.APPROVE_REFUND, note="photo clearly shows it", now=NOW)
    assert (o.reviewer, o.action, o.case_id, o.machine_decision) == ("Asha", "approve_refund", "C", Decision.ESCALATE_TO_HUMAN)
    assert q.get(t.task_id).status == "resolved" and q.list_open() == [] and q.outcomes("C") == [o] and q.outcomes("X") == []


@pytest.mark.parametrize("kwargs,msg", [
    (dict(reviewer="", action="approve_refund"), "reviewer"),
    (dict(reviewer="  ", action="approve_refund"), "reviewer"),
    (dict(reviewer="A", action="refund_everything"), "action"),
    (dict(reviewer="A", action="decline", note=""), "note"),
    (dict(reviewer="A", action="decline", note="no"), "note"),
])
def test_invalid_resolutions_are_refused_and_leave_the_task_open(kwargs, msg):
    q = InMemoryReviewQueue()
    t = q.enqueue(bare())
    with pytest.raises(ReviewError, match=msg):
        q.resolve(t.task_id, **kwargs)
    assert q.get(t.task_id).status == "open" and q.outcomes() == []


def test_only_a_human_can_decline_and_it_needs_a_reason():
    q = InMemoryReviewQueue()
    t = q.enqueue(bare())
    assert q.resolve(t.task_id, reviewer="A", action="decline", note="item visibly worn out, not defective").action == "decline"


def test_a_task_cannot_be_resolved_twice_or_unknown():
    q = InMemoryReviewQueue()
    t = q.enqueue(bare())
    q.resolve(t.task_id, reviewer="A", action="approve_refund")
    with pytest.raises(ReviewError, match="already resolved"):
        q.resolve(t.task_id, reviewer="B", action="decline", note="changed my mind")
    with pytest.raises(ReviewError, match="unknown"):
        q.resolve("RT-99999", reviewer="A", action="approve_refund")
    assert len(q.outcomes()) == 1


def test_overrides_are_flagged_only_for_spot_checks_the_human_did_not_confirm():
    q = InMemoryReviewQueue()
    a, b, c = (q.enqueue(bare(str(i), codes=("AUDIT_SAMPLE",), spot=True)) for i in range(3))
    n = q.enqueue(bare("N"))
    assert q.resolve(a.task_id, reviewer="R", action="approve_refund").overrides_machine is False
    assert q.resolve(b.task_id, reviewer="R", action="decline", note="not a real defect").overrides_machine is True
    assert q.resolve(c.task_id, reviewer="R", action="request_evidence").overrides_machine is True
    assert q.resolve(n.task_id, reviewer="R", action="decline", note="not covered by policy").overrides_machine is False


# ----------------------------------------------------------------------------- JSONL queue
def test_jsonl_persists_and_replays_everything(tmp_path, make_bundle):
    p = tmp_path / "sub" / "review.jsonl"
    q = JsonlReviewQueue(p)
    full, _ = task_for(make_bundle, days=45)
    t1 = q.enqueue(full)
    t2 = q.enqueue(bare("B", priority="high"))
    q.resolve(t2.task_id, reviewer="Asha", action="approve_replacement", note="ok", now=NOW)

    q2 = JsonlReviewQueue(p)
    assert q2.load_warnings == []
    assert q2.get(t1.task_id) == t1 and q2.get(t2.task_id).status == "resolved"
    assert [t.task_id for t in q2.list_open()] == [t1.task_id] and q2.outcomes()[0].reviewer == "Asha"
    assert q2.enqueue(bare("C")).task_id == "RT-00003"        # id sequence continues after restart
    assert q2.get(t1.task_id).evidence.features == full.evidence.features  # full evidence survives the round trip


def test_jsonl_is_append_only(tmp_path):
    p = tmp_path / "r.jsonl"
    q = JsonlReviewQueue(p)
    t = q.enqueue(bare())
    first = p.read_text().splitlines()
    q.resolve(t.task_id, reviewer="A", action="approve_refund")
    after = p.read_text().splitlines()
    assert after[: len(first)] == first and len(after) == len(first) + 1
    assert [json.loads(l)["type"] for l in after] == ["task", "outcome"]


def test_a_corrupt_line_is_skipped_without_losing_the_rest(tmp_path):
    p = tmp_path / "r.jsonl"
    q = JsonlReviewQueue(p)
    q.enqueue(bare("A"))
    with open(p, "a") as fh:
        fh.write("{this is not json\n" + json.dumps({"type": "mystery"}) + "\n")
    q.enqueue(bare("B"))
    q2 = JsonlReviewQueue(p)
    assert len(q2.load_warnings) == 2 and {t.case_id for t in q2.list_open()} == {"A", "B"}


def test_a_failed_write_leaves_memory_unchanged(tmp_path):
    q = JsonlReviewQueue(tmp_path / "r.jsonl")
    t = q.enqueue(bare("A"))

    def boom(_event):
        raise OSError("disk full")

    q._persist = boom
    with pytest.raises(OSError):
        q.enqueue(bare("B"))
    with pytest.raises(OSError):
        q.resolve(t.task_id, reviewer="A", action="approve_refund")
    assert [x.case_id for x in q.list_open()] == ["A"] and q.outcomes() == []
    q._persist = type(q)._persist.__get__(q)
    assert q.enqueue(bare("C")).task_id == "RT-00002"   # sequence did not skip or reuse


def test_concurrent_enqueues_get_unique_ids_and_whole_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    q = JsonlReviewQueue(p)
    with ThreadPoolExecutor(8) as ex:
        tasks = list(ex.map(lambda i: q.enqueue(bare(f"C{i}")), range(60)))
    assert len({t.task_id for t in tasks}) == 60
    lines = p.read_text().splitlines()
    assert len(lines) == 60 and all(json.loads(l)["type"] == "task" for l in lines)
    assert len(JsonlReviewQueue(p).list_open()) == 60
