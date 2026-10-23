import cv2
import pytest
from fastapi.testclient import TestClient

from app import config, decision, policy
from app.main import app
from make_synthetic_data import make_image


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _post(client, order, reason, img, ctype="image/png", name="p.png"):
    ok, buf = cv2.imencode(".png", img)
    return client.post("/api/cases", data={"order_id": order, "reason": reason}, files={"image": (name, buf.tobytes(), ctype)})


# ---------- policy RAG
def test_policy_has_seven_clauses_and_must_include():
    assert policy.clause_ids() == {f"R{i}" for i in range(1, 8)}
    hits = policy.retrieve("cracked screen refund", k=2, must_include=["R1"])
    assert "R1" in {h["id"] for h in hits}


# ---------- decision engine (pure functions)
def facts(**kw):
    base = {"order": {"price": 1000, "category": "Home", "product": "Mug"}, "days_since_delivery": 5, "return_window_days": 30,
            "reason": "broken", "claims_defect": True, "vision": {"label": "defective", "confidence": 0.9, "defect_type": "crack"}, "clauses": []}
    base.update(kw)
    return base


def test_rules_approve_verified_defect():
    r = decision.finalize(facts(), None, "rules")
    assert r["decision"] == "approve" and "R2" in r["cited_clauses"]


def test_late_return_is_rejected_even_if_llm_says_approve():
    f = facts(days_since_delivery=45)
    r = decision.finalize(f, {"decision": "approve", "cited_clauses": ["R2"]}, "anthropic")
    assert r["decision"] == "reject" and r["cited_clauses"] == ["R1"]


def test_llm_disagreement_on_soft_rule_escalates():
    r = decision.finalize(facts(), {"decision": "reject", "cited_clauses": ["R1"]}, "anthropic")
    assert r["decision"] == "escalate" and r["source"] == "guardrail"


def test_invented_clause_is_dropped():
    r = decision.finalize(facts(), {"decision": "approve", "cited_clauses": ["R99", "R2"], "summary": "ok"}, "anthropic")
    assert "R99" not in r["cited_clauses"] and r["notes"]


def test_high_value_and_mismatch_and_inconclusive_escalate():
    assert decision.finalize(facts(order={"price": 50000, "category": "Electronics"}), None, "rules")["decision"] == "escalate"
    assert decision.finalize(facts(vision={"label": "intact", "confidence": 0.9}), None, "rules")["decision"] == "escalate"
    assert decision.finalize(facts(vision={"label": "inconclusive", "confidence": 0.2}), None, "rules")["decision"] == "escalate"


def test_non_returnable_category():
    f = facts(order={"price": 500, "category": "Personal Care"}, claims_defect=False, vision={"label": "intact", "confidence": 0.9})
    assert decision.finalize(f, None, "rules")["decision"] == "reject"


# ---------- end to end through the API
def test_end_to_end_scenarios(client):
    crack, _ = make_image(1, "crack")
    intact, _ = make_image(2, None)
    cases = [
        ("ORD-1001", "Arrived with a crack on the body", crack, "Approved"),
        ("ORD-1002", "I changed my mind", intact, "Approved"),
        ("ORD-1007", "The strap is torn", intact, "Needs human review"),
        ("ORD-1004", "It is cracked", crack, "Rejected"),
        ("ORD-9999", "broken", crack, "Needs human review"),
    ]
    for order, reason, img, status in cases:
        r = _post(client, order, reason, img)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == status, (order, body["summary"])
        assert body["ticket_id"] and body["trace"] and body["citations"]
    detail = client.get(f"/api/cases/{body['id']}").json()
    assert detail["events"]


def test_staff_review_overrides_and_logs(client):
    crack, _ = make_image(3, "crack")
    case = _post(client, "ORD-1001", "cracked", crack).json()
    r = client.post(f"/api/cases/{case['id']}/review", json={"action": "reject", "note": "photo is of a different item", "vision_feedback": "false_positive"})
    body = r.json()
    assert body["status"] == "Rejected" and body["decided_by"] == "store staff"
    assert any(e["action"] == "review" and e["detail"]["overrode_agent"] for e in body["events"])


def test_upload_validation(client):
    img, _ = make_image(4, None)
    assert _post(client, "ORD-1001", "x", img, ctype="text/plain", name="a.txt").status_code == 415
    bad = client.post("/api/cases", data={"order_id": "ORD-1001", "reason": "x"}, files={"image": ("a.png", b"not an image", "image/png")})
    assert bad.status_code == 422
    assert client.post("/api/cases", data={"order_id": "ORD-1001", "reason": "  "}, files={"image": ("a.png", b"x", "image/png")}).status_code == 422
    assert client.get("/api/cases/99999").status_code == 404


def test_index_and_media_are_served(client):
    assert client.get("/").status_code == 200
    case = client.get("/api/cases").json()[0]
    assert client.get(case["image_url"]).status_code == 200
