import itertools

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, decision, llm, policy, preanalysis
from app.main import app
from make_synthetic_data import make_image

APPROVE, MORE, ESCALATE, REJECT = decision.APPROVE, decision.MORE, decision.ESCALATE, decision.REJECT


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def login(client, username, password="demo1234", role="customer"):
    client.post("/api/logout")
    r = client.post("/api/login", json={"username": username, "password": password, "role": role})
    assert r.status_code == 200, r.text


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def submit(client, order, reason, *imgs):
    files = [("images", (f"p{i}.png", png(img), "image/png")) for i, img in enumerate(imgs)]
    return client.post("/api/cases", data={"order_id": order, "reason": reason}, files=files)


# ---------- policy RAG
def test_policy_has_eight_clothing_clauses():
    assert policy.clause_ids() == {f"R{i}" for i in range(1, 9)}
    hits = policy.retrieve("hole in the sleeve refund", k=2, must_include=["R1"])
    assert "R1" in {h["id"] for h in hits}


# ---------- decision engine (pure functions)
def facts(**kw):
    base = {"order": {"price": 1000, "product": "Cotton T-Shirt", "product_type": "t-shirt"}, "days_since_delivery": 3,
            "return_window_days": 10, "reason": "there is a hole", "claim": {"defect_type": "tear_hole", "location": ""},
            "claims_defect": True, "photo_quality": {"ok_photos": 1, "total": 1, "instructions": []}, "integrity_flags": [],
            "detection": {"verdict": "damaged", "confidence": 0.8, "defect_type": "hole"},
            "vision_check": {"status": "damage_seen", "confirms": True, "matches_claim": True},
            "evidence_rounds_used": 0, "clauses": []}
    base.update(kw)
    return base


def verdict(**kw):
    return decision.finalize(facts(**kw), None, "rules")


def test_confirmed_defect_is_auto_approved():
    r = verdict()
    assert r["decision"] == APPROVE and "R2" in r["cited_clauses"] and r["resolution"] == "full_refund_or_replacement"


def test_late_return_is_declined_even_if_llm_says_approve(monkeypatch):
    r = decision.finalize(facts(days_since_delivery=25), {"decision": APPROVE, "cited_clauses": ["R2"]}, "openai")
    assert r["decision"] == REJECT and r["status"] == "Rejected" and r["reason_code"] == "OUTSIDE_WINDOW"
    assert "appeal" in r["customer_message"]
    monkeypatch.setattr(config, "AUTO_REJECT_OUTSIDE_WINDOW", False)  # the plan's "never auto-reject" mode
    assert decision.finalize(facts(days_since_delivery=25), None, "rules")["decision"] == ESCALATE


def test_llm_disagreeing_with_an_approval_escalates():
    r = decision.finalize(facts(), {"decision": ESCALATE, "cited_clauses": ["R6"]}, "openai")
    assert r["decision"] == ESCALATE and r["source"] == "guardrail"


def test_invented_clause_is_dropped():
    r = decision.finalize(facts(), {"decision": APPROVE, "cited_clauses": ["R99", "R2"], "summary": "ok"}, "openai")
    assert "R99" not in r["cited_clauses"] and r["notes"]


def test_vision_check_is_required_for_a_defect_approval():
    r = verdict(vision_check={"status": "unavailable", "confirms": None})
    assert r["decision"] == ESCALATE and r["reason_code"] == "UNCONFIRMED_DEFECT"
    assert verdict(vision_check={"status": "no_damage_seen", "confirms": False})["decision"] == ESCALATE


def test_high_value_integrity_and_unmentioned_damage_escalate():
    assert verdict(order={"price": 6999, "product": "Jacket", "product_type": "jacket"})["reason_code"] == "HIGH_VALUE"
    assert verdict(integrity_flags=["Same photo as case 3."])["reason_code"] == "INTEGRITY_FLAG"
    r = verdict(claims_defect=False, claim={"defect_type": None}, reason="changed my mind")
    assert r["decision"] == ESCALATE and r["reason_code"] == "UNMENTIONED_DAMAGE"


def test_bad_photos_ask_for_more_then_escalate():
    bad = {"ok_photos": 0, "total": 1, "instructions": ["The photo is blurry."]}
    r = verdict(photo_quality=bad)
    assert r["decision"] == MORE and "blurry" in r["customer_message"]
    assert verdict(photo_quality=bad, evidence_rounds_used=2)["decision"] == ESCALATE


def test_claimed_damage_not_visible():
    clean = {"verdict": "clean", "confidence": 0.9}
    no_vlm = {"status": "no_damage_seen", "confirms": False}
    assert verdict(detection=clean, vision_check=no_vlm)["decision"] == MORE
    r = verdict(detection=clean, vision_check=no_vlm, evidence_rounds_used=2)
    assert r["decision"] == ESCALATE and r["verdict"] == "CONTRADICTED"


def test_change_of_mind_and_non_returnable():
    r = verdict(claims_defect=False, claim={"defect_type": None}, detection={"verdict": "clean", "confidence": 0.9}, vision_check=None)
    assert r["decision"] == APPROVE and r["resolution"] == "store_credit"
    socks = {"price": 399, "product": "Socks", "product_type": "socks"}
    assert verdict(order=socks, claims_defect=False, claim={"defect_type": None})["decision"] == ESCALATE


def test_only_late_returns_are_declined_automatically():
    grid = itertools.product([None, facts()["order"]], [3, 25], [True, False], ["damaged", "unsure", "clean", "no_garment"],
                             [None, True, False], [0, 1], [0, 2])
    for order, days, claim, det, vlm_ok, ok_photos, rounds in grid:
        f = facts(order=order, days_since_delivery=days, claims_defect=claim, detection={"verdict": det, "confidence": 0.7},
                  vision_check={"confirms": vlm_ok, "matches_claim": True}, photo_quality={"ok_photos": ok_photos, "total": 1},
                  evidence_rounds_used=rounds)
        d = decision.policy_verdict(f)["decision"]
        assert d in (APPROVE, MORE, ESCALATE, REJECT)
        assert (d == REJECT) == (order is not None and days > 10)


# ---------- pre-analysis and claim parsing
def test_quality_gate_flags_blur_and_darkness():
    img, _ = make_image(5, None)
    assert preanalysis.assess_quality(img)["ok"]
    assert "blurry" in preanalysis.assess_quality(cv2.GaussianBlur(img, (0, 0), 6))["problems"]
    assert "too_dark" in preanalysis.assess_quality((img * 0.05).astype(np.uint8))["problems"]


def test_keyword_claim_parsing():
    c = llm.parse_claim_keywords("There is a hole near the collar, it arrived like this. I want a refund.")
    assert c["defect_type"] == "tear_hole" and c["location"] == "collar" and c["timing"] == "on_arrival" and c["requested_resolution"] == "refund"
    assert llm.parse_claim_keywords("changed my mind, too big")["defect_type"] is None


# ---------- accounts and access control
def test_login_and_roles(client):
    client.post("/api/logout")
    assert client.post("/api/login", json={"username": "asha", "password": "nope"}).status_code == 401
    assert client.post("/api/login", json={"username": "asha", "password": "demo1234", "role": "reviewer"}).status_code == 403
    assert client.get("/api/me").status_code == 401
    login(client, "asha")
    assert client.get("/api/me").json()["role"] == "customer"
    assert client.get("/api/review/queue").status_code == 403
    assert {o["order_id"] for o in client.get("/api/orders").json()} == {"ORD-2001", "ORD-2002", "ORD-2003"}
    assert submit(client, "ORD-2004", "hole", make_image(1, None)[0]).status_code == 404  # Ravi's order


def test_pages_redirect_by_role(client):
    client.post("/api/logout")
    assert client.get("/login").status_code == 200
    assert client.get("/customer", follow_redirects=False).status_code in (302, 307)
    login(client, "asha")
    assert client.get("/customer").status_code == 200
    assert client.get("/review", follow_redirects=False).status_code in (302, 307)


# ---------- end to end through the API
def test_change_of_mind_is_approved_and_photos_are_private(client):
    login(client, "asha")
    img, _ = make_image(11, None)
    body = submit(client, "ORD-2001", "I changed my mind, it does not fit", img).json()
    assert body["status"] == "Approved" and body["resolution"] == "store_credit", body
    assert [s["step"] for s in body["steps"]][:3] == ["get_order", "parse_claim", "check_photos"]
    url = body["photos"][0]["url"]
    assert client.get(url).status_code == 200
    login(client, "ravi")
    assert client.get(url).status_code == 404 and client.get(f"/api/cases/{body['id']}").status_code == 404


def test_damage_claim_goes_to_reviewer_without_vision_check(client):
    login(client, "asha")
    img, _ = make_image(12, "stain")
    body = submit(client, "ORD-2002", "there is a big stain on the front", img).json()
    assert body["status"] == "With reviewer", body


def test_blurry_photo_asks_for_more_then_rechecks(client):
    login(client, "ravi")
    img, _ = make_image(13, None)
    body = submit(client, "ORD-2005", "changed my mind", cv2.GaussianBlur(img, (0, 0), 8)).json()
    assert body["status"] == "Photos needed" and "blurry" in body["request"]
    files = [("images", ("new.png", png(make_image(14, None)[0]), "image/png"))]
    again = client.post(f"/api/cases/{body['id']}/evidence", files=files).json()
    assert again["evidence_rounds"] == 1 and again["status"] == "Approved", again


def test_reused_photo_is_flagged_and_reviewer_flow(client):
    login(client, "neha")
    img, _ = make_image(15, None)
    first = submit(client, "ORD-2009", "changed my mind", img).json()
    second = submit(client, "ORD-2007", "changed my mind", img).json()
    assert first["status"] == "Approved" and second["status"] == "With reviewer"

    login(client, "reviewer", "review1234", "reviewer")
    q = client.get("/api/review/queue?filter=flagged").json()
    row = next(r for r in q["cases"] if r["id"] == second["id"])
    assert row["reason"] == "Photo integrity flag" and row["priority"] == "High"
    pack = client.get(f"/api/cases/{second['id']}").json()
    assert pack["tool_calls"] and pack["preanalysis"]["integrity_flags"]
    assert client.post(f"/api/cases/{second['id']}/review", json={"action": "reject", "reason_code": "APPROVE_ME"}).status_code == 422
    r = client.post(f"/api/cases/{second['id']}/review", json={"action": "reject", "reason_code": "DIFFERENT_ITEM", "note": "same photo as before"})
    assert r.json()["status"] == "Rejected"

    login(client, "neha")
    appealed = client.post(f"/api/cases/{second['id']}/appeal", json={"note": "it is a different dress"}).json()
    assert appealed["status"] == "With reviewer"


def test_late_return_skips_photo_checks_and_can_be_appealed(client):
    login(client, "asha")
    body = submit(client, "ORD-2003", "the jeans have a hole", make_image(21, None)[0]).json()
    assert body["status"] == "Rejected" and "appeal" in body["message"]
    assert [s["step"] for s in body["steps"]] == ["get_order", "retrieve_policy", "decide", "finalize"]
    assert client.post(f"/api/cases/{body['id']}/appeal", json={}).json()["status"] == "With reviewer"


def test_upload_validation(client):
    login(client, "asha")
    img, _ = make_image(4, None)
    files = [("images", ("a.txt", png(img), "text/plain"))]
    assert client.post("/api/cases", data={"order_id": "ORD-2001", "reason": "x"}, files=files).status_code == 415
    bad = [("images", ("a.png", b"not an image", "image/png"))]
    assert client.post("/api/cases", data={"order_id": "ORD-2001", "reason": "x"}, files=bad).status_code == 422
    assert client.post("/api/cases", data={"order_id": "ORD-2001", "reason": "  "}, files=bad).status_code == 422
    too_many = [("images", (f"{i}.png", png(img), "image/png")) for i in range(config.MAX_PHOTOS + 1)]
    assert client.post("/api/cases", data={"order_id": "ORD-2001", "reason": "x"}, files=too_many).status_code == 422
    assert client.get("/api/cases/99999").status_code == 404
