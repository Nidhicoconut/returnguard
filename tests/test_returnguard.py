import itertools

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, db, decision, llm, policy, preanalysis, product
from app.main import app
from make_synthetic_data import make_image

APPROVE, MORE, ESCALATE, REJECT = decision.APPROVE, decision.MORE, decision.ESCALATE, decision.REJECT


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        # A late order for the return-window tests (the seeded demo orders are all inside the window).
        with db.conn() as con:
            uid = con.execute("SELECT id FROM users WHERE username = 'asha'").fetchone()[0]
            con.execute("INSERT OR IGNORE INTO orders (order_id, user_id, customer, product, product_type, category, price, "
                        "delivered_date, sku, colour) VALUES ('ORD-9001', ?, 'Asha Menon', 'Slim Fit Jeans', 'jeans', 'Clothing', "
                        "1999, date('now', '-25 days'), 'TEST-JEANS', 'blue')", (uid,))
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
def test_policy_has_nine_clothing_clauses():
    assert policy.clause_ids() == {f"R{i}" for i in range(1, 10)}
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
    assert {o["order_id"] for o in client.get("/api/orders").json()} == {"ORD-3001", "ORD-3002", "ORD-9001"}
    assert submit(client, "ORD-3003", "hole", make_image(1, None)[0]).status_code == 404  # Ravi's order


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
    body = submit(client, "ORD-3002", "I changed my mind, it does not fit", img).json()
    assert body["status"] == "Approved" and body["resolution"] == "store_credit", body
    assert [s["step"] for s in body["steps"]][:3] == ["get_order", "parse_claim", "check_photos"]
    url = body["photos"][0]["url"]
    assert client.get(url).status_code == 200
    login(client, "ravi")
    assert client.get(url).status_code == 404 and client.get(f"/api/cases/{body['id']}").status_code == 404


def test_damage_claim_goes_to_reviewer_without_vision_check(client):
    login(client, "asha")
    img, _ = make_image(12, "stain")
    body = submit(client, "ORD-3001", "there is a big stain on the front", img).json()
    assert body["status"] == "With reviewer", body


def test_blurry_photo_asks_for_more_then_rechecks(client):
    login(client, "ravi")
    img, _ = make_image(13, None)
    body = submit(client, "ORD-3004", "changed my mind", cv2.GaussianBlur(img, (0, 0), 8)).json()
    assert body["status"] == "Photos needed" and "blurry" in body["request"]
    files = [("images", ("new.png", png(make_image(14, None)[0]), "image/png"))]
    again = client.post(f"/api/cases/{body['id']}/evidence", files=files).json()
    assert again["evidence_rounds"] == 1 and again["status"] == "Approved", again


def test_reused_photo_is_flagged_and_reviewer_flow(client):
    login(client, "neha")
    img, _ = make_image(15, None)
    first = submit(client, "ORD-3005", "changed my mind", img).json()
    second = submit(client, "ORD-3006", "changed my mind", img).json()
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
    body = submit(client, "ORD-9001", "the jeans have a hole", make_image(21, None)[0]).json()
    assert body["status"] == "Rejected" and "appeal" in body["message"]
    assert [s["step"] for s in body["steps"]] == ["get_order", "retrieve_policy", "decide", "finalize"]
    assert client.post(f"/api/cases/{body['id']}/appeal", json={}).json()["status"] == "With reviewer"


def test_product_verdict_needs_a_confident_no_and_no_yes():
    assert product.verdict([{"visible": "yes"}, {"visible": "no", "confidence": 0.9}])["status"] == "match"
    assert product.verdict([{"visible": "no", "confidence": 0.9}, {"visible": "unclear"}])["status"] == "mismatch"
    assert product.verdict([{"visible": "no", "confidence": 0.4}])["status"] == "unverified"  # not sure enough
    assert product.verdict([{"visible": "unclear"}])["status"] == "unverified"                 # close-up of fabric
    assert product.verdict([{"catalog_sim": 0.45}])["status"] == "unverified"  # low similarity is not evidence
    assert product.verdict([{"catalog_sim": 0.95, "visible": "unclear"}])["status"] == "match"


def test_product_mismatch_asks_for_a_photo_then_escalates():
    pc = {"status": "mismatch", "seen": ["blue jeans"]}
    r = verdict(product_check=pc, detection=None)
    assert r["decision"] == MORE and r["reason_code"] == "PRODUCT_MISMATCH" and "Cotton T-Shirt" in r["customer_message"]
    r = verdict(product_check=pc, detection=None, evidence_rounds_used=2)
    assert r["decision"] == ESCALATE and r["reason_code"] == "PRODUCT_MISMATCH"


def test_refund_needs_the_item_identified_in_some_photo():
    unverified = {"status": "unverified"}
    r = verdict(product_check=unverified)  # detector + vision check confirm damage, but only close-ups
    assert r["decision"] == MORE and r["reason_code"] == "PRODUCT_UNVERIFIED" and "whole" in r["customer_message"]
    assert verdict(product_check=unverified, evidence_rounds_used=2)["decision"] == ESCALATE
    assert verdict(product_check={"status": "match"})["decision"] == APPROVE
    assert verdict(product_check={"status": "not_run"})["decision"] == APPROVE  # no Gemini key: gate not applied


def test_wrong_item_photo_end_to_end(client, monkeypatch):
    """Gemini (faked) says the first photo shows jeans, not the ordered T-shirt; the second photo matches."""
    answers = iter([[{"visible": "no", "seen": "blue jeans", "confidence": 0.95}],
                    [{"visible": "no", "seen": "blue jeans", "confidence": 0.95}, {"visible": "yes", "seen": "white t-shirt", "confidence": 0.9}]])
    monkeypatch.setattr(product, "available", lambda: True)
    monkeypatch.setattr(product, "ask_gemini", lambda imgs, order: next(answers))
    login(client, "asha")
    body = submit(client, "ORD-3002", "changed my mind", make_image(31, None)[0]).json()
    assert body["status"] == "Photos needed" and "Ribbed Knit Tee" in body["request"]
    assert "detect_defects" not in [s["step"] for s in body["steps"]]  # wrong item: damage check skipped
    files = [("images", ("right.png", png(make_image(32, None)[0]), "image/png"))]
    again = client.post(f"/api/cases/{body['id']}/evidence", files=files).json()
    assert again["status"] == "Approved" and any("match your SheIn Mock Neck Ribbed Knit Tee" in k["text"] for k in again["checks"]), again


def test_upload_validation(client):
    login(client, "asha")
    img, _ = make_image(4, None)
    files = [("images", ("a.txt", png(img), "text/plain"))]
    assert client.post("/api/cases", data={"order_id": "ORD-3001", "reason": "x"}, files=files).status_code == 415
    bad = [("images", ("a.png", b"not an image", "image/png"))]
    assert client.post("/api/cases", data={"order_id": "ORD-3001", "reason": "x"}, files=bad).status_code == 422
    assert client.post("/api/cases", data={"order_id": "ORD-3001", "reason": "  "}, files=bad).status_code == 422
    too_many = [("images", (f"{i}.png", png(img), "image/png")) for i in range(config.MAX_PHOTOS + 1)]
    assert client.post("/api/cases", data={"order_id": "ORD-3001", "reason": "x"}, files=too_many).status_code == 422
    assert client.get("/api/cases/99999").status_code == 404


def test_gemini_outage_retries_with_the_fallback_model_then_pauses(monkeypatch):
    import httpx
    from app import vlm
    calls = []

    class Reply:
        def __init__(self, status, model=""):
            self.status_code, self.headers, self.model = status, {}, model
        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=None)
        def json(self):
            return {"choices": [{"message": {"content": self.model}}]}

    def fake_post(url, json, headers, timeout):
        calls.append(json["model"])
        return Reply(503) if json["model"] == "primary" else Reply(200, json["model"])

    monkeypatch.setattr(config, "VLM_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(vlm, "_down_until", 0.0)
    monkeypatch.setattr(vlm.httpx, "post", fake_post)
    assert vlm.chat({"model": "primary"})["choices"][0]["message"]["content"] == "backup"
    assert calls == ["primary", "backup"]

    monkeypatch.setattr(vlm.httpx, "post", lambda *a, **k: Reply(503))
    with pytest.raises(httpx.HTTPStatusError):
        vlm.chat({"model": "primary"})
    with pytest.raises(RuntimeError, match="skipped"):  # both failed: Gemini is skipped for a minute
        vlm.chat({"model": "primary"})


def test_customer_is_not_told_damage_was_found_when_the_second_check_disagrees(client):
    from app.main import _checks_for_customer
    case = {"order_id": "ORD-3001", "status": "With reviewer", "preanalysis": None, "decision_detail": {},
            "vision": {"verdict": "damaged", "defect_type": "stain"}, "vlm": {"status": "no_damage_seen"}}
    texts = [k["text"] for k in _checks_for_customer(case)]
    assert not any("Damage found" in t for t in texts) and "Our team will check the damage in your photos" in texts
    case["vlm"] = {"status": "damage_seen"}
    assert any("Damage found" in k["text"] for k in _checks_for_customer(case))


def _courier_job(client, case_id):
    login(client, "courier", "pickup1234", None)
    job = next(j for j in client.get("/api/pickups").json() if j["case_id"] == case_id)
    return job, client.get(f"/api/pickups/{job['id']}").json()


def test_approved_return_is_picked_up_and_checked_before_the_refund(client):
    login(client, "asha")
    img, _ = make_image(41, None)
    body = submit(client, "ORD-3002", "I changed my mind, it does not fit", img).json()
    assert body["status"] == "Approved" and body["display_status"] == "Pickup scheduled", body
    code = body["pickup"]["code"]
    assert len(code) == 4 and body["pickup"]["refund_amount"] is None
    assert client.get("/api/pickups").status_code == 403  # customers can't use the pickup partner's API

    job, detail = _courier_job(client, body["id"])
    assert detail["address"] and detail["customer"] == "Asha Menon" and "code" not in detail
    keys = [k["key"] for k in detail["checklist"]]
    assert "tags" in detail["checklist"][1]["text"]  # change of mind: unworn with tags
    url = f"/api/pickups/{job['id']}/complete"
    assert client.post(url, data={"code": "0000" if code != "0000" else "1111", "checks": ",".join(keys)}).status_code == 422
    assert client.post(url, data={"code": code, "checks": "item"}).status_code == 422  # every check must be ticked
    done = client.post(url, data={"code": code, "checks": ",".join(keys), "note": "Tags on, folded"},
                       files={"photo": ("door.png", png(img), "image/png")}).json()
    assert done["status"] == "picked_up" and done["photo"]
    assert client.post(url, data={"code": code, "checks": ",".join(keys)}).status_code == 422  # no second pickup

    login(client, "asha")
    c = client.get(f"/api/cases/{body['id']}").json()
    assert c["display_status"] == "Credit issued" and c["pickup"]["refund_amount"] == 899 and c["pickup"]["code"] is None
    assert "checked it at your door" in c["message"]
    login(client, "reviewer", "review1234", "reviewer")
    r = client.get(f"/api/cases/{body['id']}").json()
    assert r["pickup"]["status"] == "picked_up" and all(k["ok"] for k in r["pickup"]["checks"]) and r["pickup"]["note"] == "Tags on, folded"
    assert client.get(r["pickup"]["photo"]).status_code == 200


def test_failed_doorstep_check_goes_back_to_a_reviewer(client):
    login(client, "asha")
    img, _ = make_image(42, None)
    body = submit(client, "ORD-3001", "I changed my mind, the colour is wrong for me", img).json()
    assert body["display_status"] == "Pickup scheduled"
    job, _ = _courier_job(client, body["id"])
    fail = f"/api/pickups/{job['id']}/fail"
    first = client.post(fail, json={"reason": "not_home"}).json()
    assert first["outcome"] == "rescheduled" and first["attempts"] == 2 and first["scheduled_for"] > job["scheduled_for"]
    second = client.post(fail, json={"reason": "wrong_item", "note": "Customer handed over a different jumper", "checks": ["complete"]}).json()
    assert second["outcome"] == "failed"

    login(client, "asha")
    c = client.get(f"/api/cases/{body['id']}").json()
    assert c["status"] == "With reviewer" and "didn't pass the check at your door" in c["message"]
    login(client, "reviewer", "review1234", "reviewer")
    row = next(r for r in client.get("/api/review/queue?filter=escalated").json()["cases"] if r["id"] == body["id"])
    assert row["reason_code"] == "PICKUP_CHECK_FAILED" and row["priority"] == "High"
    r = client.post(f"/api/cases/{body['id']}/review", json={"action": "approve", "reason_code": "GOODWILL", "note": "Spoke to customer"}).json()
    assert r["pickup"]["status"] == "scheduled" and r["pickup"]["attempts"] == 1  # approving books a new pickup


def test_demo_reset_clears_returns_and_photos(client, monkeypatch):  # keep last: it empties the shared database
    login(client, "asha")
    img, _ = make_image(12, None)
    body = submit(client, "ORD-3002", "I changed my mind, it does not fit", img).json()
    photo = config.UPLOAD_DIR / body["photos"][0]["url"].rsplit("/", 1)[-1]
    assert photo.is_file()
    monkeypatch.setattr(config, "SHOW_DEMO_ACCOUNTS", False)
    assert client.post("/api/demo/reset").status_code == 404
    monkeypatch.setattr(config, "SHOW_DEMO_ACCOUNTS", True)
    r = client.post("/api/demo/reset").json()
    assert r["cases"] >= 1 and r["photos"] >= 1 and not photo.exists()
    assert client.get("/api/cases").json() == []
    assert {o["order_id"] for o in client.get("/api/orders").json()} == {"ORD-3001", "ORD-3002"}
    assert submit(client, "ORD-3001", "I changed my mind", img).json()["id"] == 1  # case numbers start again
