"""FastAPI backend: sign in, file a clothing return with photos, watch the agent check it, and review
escalated cases. Pages: /login, /customer (customers), /review (reviewers)."""
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import cv2
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from . import agent, auth, config, db, decision, llm, pickup, preanalysis, product, tools, vision, vlm
from .security import verify_password

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("returnguard")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="ReturnGuard", version="2.0", lifespan=lifespan)
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- pages
def _page(name: str):
    # no-cache: the browser re-checks each time, so an edited page or script is never served from a stale copy
    return FileResponse(config.STATIC_DIR / name, headers={"Cache-Control": "no-cache"})


HOME = {"customer": "/account", "reviewer": "/admin", "courier": "/pickup"}


@app.get("/")
def index(user=Depends(auth.current_user)):
    return RedirectResponse(HOME[user["role"]] if user else "/login")


@app.get("/login")
def login_page():
    return _page("login.html")


@app.get("/account")
@app.get("/customer")  # older link
def customer_page(user=Depends(auth.current_user)):
    if user and user["role"] == "customer":
        return _page("customer.html")
    return RedirectResponse(HOME[user["role"]] if user else "/login")


@app.get("/admin")
@app.get("/review")  # older link
def review_page(user=Depends(auth.current_user)):
    if user and user["role"] == "reviewer":
        return _page("review.html")
    return RedirectResponse(HOME[user["role"]] if user else "/login")


@app.get("/pickup")
def pickup_page(user=Depends(auth.current_user)):
    if user and user["role"] == "courier":
        return _page("pickup.html")
    return RedirectResponse(HOME[user["role"]] if user else "/login")


@app.get("/policy")
def policy_page():
    return _page("policy.html")


@app.get("/api/policy")
def policy_clauses():
    from . import policy
    return policy.all_clauses()


@app.get("/static/{name}")
def static_file(name: str):
    if name not in ("app.css", "app.js", "favicon.svg"):
        raise HTTPException(404, "Not found.")
    return _page(name)


# ---------------------------------------------------------------- accounts
class Login(BaseModel):
    username: str
    password: str
    role: Literal["customer", "reviewer", "courier"] | None = None


@app.post("/api/login")
def login(body: Login, response: Response):
    user = db.get_user_by_name(body.username)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Wrong username or password.")
    if body.role and user["role"] != body.role:
        raise HTTPException(403, f"This is a {user['role']} account. Choose “Continue as {user['role'].title()}”.")
    token = db.create_session(user["id"])
    response.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax", max_age=config.SESSION_DAYS * 86400)
    return {"id": user["id"], "username": user["username"], "name": user["name"], "role": user["role"]}


@app.post("/api/logout")
def logout(response: Response, rg_session: str | None = Cookie(default=None)):
    if rg_session:
        db.delete_session(rg_session)
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(auth.require_user)):
    return user


@app.get("/api/health")
def health():
    return {"ok": True, "vision_backend": "clothes-yolo" if vision.clothes_available() else vision.active_backend(),
            "llm_provider": llm.provider(), "vlm": vlm.available(), "face_blur": preanalysis.face_blur_available(),
            "return_window_days": config.RETURN_WINDOW_DAYS, "auto_approve_max_price": config.AUTO_APPROVE_MAX_PRICE}


@app.post("/api/demo/reset")
def demo_reset():
    # Only offered with the test accounts (SHOW_DEMO_ACCOUNTS), so a public deployment cannot be wiped.
    if not config.SHOW_DEMO_ACCOUNTS:
        raise HTTPException(404, "Not found.")
    out = db.reset_demo()
    log.info("demo reset: %(cases)d returns and %(photos)d photos removed", out)
    return out


@app.get("/api/config")
def public_config():
    demo = [{"username": u, "name": n, "password": p, "role": r} for u, n, r, p in db.DEMO_USERS] if config.SHOW_DEMO_ACCOUNTS else []
    return {"demo_accounts": demo, "return_window_days": config.RETURN_WINDOW_DAYS, "max_photos": config.MAX_PHOTOS,
            "max_evidence_rounds": config.MAX_EVIDENCE_ROUNDS}


# ---------------------------------------------------------------- photos
async def _read_photos(images: list[UploadFile]) -> list[tuple[bytes, object]]:
    images = [i for i in images if i and i.filename]
    if not images:
        raise HTTPException(422, "Add at least one photo of the garment.")
    if len(images) > config.MAX_PHOTOS:
        raise HTTPException(422, f"Add at most {config.MAX_PHOTOS} photos.")
    out = []
    for image in images:
        if image.content_type not in config.ALLOWED_IMAGE_TYPES:
            raise HTTPException(415, "Upload JPEG, PNG or WebP photos.")
        data = await image.read()
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Each photo must be smaller than 8 MB.")
        img = preanalysis.decode(data)
        if img is None:
            raise HTTPException(422, "One of the files could not be read as an image.")
        out.append((data, img))
    return out


def _store_photos(case_id: int, round_no: int, photos) -> None:
    for data, img in photos:
        filename = f"{uuid.uuid4().hex}.jpg"  # never trust the client's filename
        exif = preanalysis.read_exif(data)     # read before re-encoding, which drops all metadata
        cv2.imwrite(str(config.UPLOAD_DIR / filename), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = img.shape[:2]
        db.add_media(case_id, round_no, filename, preanalysis.sha256(data), preanalysis.dhash(img), w, h,
                     integrity={"exif": exif})


@app.post("/api/precheck")
async def precheck(image: UploadFile = File(...), user=Depends(auth.require_customer)):
    """Quality check of one photo before the return is submitted (nothing is stored)."""
    (_, img), = await _read_photos([image])
    return preanalysis.assess_quality(img)


def _courier_skus(courier_id: int) -> set:
    return {(db.get_order(db.get_case(p["case_id"])["order_id"]) or {}).get("sku") for p in db.list_pickups(courier_id)}


@app.get("/catalog/{sku}/{file}")
def catalog_photo(sku: str, file: str, user=Depends(auth.require_user)):
    allowed = (user["role"] == "reviewer" or (user["role"] == "courier" and sku in _courier_skus(user["id"]))
               or sku in {o["sku"] for o in db.list_orders(user["id"])})
    if not allowed:
        raise HTTPException(404, "Photo not found.")
    match = [p for p in product.catalog_photos(sku) if p.name == file]
    if not match:
        raise HTTPException(404, "Photo not found.")
    return FileResponse(match[0])


@app.get("/media/{file}")
def media(file: str, user=Depends(auth.require_user)):
    m = db.media_by_file(file) or db.pickup_by_photo(file)  # customer photos, or a pickup partner's doorstep photo
    if not m:
        raise HTTPException(404, "Photo not found.")
    case = db.get_case(m["case_id"])
    p = db.pickup_for_case(case["id"])
    allowed = (user["role"] == "reviewer" or case["user_id"] == user["id"]
               or (user["role"] == "courier" and p is not None and p["courier_id"] == user["id"]))
    if not allowed:
        raise HTTPException(404, "Photo not found.")
    return FileResponse(config.UPLOAD_DIR / file)


# ---------------------------------------------------------------- cases
def _start_agent(case_id: int) -> None:
    if config.AGENT_INLINE:
        agent.run_case(case_id)
    else:
        threading.Thread(target=agent.run_case, args=(case_id,), daemon=True).start()


def _checks_for_customer(c: dict) -> list[dict]:
    """The 'What we checked' list on the customer's case page: plain facts, no internal scores."""
    out, order = [], tools.lookup_order(c["order_id"])
    if order:
        out.append({"ok": True, "text": f"Order {order['order_id']}: {order['product']}"})
        out.append({"ok": order["window_ok"],
                    "text": f"{'Within' if order['window_ok'] else 'Outside'} the {config.RETURN_WINDOW_DAYS}-day return window"})
    pre = c.get("preanalysis") or {}
    if pre:
        bad = pre["total"] - pre["ok_photos"]
        out.append({"ok": bad == 0, "text": "Photos are clear" if bad == 0 else f"{bad} of {pre['total']} photos were not clear enough"})
    pc = (c.get("decision_detail") or {}).get("product_check") or {}
    if order and pc.get("status") == "match":
        out.append({"ok": True, "text": f"Photos match your {order['product']}"})
    elif order and pc.get("status") == "mismatch":
        out.append({"ok": False, "text": f"The photos don't look like your {order['product']}"})
    v = c.get("vision") or {}
    if v and c.get("status") != "Checking":
        claimed = (c.get("parsed_claim") or {}).get("defect_type")
        confirmed = (c.get("vlm") or {}).get("status") == "damage_seen"
        if v.get("verdict") == "damaged" and confirmed:
            out.append({"ok": True, "text": f"Damage found in your photo ({v.get('defect_type')})"})
        elif v.get("verdict") == "damaged":
            out.append({"ok": False, "text": "Our team will check the damage in your photos"})
        elif claimed:
            out.append({"ok": False, "text": "We could not see the damage you described"})
        elif v.get("verdict") == "clean":
            out.append({"ok": True, "text": "The garment looks undamaged"})
    return out


PROBLEM_TEXT = {"too_small": "too small", "blurry": "blurry", "too_dark": "too dark", "too_bright": "too bright"}


def _photo_status(c: dict | None, m: dict) -> dict | None:
    """How a photo was used, for the green / yellow / red marks on the customer and reviewer pages."""
    if not c or c.get("status") == "Checking" or m["quality"] is None:
        return None
    q, flags = m["quality"], (m["integrity"] or {}).get("flags") or []
    if not q["ok"]:
        return {"state": "bad", "text": "Not used: " + ", ".join(PROBLEM_TEXT.get(p, p) for p in q["problems"])}
    detail = c.get("decision_detail") or {}
    pp = next((p for p in (detail.get("product_check") or {}).get("per_photo", []) if p.get("media_id") == m["id"]), {})
    if pp.get("visible") == "no" and pp.get("confidence", 0) >= config.PRODUCT_MISMATCH_CONF:
        order = db.get_order(c["order_id"]) or {}
        return {"state": "bad", "text": f"Not used: doesn't look like your {order.get('product', 'order')}"}
    if flags:
        return {"state": "warn", "text": "Our team will check this photo"}
    v = c.get("vision") or {}
    confirmed = (c.get("vlm") or {}).get("status") == "damage_seen"
    if v.get("media_id") == m["id"] and v.get("verdict") == "damaged" and confirmed:
        return {"state": "ok", "text": "Damage found here"}
    if c.get("status") == decision.STATUS[decision.ESCALATE]:
        return None  # no verdict on single photos while a reviewer decides
    return {"state": "ok", "text": "Accepted"}


def _photo_list(case_id: int, c: dict | None = None):
    return [{"id": m["id"], "url": f"/media/{m['file']}", "round": m["round"], "quality": m["quality"],
             "integrity": {k: v for k, v in (m["integrity"] or {}).items() if k != "exif"}, "status": _photo_status(c, m)}
            for m in db.media_for(case_id)]


def _customer_view(c: dict) -> dict:
    reviews = db.reviews_for(c["id"])
    last = reviews[-1] if reviews else None
    p = db.pickup_for_case(c["id"])
    return {"id": c["id"], "order_id": c["order_id"], "reason": c["reason"], "status": c["status"],
            "display_status": pickup.display_status(c, p), "pickup": pickup.customer_info(c, p),
            "decision": c["decision"], "resolution": c["resolution"], "message": c["customer_message"],
            "request": c["request_message"], "evidence_rounds": c["evidence_rounds"], "max_evidence_rounds": config.MAX_EVIDENCE_ROUNDS,
            "decided_by": "reviewer" if last else ("agent" if c["decided_by"] == "agent" else c["decided_by"]),
            "review_note": last["note"] if last and last["action"] != "approve" else None,
            "steps": [{"step": s["step"], "summary": s["summary"]} for s in (c.get("trace") or [])],
            "checks": _checks_for_customer(c), "photos": _photo_list(c["id"], c),
            "created_at": c["created_at"], "updated_at": c["updated_at"], "ticket_id": c["ticket_id"],
            "product": (db.get_order(c["order_id"]) or {}).get("product")}


PRIORITY_HIGH = {"PICKUP_CHECK_FAILED", "PRODUCT_MISMATCH", "INTEGRITY_FLAG", "HIGH_VALUE", "CLAIM_CONTRADICTED", "DIFFERENT_DEFECT", "UNMENTIONED_DAMAGE", "GUARDRAIL", "APPEAL"}


def _priority(c: dict) -> str:
    code = (c.get("decision_detail") or {}).get("reason_code")
    if c["status"] in ("Approved", "Rejected"):
        return "Low"
    if code in PRIORITY_HIGH:
        return "High"
    return "Medium" if c["status"] == "With reviewer" else "Low"


def _reviewer_row(c: dict) -> dict:
    order = db.get_order(c["order_id"]) or {}
    code = (c.get("decision_detail") or {}).get("reason_code")
    photos = db.media_for(c["id"])
    return {"id": c["id"], "order_id": c["order_id"], "product": order.get("product"), "product_type": order.get("product_type"),
            "customer": order.get("customer"), "status": c["status"], "decision": c["decision"], "reason_code": code,
            "reason": decision.REASONS.get(code, code or "Checking"), "priority": _priority(c),
            "flagged": any((m["integrity"] or {}).get("flags") for m in photos),
            "display_status": pickup.display_status(c, db.pickup_for_case(c["id"])),
            "thumb": f"/media/{photos[0]['file']}" if photos else None, "created_at": c["created_at"], "updated_at": c["updated_at"]}


def _tidy_boxes(c: dict) -> dict:
    """Cases checked before box cleaning was added still hold raw boxes; tidy them for display."""
    def tidy(regions):
        return regions if not regions or "weak" in regions[0] else vision.clean_regions(regions)
    if c.get("vision"):
        c["vision"] = {**c["vision"], "regions": tidy(c["vision"].get("regions"))}
    detail = c.get("decision_detail") or {}
    if detail.get("per_photo"):
        c["decision_detail"] = {**detail, "per_photo": [{**p, "regions": tidy(p.get("regions"))} for p in detail["per_photo"]]}
    return c


def _reviewer_view(c: dict) -> dict:
    c = _tidy_boxes(c)
    order = tools.lookup_order(c["order_id"])
    catalog = [f"/catalog/{order['sku']}/{p.name}" for p in product.catalog_photos(order.get("sku"))] if order else []
    return {**c, "order": order, "catalog_photos": catalog, "photos": _photo_list(c["id"], c), "tool_calls": db.tool_calls_for(c["id"]),
            "pickup": _pickup_for_reviewer(c), "display_status": pickup.display_status(c, db.pickup_for_case(c["id"])),
            "reviews": db.reviews_for(c["id"]), "events": db.events_for(c["id"]), "priority": _priority(c),
            "reason_label": decision.REASONS.get((c.get("decision_detail") or {}).get("reason_code"), ""),
            "review_reasons": decision.REVIEW_REASONS}


def _pickup_for_reviewer(c: dict):
    p = db.pickup_for_case(c["id"])
    if not p:
        return None
    courier = db.get_user(p["courier_id"]) if p["courier_id"] else None
    labels = {k["key"]: k["text"] for k in pickup.checklist(c, db.get_order(c["order_id"]) or {"product": "item"})}
    return {"status": p["status"], "scheduled_for": p["scheduled_for"], "slot": p["slot"], "attempts": p["attempts"],
            "courier": courier["name"] if courier else None, "verified_at": p["verified_at"], "note": p["note"],
            "checks": [{"text": labels.get(k, k), "ok": v} for k, v in (p["checks"] or {}).items()],
            "photo": f"/media/{p['photo']}" if p["photo"] else None, "refund_amount": p["refund_amount"],
            "refunded_at": p["refunded_at"], "failure": pickup.FAIL_REASONS.get(p["failure_reason"]) if p["failure_reason"] else None}


def _own_case(case_id: int, user: dict) -> dict:
    c = db.get_case(case_id)
    if not c or (user["role"] != "reviewer" and c["user_id"] != user["id"]):
        raise HTTPException(404, "Case not found.")
    return c


@app.get("/api/orders")
def orders(user=Depends(auth.require_customer)):
    out = []
    for o in db.list_orders(user["id"]):
        full = tools.lookup_order(o["order_id"])
        photos = product.catalog_photos(full.get("sku"))
        cases = [c for c in db.list_cases(user_id=user["id"]) if c["order_id"] == o["order_id"]]
        out.append({**{k: full[k] for k in ("order_id", "product", "product_type", "colour", "price", "delivered_date",
                                            "days_since_delivery", "window_ok")},
                    "image": f"/catalog/{full['sku']}/{photos[0].name}" if photos else None,
                    "return_id": cases[0]["id"] if cases else None,
                    "return_status": pickup.display_status(cases[0], db.pickup_for_case(cases[0]["id"])) if cases else None})
    return out


@app.post("/api/cases")
async def create_case(order_id: str = Form(...), reason: str = Form(...), images: list[UploadFile] = File(...),
                      user=Depends(auth.require_customer)):
    order_id, reason = order_id.strip().upper(), reason.strip()
    if not order_id or not reason:
        raise HTTPException(422, "Choose an order and describe the problem.")
    if len(reason) > 1000:
        raise HTTPException(422, "The description is too long (1000 characters at most).")
    order = db.get_order(order_id)
    if not order or order["user_id"] != user["id"]:
        raise HTTPException(404, "That order is not on your account.")
    photos = await _read_photos(images)
    case_id = db.create_case(user["id"], order_id, reason)
    _store_photos(case_id, 0, photos)
    db.add_event(case_id, user["name"], "submitted", {"order_id": order_id, "reason": reason, "photos": len(photos)})
    _start_agent(case_id)
    return _customer_view(db.get_case(case_id))


@app.get("/api/cases")
def my_cases(user=Depends(auth.require_customer)):
    return [_customer_view(c) for c in db.list_cases(user_id=user["id"])]


@app.get("/api/cases/{case_id}")
def case_detail(case_id: int, user=Depends(auth.require_user)):
    c = _own_case(case_id, user)
    return _reviewer_view(c) if user["role"] == "reviewer" else _customer_view(c)


@app.post("/api/cases/{case_id}/evidence")
async def add_evidence(case_id: int, images: list[UploadFile] = File(...), user=Depends(auth.require_customer)):
    c = _own_case(case_id, user)
    if c["status"] != "Photos needed":
        raise HTTPException(409, "This return is not waiting for photos.")
    photos = await _read_photos(images)
    rounds = (c["evidence_rounds"] or 0) + 1
    _store_photos(case_id, rounds, photos)
    db.update_case(case_id, status="Checking", evidence_rounds=rounds, request_message=None)
    db.add_event(case_id, user["name"], "added_photos", {"round": rounds, "photos": len(photos)})
    _start_agent(case_id)
    return _customer_view(db.get_case(case_id))


class Appeal(BaseModel):
    note: str = ""


@app.post("/api/cases/{case_id}/appeal")
def appeal(case_id: int, body: Appeal, user=Depends(auth.require_customer)):
    c = _own_case(case_id, user)
    if c["status"] != "Rejected":
        raise HTTPException(409, "Only a declined return can be appealed.")
    detail = {**(c.get("decision_detail") or {}), "reason_code": "APPEAL"}
    db.update_case(case_id, status="With reviewer", decision=decision.ESCALATE, decision_detail=detail,
                   recommendation="The customer appealed a declined return. A different reviewer should take a second look.",
                   customer_message="Your appeal was received. A different member of our team will look at your return.")
    db.add_event(case_id, user["name"], "appealed", {"note": body.note[:500]})
    return _customer_view(db.get_case(case_id))


# ---------------------------------------------------------------- reviewer
@app.get("/api/review/queue")
def queue(filter: Literal["all", "escalated", "photos", "flagged", "decided"] = "all", user=Depends(auth.require_reviewer)):
    rows = [_reviewer_row(c) for c in db.list_cases(limit=300)]
    counts = {"total": len(rows), "escalated": sum(r["status"] == "With reviewer" for r in rows),
              "photos": sum(r["status"] == "Photos needed" for r in rows), "flagged": sum(r["flagged"] for r in rows),
              "decided": sum(r["status"] in ("Approved", "Rejected") for r in rows)}
    keep = {"all": lambda r: True, "escalated": lambda r: r["status"] == "With reviewer",
            "photos": lambda r: r["status"] == "Photos needed", "flagged": lambda r: r["flagged"],
            "decided": lambda r: r["status"] in ("Approved", "Rejected")}[filter]
    order = {"High": 0, "Medium": 1, "Low": 2}
    rows = sorted((r for r in rows if keep(r)), key=lambda r: (order[r["priority"]], -r["id"]))
    return {"counts": counts, "cases": rows}


class Review(BaseModel):
    action: Literal["approve", "reject", "request_info"]
    reason_code: str
    note: str = ""
    message: str = ""  # shown to the customer for request_info
    vision_feedback: Literal["correct", "false_positive", "missed_defect"] | None = None


@app.post("/api/cases/{case_id}/review")
def review(case_id: int, body: Review, user=Depends(auth.require_reviewer)):
    c = _own_case(case_id, user)
    if body.reason_code not in decision.REVIEW_REASONS[body.action]:
        raise HTTPException(422, "Choose a reason code for this decision.")
    if body.reason_code == "OTHER" and not (body.note.strip() or body.message.strip()):
        raise HTTPException(422, "Explain the reason in the note.")
    agent_decision = c["decision"]
    if body.action == "approve":
        resolution = c["resolution"] or ("store_credit" if body.reason_code == "CHANGE_OF_MIND_OK" else "full_refund_or_replacement")
        fields = dict(status="Approved", resolution=resolution,
                      customer_message=decision.customer_message(decision.APPROVE, resolution))
    elif body.action == "reject":
        fields = dict(status="Rejected", resolution=None,
                      customer_message="A member of our team reviewed your photos and could not approve this return: "
                                       + decision.REVIEW_REASONS["reject"][body.reason_code].lower() + ".")
    else:
        text = body.message.strip() or decision.REVIEW_REASONS["request_info"][body.reason_code] + "."
        fields = dict(status="Photos needed", request_message=text, customer_message=text)
    db.update_case(case_id, decided_by=user["name"], **fields)
    if body.action == "approve":
        pickup.schedule(case_id, user["name"])  # R9: collected and checked at the door before the refund
    else:
        pickup.cancel(case_id, user["name"])
    db.add_review(case_id, user["id"], body.action, body.reason_code, body.note.strip()[:1000], body.vision_feedback)
    db.add_event(case_id, user["name"], "review", {"action": body.action, "reason_code": body.reason_code, "note": body.note[:500],
                                                   "agent_decision": agent_decision, "vision_feedback": body.vision_feedback})
    return _reviewer_view(db.get_case(case_id))


# ---------------------------------------------------------------- pickup partner (R9)
def _assigned(pickup_id: int, user: dict) -> dict:
    p = db.get_pickup(pickup_id)
    if not p or p["courier_id"] != user["id"]:
        raise HTTPException(404, "Pickup not found.")
    return p


def _pickup_view(p: dict, full: bool = False) -> dict:
    c = _tidy_boxes(db.get_case(p["case_id"]))
    order = tools.lookup_order(c["order_id"]) or {}
    customer = db.get_user(c["user_id"]) or {}
    catalog = [f"/catalog/{order['sku']}/{f.name}" for f in product.catalog_photos(order.get("sku"))] if order else []
    out = {"id": p["id"], "case_id": c["id"], "status": p["status"], "scheduled_for": p["scheduled_for"], "slot": p["slot"],
           "attempts": p["attempts"], "customer": customer.get("name"), "address": customer.get("address"),
           "order_id": c["order_id"], "product": order.get("product"), "colour": order.get("colour"), "price": order.get("price"),
           "image": catalog[0] if catalog else None, "kind": "Store credit" if c["resolution"] == "store_credit" else "Refund",
           "verified_at": p["verified_at"], "failure": pickup.FAIL_REASONS.get(p["failure_reason"]) if p["failure_reason"] else None}
    if full:
        v = c.get("vision") or {}
        photos = []
        for m in db.media_for(c["id"]):
            st = _photo_status(c, m)
            if st and st["state"] == "bad":
                continue  # photos that weren't used (unclear, or a different item) would only confuse the check
            photos.append({"url": f"/media/{m['file']}", "regions": v.get("regions") if v.get("media_id") == m["id"] else [],
                           "damage": bool(st and st["text"] == "Damage found here")})
        out.update(reason=c["reason"], catalog_photos=catalog, photos=photos, checklist=pickup.checklist(c, order),
                   fail_reasons=pickup.FAIL_REASONS, note=p["note"], photo=f"/media/{p['photo']}" if p["photo"] else None,
                   max_attempts=pickup.MAX_ATTEMPTS)
    return out


@app.get("/api/pickups")
def my_pickups(user=Depends(auth.require_courier)):
    return [_pickup_view(p) for p in db.list_pickups(user["id"]) if p["status"] != "cancelled"]


@app.get("/api/pickups/{pickup_id}")
def pickup_detail(pickup_id: int, user=Depends(auth.require_courier)):
    return _pickup_view(_assigned(pickup_id, user), full=True)


@app.post("/api/pickups/{pickup_id}/complete")
async def pickup_complete(pickup_id: int, code: str = Form(...), checks: str = Form(""), note: str = Form(""),
                          photo: UploadFile | None = File(None), user=Depends(auth.require_courier)):
    p = _assigned(pickup_id, user)
    filename = None
    if photo is not None and photo.filename:
        (_, img), = await _read_photos([photo])
        filename = f"pickup_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(config.UPLOAD_DIR / filename), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    try:
        p = pickup.complete(p, user, {k for k in checks.split(",") if k}, code, note, filename)
    except ValueError as exc:
        if filename:
            (config.UPLOAD_DIR / filename).unlink(missing_ok=True)
        raise HTTPException(422, str(exc))
    return _pickup_view(p, full=True)


class PickupFailure(BaseModel):
    reason: str
    note: str = ""
    checks: list[str] = []


@app.post("/api/pickups/{pickup_id}/fail")
def pickup_fail(pickup_id: int, body: PickupFailure, user=Depends(auth.require_courier)):
    p = _assigned(pickup_id, user)
    try:
        outcome = pickup.fail(p, user, body.reason, body.note, set(body.checks))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"outcome": outcome, **_pickup_view(db.get_pickup(pickup_id), full=True)}
