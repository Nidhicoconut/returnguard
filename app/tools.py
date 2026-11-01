"""Tools the agent can call (numbered as in the project plan). The same functions are exposed over MCP
by app/mcp_server.py. Every tool takes plain arguments and returns a JSON-serialisable dict."""
import datetime as dt
import threading

import cv2

from . import config, db, llm, policy, preanalysis, vision, vlm

_VISION_LOCK = threading.Lock()  # one model run at a time; cases may be processed in parallel threads


def lookup_order(order_id: str) -> dict | None:
    """T0: find an order and compute how many days ago it was delivered."""
    order = db.get_order(order_id or "")
    if not order:
        return None
    delivered = dt.date.fromisoformat(order["delivered_date"])
    order["days_since_delivery"] = (dt.date.today() - delivered).days
    order["window_ok"] = order["days_since_delivery"] <= config.RETURN_WINDOW_DAYS
    return order


def parse_claim(reason: str) -> dict:
    """Turn the customer's text into a structured claim (LLM, schema-validated; keyword fallback)."""
    claim, source = llm.parse_claim(reason)
    return {"claim": claim, "source": source}


def assess_photos(case_id: int, delivered_date: str | None) -> dict:
    """T1 + T7: quality gate and integrity checks for every photo of the case not yet checked."""
    photos, instructions, flags = [], [], []
    media = db.media_for(case_id)
    latest_round = max((m["round"] for m in media), default=0)
    for m in media:
        if m["quality"] is None:
            img = cv2.imread(str(config.UPLOAD_DIR / m["file"]), cv2.IMREAD_COLOR)
            quality = preanalysis.assess_quality(img)
            integrity = preanalysis.check_integrity(case_id, m["sha256"], m["dhash"], (m["integrity"] or {}).get("exif", {}), delivered_date)
            db.update_media(m["id"], quality=quality, integrity=integrity)
            m["quality"], m["integrity"] = quality, integrity
        photos.append({"media_id": m["id"], "file": m["file"], "round": m["round"], "ok": m["quality"]["ok"],
                       "problems": m["quality"]["problems"], "sharpness": m["quality"]["sharpness"],
                       "flags": m["integrity"]["flags"]})
        if m["round"] == latest_round:  # instructions only for the latest photos
            instructions += [i for i in m["quality"]["instructions"] if i not in instructions]
        flags += [f for f in m["integrity"]["flags"] if f not in flags]
    ok = [p for p in photos if p["ok"]]
    return {"photos": photos, "ok_photos": len(ok), "total": len(photos), "instructions": instructions, "integrity_flags": flags}


def detect_defects(photos: list[dict], tiles: bool = False) -> dict:
    """T4: run the clothing defect models on each acceptable photo; the most damaged one is the evidence."""
    results = []
    for p in photos:
        with _VISION_LOCK:
            r = vision.analyze(config.UPLOAD_DIR / p["file"], category="clothing", tiles=tiles)
        results.append({"media_id": p["media_id"], "file": p["file"], **r})
    # Most damaged photo first; a clean photo of the garment outranks one where no garment was found.
    best = max(results, key=lambda r: (vision.VERDICT_RANK.get(r["verdict"], 0), r.get("damage_score", 0))) if results else None
    return {"best": best, "per_photo": results, "mode": "tiles" if tiles else "full"}


def verify_with_vlm(file: str, regions: list[dict], claimed_class: str | None) -> dict:
    """T9: blind check of the top marked regions (or the whole photo when none) by a vision-language model."""
    if not vlm.available():
        return {"status": "unavailable", "confirms": None, "observations": [],
                "note": "No GEMINI_API_KEY set, so the independent vision check did not run."}
    img = cv2.imread(str(config.UPLOAD_DIR / file), cv2.IMREAD_COLOR)
    targets = regions[: config.VLM_MAX_CALLS - 1] or [None]
    observations, errors = [], []
    for region in targets:
        try:
            observations.append({"region": region, **vlm.describe(img, region)})
        except Exception as exc:  # quota, network, unreadable reply
            errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        if observations[-1]["damage_visible"]:
            break  # confirmed; more calls would only use up the free-tier quota
    if not observations:
        return {"status": "failed", "confirms": None, "observations": [], "errors": errors}
    return {**vlm.compare(observations, claimed_class), "observations": observations, "errors": errors}


def retrieve_policy(query: str, k: int = 3, must_include=("R1",)) -> list[dict]:
    """T8: retrieve the policy clauses most relevant to the query."""
    return policy.retrieve(query, k=k, must_include=must_include)


def create_ticket(case_id: int, decision: str, summary: str) -> dict:
    """T10: open a ticket in the right queue and attach it to the case."""
    queue = {"AUTO_APPROVE": "refunds", "REQUEST_MORE_EVIDENCE": "awaiting-customer", "AUTO_REJECT": "customer-care",
             "ESCALATE_TO_HUMAN": "human-review"}.get(decision, "human-review")
    ticket_id = f"TKT-{case_id:05d}"
    db.update_case(case_id, ticket_id=ticket_id)
    db.add_event(case_id, "agent", "ticket_created", {"ticket_id": ticket_id, "queue": queue, "summary": summary})
    return {"ticket_id": ticket_id, "queue": queue}


TOOLS = {
    "lookup_order": lookup_order,
    "parse_claim": parse_claim,
    "assess_photos": assess_photos,
    "detect_defects": detect_defects,
    "verify_with_vlm": verify_with_vlm,
    "retrieve_policy": retrieve_policy,
    "create_ticket": create_ticket,
}
