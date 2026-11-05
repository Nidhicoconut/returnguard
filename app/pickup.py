"""Doorstep pickup (policy R9): after a return is approved, a pickup partner collects the item and checks it at the door
against the order and the customer's photos. The customer confirms the handover with a 4-digit pickup code. The refund
is issued only after that check, as large online stores do. A failed check sends the case back to a reviewer."""
import datetime as dt
import secrets

from . import db, decision

SLOT = "10 am – 2 pm"
MAX_ATTEMPTS = 3  # visits when the customer is not at home, before a reviewer takes over
FAIL_REASONS = {
    "not_home": "The customer was not available",
    "wrong_item": "The item doesn't match the order",
    "condition": "The item's condition doesn't match the return",
    "incomplete": "Parts of the item are missing",
}


def _tomorrow(after: str | None = None) -> str:
    start = max(dt.date.today(), dt.date.fromisoformat(after)) if after else dt.date.today()
    return (start + dt.timedelta(days=1)).isoformat()


def _nice_date(day: str) -> str:
    d = dt.date.fromisoformat(day)
    return f"{d:%a} {d.day} {d:%b}"


def checklist(case: dict, order: dict) -> list[dict]:
    """What the pickup partner confirms at the door. Damage returns are checked against the photos; change-of-mind
    returns must be unworn with tags."""
    colour = f" in {order['colour']}" if order.get("colour") else ""
    damage = case.get("resolution") != "store_credit"
    return [
        {"key": "item", "text": f"It is the {order['product']}{colour}, as in the product photo"},
        {"key": "condition", "text": "The damage is the same as in the customer's photos" if damage
                                     else "Unworn and unwashed, with the original tags attached"},
        {"key": "complete", "text": "Nothing is missing (every piece of a set or pack)"},
    ]


def schedule(case_id: int, actor: str = "system"):
    """Book a pickup for an approved return. Keeps a pickup that is already booked or done."""
    case = db.get_case(case_id)
    if not case or case["status"] != decision.STATUS[decision.APPROVE]:
        return None
    current = db.pickup_for_case(case_id)
    if current and current["status"] in ("scheduled", "picked_up"):
        return current
    courier, day = db.first_courier(), _tomorrow()
    db.save_pickup(case_id, courier_id=courier["id"] if courier else None, status="scheduled", scheduled_for=day, slot=SLOT,
                   code=f"{secrets.randbelow(10000):04d}", attempts=1, checks=None, note=None, photo=None,
                   failure_reason=None, verified_at=None, refund_amount=None, refunded_at=None)
    db.add_event(case_id, actor, "pickup_scheduled", {"date": day, "slot": SLOT, "courier": courier["name"] if courier else None})
    return db.pickup_for_case(case_id)


def cancel(case_id: int, actor: str) -> None:
    """A reviewer changed an approval: call off a pickup that hasn't happened yet."""
    p = db.pickup_for_case(case_id)
    if p and p["status"] == "scheduled":
        db.update_pickup(p["id"], status="cancelled")
        db.add_event(case_id, actor, "pickup_cancelled", None)


def complete(p: dict, courier: dict, ticked: set[str], code: str, note: str = "", photo: str | None = None) -> dict:
    """The partner checked the item and the customer gave the right code: mark it collected and issue the refund."""
    if p["status"] != "scheduled":
        raise ValueError("This pickup is no longer open.")
    case = db.get_case(p["case_id"])
    order = db.get_order(case["order_id"])
    keys = [k["key"] for k in checklist(case, order)]
    if set(keys) - ticked:
        raise ValueError("Tick every check before confirming. If something doesn't match, choose “Can't collect” instead.")
    if code.strip() != p["code"]:
        raise ValueError("That pickup code is wrong. Ask the customer for the 4-digit code on their return page.")
    ts, credit = db.now(), case["resolution"] == "store_credit"
    amount = order["price"]
    db.update_pickup(p["id"], status="picked_up", checks={k: True for k in keys}, note=note.strip()[:500] or None, photo=photo,
                     verified_at=ts, refund_amount=amount, refunded_at=ts)
    where = "added to your account as store credit" if credit else "issued to your original payment method. It usually shows within 3 to 5 working days"
    db.update_case(case["id"], customer_message=f"We collected your {order['product']} and checked it at your door. "
                                                f"Your refund of ₹{amount:,.0f} has been {where}.")
    db.add_event(case["id"], courier["name"], "picked_up", {"checks": keys, "note": note.strip()[:500] or None, "photo": bool(photo)})
    db.add_event(case["id"], "system", "refund_issued", {"amount": amount, "method": "store credit" if credit else "original payment method"})
    return db.get_pickup(p["id"])


def fail(p: dict, courier: dict, reason: str, note: str = "", ticked: set[str] | None = None) -> str:
    """The item could not be collected. Not at home: try again the next day. Failed check: back to a reviewer."""
    if reason not in FAIL_REASONS:
        raise ValueError("Choose why the item could not be collected.")
    if p["status"] != "scheduled":
        raise ValueError("This pickup is no longer open.")
    case_id, note = p["case_id"], note.strip()[:500]
    if reason == "not_home" and p["attempts"] < MAX_ATTEMPTS:
        day = _tomorrow(p["scheduled_for"])
        db.update_pickup(p["id"], scheduled_for=day, attempts=p["attempts"] + 1, note=note or None)
        db.update_case(case_id, customer_message=f"Our pickup partner couldn't reach you. We'll try again on {_nice_date(day)}, {p['slot']}.")
        db.add_event(case_id, courier["name"], "pickup_missed", {"next_date": day, "note": note or None})
        return "rescheduled"
    case = db.get_case(case_id)
    db.update_pickup(p["id"], status="failed", failure_reason=reason, note=note or None,
                     checks={k: k in (ticked or set()) for k in ("item", "condition", "complete")})
    text = FAIL_REASONS[reason]
    detail = {**(case.get("decision_detail") or {}), "reason_code": "PICKUP_CHECK_FAILED", "pickup_failure": text}
    db.update_case(case_id, status=decision.STATUS[decision.ESCALATE], decision=decision.ESCALATE, decision_detail=detail,
                   recommendation=f"The pickup partner did not collect the item: {text.lower()}."
                                  + (f" Their note: “{note}”." if note else "")
                                  + " Compare the pickup notes with the customer's photos. Approving books a new pickup.",
                   customer_message="Our pickup partner couldn't collect your item"
                                    + (" because we couldn't reach you after several attempts." if reason == "not_home"
                                       else " because it didn't pass the check at your door.")
                                    + " A member of our returns team will look at it and update this page, usually within one working day.")
    db.add_event(case_id, courier["name"], "pickup_failed", {"reason": reason, "text": text, "note": note or None})
    return "failed"


def display_status(case: dict, p: dict | None) -> str:
    """The status shown to customers: an approved return moves on to 'Pickup scheduled' and then 'Refunded'."""
    if case["status"] == decision.STATUS[decision.APPROVE] and p:
        if p["status"] == "scheduled":
            return "Pickup scheduled"
        if p["status"] == "picked_up":
            return "Credit issued" if case.get("resolution") == "store_credit" else "Refunded"
    return case["status"]


def customer_info(case: dict, p: dict | None):
    if not p or p["status"] == "cancelled":
        return None
    return {"status": p["status"], "scheduled_for": p["scheduled_for"], "slot": p["slot"], "attempts": p["attempts"],
            "code": p["code"] if p["status"] == "scheduled" else None, "verified_at": p["verified_at"],
            "refund_amount": p["refund_amount"], "refunded_at": p["refunded_at"],
            "method": "store credit" if case.get("resolution") == "store_credit" else "original payment method",
            "failure": FAIL_REASONS.get(p["failure_reason"]) if p["status"] == "failed" else None}
