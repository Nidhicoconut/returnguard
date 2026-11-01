"""Decision engine: the LLM proposes, deterministic policy rules dispose.

Outcomes follow the project plan: AUTO_APPROVE, REQUEST_MORE_EVIDENCE or ESCALATE_TO_HUMAN, plus one
exception chosen by the team: a return outside the return window is declined automatically (AUTO_REJECT,
config AUTO_REJECT_OUTSIDE_WINDOW). The customer can appeal, which sends it to a reviewer. Every other
rejection is made by a reviewer.

* `policy_verdict` encodes the policy as code and is the source of truth.
* `finalize` keeps the LLM's wording when it agrees with the rules. The LLM can only make the outcome
  more cautious: if it disagrees with an automatic approval, the case goes to a reviewer. It can only
  cite clauses that really exist.
"""
import re

from . import config, policy

APPROVE, MORE, ESCALATE, REJECT = "AUTO_APPROVE", "REQUEST_MORE_EVIDENCE", "ESCALATE_TO_HUMAN", "AUTO_REJECT"
STATUS = {APPROVE: "Approved", MORE: "Photos needed", ESCALATE: "With reviewer", REJECT: "Rejected"}
DEFECT_NAMES = {"tear_hole": "hole or tear", "stain_discoloration": "stain", "missing_part": "missing part"}

# Short labels shown in the reviewer queue's "Reason" column.
REASONS = {
    "NO_ORDER": "Order not found",
    "OUTSIDE_WINDOW": "Outside the return window",
    "NON_RETURNABLE": "Non-returnable item",
    "INTEGRITY_FLAG": "Photo integrity flag",
    "LOW_QUALITY": "Low image quality",
    "NO_GARMENT": "Garment not visible",
    "TOOL_FAILURE": "A check failed",
    "HIGH_VALUE": "High-value order",
    "DEFECT_CONFIRMED": "Defect confirmed",
    "UNCONFIRMED_DEFECT": "Defect not confirmed by second check",
    "DIFFERENT_DEFECT": "Claim partly contradicted",
    "WEAK_DETECTION": "Weak detection",
    "VLM_ONLY": "Only the vision check sees damage",
    "DAMAGE_NOT_VISIBLE": "Claimed damage not visible",
    "CLAIM_CONTRADICTED": "Claim not supported by photos",
    "UNMENTIONED_DAMAGE": "Damage not mentioned by customer",
    "UNCLEAR_PHOTO": "Unclear photo",
    "CHANGE_OF_MIND": "Change of mind, item undamaged",
    "LLM_DISAGREES": "Language model disagreed",
    "GUARDRAIL": "Agent limit reached",
}

# Reason codes a reviewer must pick for each action (the plan requires one for every human decision).
REVIEW_REASONS = {
    "approve": {"DEFECT_CONFIRMED": "Defect visible in the photos", "CHANGE_OF_MIND_OK": "Undamaged, change of mind accepted",
                "GOODWILL": "Goodwill exception", "OTHER": "Other (explain in the note)"},
    "reject": {"NO_DEFECT_VISIBLE": "No defect visible", "DIFFERENT_ITEM": "Photo shows a different item",
               "OUTSIDE_WINDOW": "Outside the return window", "NON_RETURNABLE": "Non-returnable item",
               "WEAR_DAMAGE": "Damage from wear or use", "OTHER": "Other (explain in the note)"},
    "request_info": {"NEED_CLOSE_UP": "Close-up of the damage needed", "NEED_WHOLE_GARMENT": "Photo of the whole garment needed",
                     "NEED_LABEL": "Photo of the label or tag needed", "OTHER": "Other (explain in the message)"},
}

DEFECT_WORDS = re.compile(
    r"broke|broken|damag|defect|stain|torn|\btear|ripp|\brips?\b|\bholes?\b|fray|faded|discolou?r|\bmarks?\b|missing button|split seam",
    re.I,
)


def claims_defect(reason: str, claim: dict | None = None) -> bool:
    return bool((claim or {}).get("defect_type")) or bool(DEFECT_WORDS.search(reason or ""))


def _out(decision, cited, reasons, code, recommendation="", request="", resolution=None, verdict="INSUFFICIENT",
         hard=True, confidence=0.95):
    return {"decision": decision, "cited": cited, "reasons": reasons, "reason_code": code, "recommendation": recommendation,
            "request": request, "resolution": resolution, "verdict": verdict, "hard": hard, "confidence": confidence}


def policy_verdict(f: dict) -> dict:
    """Return the rule outcome from the facts alone. First matching rule wins."""
    order = f.get("order")
    rounds_left = f.get("evidence_rounds_used", 0) < config.MAX_EVIDENCE_ROUNDS
    if not order:
        return _out(ESCALATE, ["R7"], ["No order matches that ID, so the purchase cannot be verified."], "NO_ORDER",
                    "Check the order number with the customer.")

    days, window = f["days_since_delivery"], f["return_window_days"]
    if days > window:
        if config.AUTO_REJECT_OUTSIDE_WINDOW:
            return _out(REJECT, ["R1"], [f"Delivered {days} days ago; the return window is {window} days."], "OUTSIDE_WINDOW",
                        "Declined automatically: outside the window. Grant a goodwill exception only on appeal.")
        return _out(ESCALATE, ["R1"], [f"Delivered {days} days ago; the return window is {window} days."], "OUTSIDE_WINDOW",
                    "Likely decline: outside the window, unless you grant a goodwill exception.")

    defect_claim = f["claims_defect"]
    if (order.get("product_type") or "").lower() in config.NON_RETURNABLE_TYPES:
        if defect_claim:
            return _out(ESCALATE, ["R4", "R7"], [f"{order['product']} is non-returnable; damage claims on it are reviewed by a person."],
                        "NON_RETURNABLE", "Check the damage; a defective non-returnable item can still be refunded.")
        return _out(ESCALATE, ["R4"], [f"{order['product']} cannot be returned for change of mind."], "NON_RETURNABLE",
                    "Likely decline: non-returnable item, change of mind.")

    if f.get("integrity_flags"):
        return _out(ESCALATE, ["R8"], ["Photo checks need a person to look: " + " ".join(f["integrity_flags"])], "INTEGRITY_FLAG",
                    "Compare the photos with the flagged case or details before deciding.")

    q, det = f.get("photo_quality") or {}, f.get("detection")
    if q.get("ok_photos", 0) == 0 or (det and det.get("verdict") == "no_garment"):
        code = "LOW_QUALITY" if q.get("ok_photos", 0) == 0 else "NO_GARMENT"
        ask = " ".join(q.get("instructions") or []) or "Take a photo of the whole garment laid flat, then a close-up of the damage."
        if rounds_left:
            return _out(MORE, ["R5"], ["The photos are not clear enough to check the garment."], code, request=ask)
        return _out(ESCALATE, ["R5"], ["The photos are still not clear enough after two requests."], code,
                    "Decide from the photos you have, or contact the customer.")

    if not det:
        return _out(ESCALATE, ["R7"], ["The photo check could not run."], "TOOL_FAILURE", "Check the photos yourself.")

    d, vc, conf = det["verdict"], f.get("vision_check") or {}, float(det.get("confidence", 0.8))
    high_value = order["price"] > config.AUTO_APPROVE_MAX_PRICE
    claimed = (f.get("claim") or {}).get("defect_type")
    found = DEFECT_NAMES.get({"hole": "tear_hole", "stain": "stain_discoloration"}.get(det.get("defect_type"), ""), det.get("defect_type") or "damage")

    if defect_claim:
        if d == "damaged":
            if vc.get("confirms") is False:
                return _out(ESCALATE, ["R6"], [f"The detector marked a {found}, but the independent vision check saw no damage."],
                            "UNCONFIRMED_DEFECT", "Look at the marked area: the two checks disagree.", confidence=conf)
            if vc.get("confirms") and vc.get("matches_claim") is False:
                return _out(ESCALATE, ["R6", "R2"], [f"Damage is visible ({', '.join(vc.get('types_seen', []))}), but not the kind the customer described."],
                            "DIFFERENT_DEFECT", "Claim partly contradicted: decide whether the visible damage justifies a refund.",
                            verdict="CONTRADICTED", confidence=conf)
            if not vc.get("confirms") and config.REQUIRE_VLM_FOR_APPROVAL:
                return _out(ESCALATE, ["R2", "R7"], [f"The detector found a {found}; the second check could not run, so a person confirms it."],
                            "UNCONFIRMED_DEFECT", "Likely approve if the marked damage is real.", verdict="SUPPORTED", confidence=conf)
            if high_value:
                return _out(ESCALATE, ["R2", "R7"], [f"Defect confirmed ({found}), but orders above ₹{config.AUTO_APPROVE_MAX_PRICE:,.0f} need a reviewer's approval."],
                            "HIGH_VALUE", "Likely approve: the defect is confirmed.", verdict="SUPPORTED", confidence=conf)
            return _out(APPROVE, ["R2", "R1"], [f"The customer reported a defect and the photos confirm it ({found}).",
                                                 f"Delivered {days} days ago, inside the {window}-day window."],
                        "DEFECT_CONFIRMED", resolution="full_refund_or_replacement", verdict="SUPPORTED", hard=False, confidence=conf)
        if vc.get("confirms"):
            return _out(ESCALATE, ["R2", "R6"], ["The vision check sees damage that the detector was not sure about."],
                        "WEAK_DETECTION" if d == "unsure" else "VLM_ONLY", "Likely approve if the damage is visible to you.", confidence=conf)
        if rounds_left:
            what = DEFECT_NAMES.get(claimed, "damage")
            return _out(MORE, ["R5"], [f"We could not see the {what} in the photos."], "UNCLEAR_PHOTO" if d == "unsure" else "DAMAGE_NOT_VISIBLE",
                        request=f"Please send a close-up of the {what}{' on the ' + f['claim']['location'] if (f.get('claim') or {}).get('location') else ''}, "
                                "in good light, with the damaged area filling most of the photo.", confidence=conf)
        return _out(ESCALATE, ["R6"], ["After two requests the photos still do not show the damage described."],
                    "CLAIM_CONTRADICTED", "Likely decline: no damage visible in any photo.", verdict="CONTRADICTED", confidence=conf)

    # No damage claimed: change of mind or fit
    if d == "damaged":
        return _out(ESCALATE, ["R6", "R3"], [f"The photos show damage ({found}) that the customer did not mention."],
                    "UNMENTIONED_DAMAGE", "Check whether the item was damaged before or after delivery.", confidence=conf)
    if d == "unsure":
        return _out(ESCALATE, ["R3", "R6"], ["The photos may show damage; a person should check before store credit is given."],
                    "UNCLEAR_PHOTO", "Approve for store credit if the garment looks undamaged.", confidence=conf)
    if high_value:
        return _out(ESCALATE, ["R3", "R7"], ["Change-of-mind return above the auto-approve limit."], "HIGH_VALUE",
                    "Likely approve for store credit.", confidence=conf)
    return _out(APPROVE, ["R3", "R1"], ["Change-of-mind return inside the window; the garment looks undamaged.",
                                         "Resolution is store credit, not a cash refund."],
                "CHANGE_OF_MIND", resolution="store_credit", verdict="SUPPORTED", hard=False, confidence=conf)


def customer_message(decision: str, resolution: str | None, request: str = "", reasons: list[str] | None = None) -> str:
    if decision == REJECT:
        return ("We can't accept this return because it is outside our "
                f"{config.RETURN_WINDOW_DAYS}-day return window. {(reasons or [''])[0]} "
                "If you think this is a mistake, you can appeal and a member of our team will take a look.")
    if decision == APPROVE:
        what = "a full refund or a replacement" if resolution == "full_refund_or_replacement" else "store credit"
        return f"Good news: your return is approved and you will receive {what}."
    if decision == MORE:
        return request or "Please send another photo of the item."
    return "Thanks. A member of our returns team will look at your photos, usually within one working day."


def finalize(facts: dict, proposal: dict | None, provider: str) -> dict:
    """Merge the LLM proposal with the rule verdict into the final, validated decision."""
    exp = policy_verdict(facts)
    valid_ids = policy.clause_ids()
    notes = []
    final = {"decision": exp["decision"], "resolution": exp["resolution"], "confidence": exp["confidence"],
             "reasons": list(exp["reasons"]), "cited_clauses": list(exp["cited"]), "source": "rules", "agreement": None,
             "reason_code": exp["reason_code"], "recommendation": exp["recommendation"], "request": exp["request"],
             "verdict": exp["verdict"]}
    final["summary"] = " ".join(exp["reasons"])
    final["customer_message"] = customer_message(exp["decision"], exp["resolution"], exp["request"], exp["reasons"])

    if proposal:
        agrees = proposal.get("decision") == exp["decision"]
        final["agreement"] = agrees
        cites = [c for c in (proposal.get("cited_clauses") or []) if c in valid_ids]
        dropped = [c for c in (proposal.get("cited_clauses") or []) if c not in valid_ids]
        if dropped:
            notes.append(f"Dropped citations to clauses that do not exist: {dropped}")
        if agrees:
            final["source"] = provider
            final["summary"] = proposal.get("summary") or final["summary"]
            final["reasons"] = proposal.get("reasons") or final["reasons"]
            if exp["decision"] not in (MORE, REJECT):  # keep the rule engine's own instruction and appeal note
                final["customer_message"] = proposal.get("customer_message") or final["customer_message"]
            final["cited_clauses"] = sorted(set(cites) | set(exp["cited"]))
        elif exp["decision"] == APPROVE:
            final.update(decision=ESCALATE, resolution=None, source="guardrail", reason_code="LLM_DISAGREES",
                         recommendation="The rules would approve; the language model proposed "
                                        f"'{proposal.get('decision')}'. Check the evidence.")
            final["reasons"] = exp["reasons"] + [
                f"The language model proposed '{proposal.get('decision')}', which disagrees with the policy check, so a person decides."]
            final["cited_clauses"] = sorted(set(exp["cited"]) | {"R7"})
            final["summary"] = " ".join(final["reasons"])
            final["customer_message"] = customer_message(ESCALATE, None)
            notes.append("LLM and policy engine disagreed on an automatic approval; escalated.")
        else:
            notes.append(f"LLM proposed '{proposal.get('decision')}', but the policy rule applies; rule outcome used.")

    final["status"] = STATUS[final["decision"]]
    final["notes"] = notes
    return final
