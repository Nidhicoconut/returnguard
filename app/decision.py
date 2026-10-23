"""Decision engine: the LLM proposes, deterministic policy rules dispose.

* `policy_verdict` encodes the policy as code. It is the source of truth for hard rules
  (return window, non-returnable items, missing order, high-value refunds, missing evidence).
* `finalize` takes the LLM's proposal and either keeps its wording (when it agrees with the rules),
  or falls back to the rules / escalates to a human (when it does not). The LLM can never approve
  something the rules would not, and it can only cite clauses that really exist.
"""
import re

from . import config, policy

DEFECT_WORDS = re.compile(
    r"broke|broken|crack|damag|defect|scratch|stain|torn|\btear|ripp|dent|not work|stopped work|faulty|leak|burn|chip",
    re.I,
)
STATUS = {"approve": "Approved", "reject": "Rejected", "escalate": "Needs human review"}


def claims_defect(reason: str) -> bool:
    return bool(DEFECT_WORDS.search(reason or ""))


def policy_verdict(f: dict) -> dict:
    """Return {decision, resolution, cited, reasons, hard, confidence} from facts alone."""
    order, v = f.get("order"), f.get("vision") or {}
    window = f["return_window_days"]

    def out(decision, cited, reasons, hard, resolution=None, confidence=0.95):
        return {"decision": decision, "resolution": resolution, "cited": cited, "reasons": reasons,
                "hard": hard, "confidence": confidence}

    if not order:
        return out("escalate", ["R7"], ["No order matches that ID, so the purchase cannot be verified."], True)

    days = f["days_since_delivery"]
    if days > window:
        return out("reject", ["R1"], [f"Delivered {days} days ago; the return window is {window} days."], True)

    category = (order.get("category") or "").lower()
    label = v.get("label", "inconclusive")
    defect_claim = f["claims_defect"]

    if category in config.NON_RETURNABLE_CATEGORIES:
        if defect_claim or label == "defective":
            return out("escalate", ["R4", "R7"], [f"{order['category']} items need human review for damage claims."], True)
        return out("reject", ["R4"], [f"{order['category']} items cannot be returned for change of mind."], True)

    if label == "inconclusive" or v.get("confidence", 0) < 0.5:
        return out("escalate", ["R5"], ["The photo does not give clear enough evidence to decide automatically."], True,
                   confidence=v.get("confidence", 0.0))

    high_value = order["price"] > config.HUMAN_APPROVAL_ABOVE
    conf = float(v.get("confidence", 0.8))

    if defect_claim and label == "defective":
        if high_value:
            return out("escalate", ["R2", "R7"], [f"Defect confirmed in the photo ({v.get('defect_type')}), but the order is above "
                                                  f"{config.HUMAN_APPROVAL_ABOVE:.0f} rupees, so a human approves the refund."], True, confidence=conf)
        return out("approve", ["R2", "R1"], [f"Customer reports a defect and the photo confirms it ({v.get('defect_type')}).",
                                             f"Delivered {days} days ago, inside the {window}-day window."], False,
                   "full_refund_or_replacement", conf)
    if defect_claim and label == "intact":
        return out("escalate", ["R6"], ["Customer reports a defect but the photo shows an undamaged item."], False, confidence=conf)
    if not defect_claim and label == "defective":
        return out("escalate", ["R6"], [f"Photo shows damage ({v.get('defect_type')}) that the customer did not mention."], False, confidence=conf)
    if high_value:
        return out("escalate", ["R3", "R7"], ["Change-of-mind return above the human-approval limit."], True, confidence=conf)
    return out("approve", ["R3", "R1"], ["Change-of-mind return inside the window; the item looks undamaged.",
                                         "Resolution is store credit, not a cash refund."], False, "store_credit", conf)


def _customer_message(decision: str, resolution: str | None, reasons: list[str]) -> str:
    if decision == "approve":
        what = "a full refund or replacement" if resolution == "full_refund_or_replacement" else "store credit"
        return f"Good news: your return is approved and you will receive {what}."
    if decision == "reject":
        return "We could not approve this return. " + reasons[0] + " A store manager can review exceptions."
    return "Thanks for your request. A member of our team will review your photo and get back to you."


def finalize(facts: dict, proposal: dict | None, provider: str) -> dict:
    """Merge the LLM proposal with the rule verdict into the final, validated decision."""
    exp = policy_verdict(facts)
    valid_ids = policy.clause_ids()
    notes = []
    final = {"decision": exp["decision"], "resolution": exp["resolution"], "confidence": exp["confidence"],
             "reasons": list(exp["reasons"]), "cited_clauses": list(exp["cited"]), "source": "rules", "agreement": None}
    final["summary"] = " ".join(exp["reasons"])
    final["customer_message"] = _customer_message(exp["decision"], exp["resolution"], exp["reasons"])

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
            final["customer_message"] = proposal.get("customer_message") or final["customer_message"]
            final["cited_clauses"] = sorted(set(cites) | set(exp["cited"]))
        elif not exp["hard"]:
            final.update(decision="escalate", resolution=None, source="guardrail")
            final["reasons"] = exp["reasons"] + [
                f"The language model proposed '{proposal.get('decision')}', which disagrees with the policy check, so a human decides."]
            final["cited_clauses"] = sorted(set(exp["cited"]) | {"R7"})
            final["summary"] = " ".join(final["reasons"])
            final["customer_message"] = _customer_message("escalate", None, final["reasons"])
            notes.append("LLM and policy engine disagreed on a soft rule; escalated.")
        else:
            notes.append(f"LLM proposed '{proposal.get('decision')}' but a hard policy rule applies; rule verdict used.")

    final["status"] = STATUS[final["decision"]]
    final["notes"] = notes
    return final
