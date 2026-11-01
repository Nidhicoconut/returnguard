"""LangGraph agent that processes one clothing return case.

    get_order ─(no order)──────────────────────────────────────────────────┐
        │                                                                 ▼
    parse_claim ─► check_photos ─(no usable photo)──────────────────► retrieve_policy ─► decide ─► finalize
                        │                                                 ▲
                        ▼                                                 │
                   detect_defects ─(weak or contradicts claim)─► zoom_in ─┤
                        │                                           │     │
                        └──────(needs a second opinion)─► verify_with_vlm ┘

The agent picks its next tool from the evidence so far: it skips detection when no photo is usable,
retries detection zoomed in when the first pass is unsure or disagrees with the claim, and only asks
the vision-language model when its answer can change the outcome. Guardrails from the project plan:
at most MAX_TOOL_CALLS tool calls and CASE_TIME_LIMIT_S seconds per case; past either, the case
goes to a reviewer. Every tool call is stored (tool_calls table) and shown live in the case trace.
"""
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import config, db, decision, llm, policy, tools, vlm


class State(TypedDict, total=False):
    case_id: int
    run_no: int
    started: float
    calls: int
    order_id: str
    reason: str
    evidence_rounds: int
    order: Any
    claim: Any
    photos: Any
    detection: Any
    vision_check: Any
    clauses: Any
    facts: Any
    result: Any
    guardrail: str
    trace: list


class Budget(Exception):
    pass


def _step(state: State, step: str, summary: str, t0: float, detail=None) -> list:
    entry = {"step": step, "summary": summary, "ms": int((time.time() - t0) * 1000), "run": state["run_no"]}
    trace = state.get("trace", []) + [entry]
    db.update_case(state["case_id"], trace=trace)  # the customer page polls this to show progress
    db.add_event(state["case_id"], "agent", step, {"summary": summary, **({"detail": detail} if detail else {})})
    return trace


def _call(state: State, tool: str, **args):
    """Run one tool under the guardrails and record it."""
    if state.get("calls", 0) >= config.MAX_TOOL_CALLS:
        raise Budget(f"tool-call limit ({config.MAX_TOOL_CALLS}) reached")
    if time.time() - state["started"] > config.CASE_TIME_LIMIT_S:
        raise Budget(f"time limit ({config.CASE_TIME_LIMIT_S} s) reached")
    state["calls"] = state.get("calls", 0) + 1
    t0, error, result = time.time(), None, None
    try:
        result = tools.TOOLS[tool](**args)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        db.add_tool_call(state["case_id"], state["run_no"], state["calls"], tool, args, result,
                         int((time.time() - t0) * 1000), error)


def _guarded(fn):
    """Turn a guardrail stop or a tool error inside a node into a note; the case then goes to a reviewer."""
    def node(state: State) -> State:
        if state.get("guardrail"):
            return {}
        try:
            return fn(state)
        except Budget as exc:
            return {"guardrail": str(exc), "trace": _step(state, fn.__name__, f"Stopped: {exc}", time.time())}
        except Exception as exc:
            return {"guardrail": f"{fn.__name__} failed ({type(exc).__name__})",
                    "trace": _step(state, fn.__name__, f"Tool failed: {type(exc).__name__}: {exc}", time.time())}
    node.__name__ = fn.__name__
    return node


# ---------------------------------------------------------------- nodes
@_guarded
def get_order(state: State) -> State:
    t0 = time.time()
    order = _call(state, "lookup_order", order_id=state["order_id"])
    msg = (f"Found {order['product']}, ₹{order['price']:,.0f}, delivered {order['days_since_delivery']} days ago."
           if order else f"No order found for {state['order_id']}.")
    return {"order": order, "calls": state["calls"], "trace": _step(state, "get_order", msg, t0)}


@_guarded
def parse_claim(state: State) -> State:
    t0 = time.time()
    r = _call(state, "parse_claim", reason=state["reason"])
    c = r["claim"]
    msg = (f"Claim: {decision.DEFECT_NAMES.get(c['defect_type'], 'no damage described')}"
           + (f" at the {c['location']}" if c.get("location") else "") + f" (read by {r['source']}).")
    return {"claim": c, "calls": state["calls"], "trace": _step(state, "parse_claim", msg, t0, r)}


@_guarded
def check_photos(state: State) -> State:
    t0 = time.time()
    r = _call(state, "assess_photos", case_id=state["case_id"], delivered_date=state["order"]["delivered_date"])
    bad = [p for p in r["photos"] if not p["ok"]]
    msg = f"{r['ok_photos']} of {r['total']} photos passed the quality check"
    msg += (": " + "; ".join(f"photo {i + 1} {', '.join(p['problems']).replace('_', ' ')}" for i, p in enumerate(r["photos"]) if not p["ok"])) if bad else "."
    if r["integrity_flags"]:
        msg += f" {len(r['integrity_flags'])} integrity flag(s)."
    return {"photos": r, "calls": state["calls"], "trace": _step(state, "check_photos", msg, t0)}


def _usable(state: State):
    return [p for p in state["photos"]["photos"] if p["ok"]]


def _describe(det: dict) -> str:
    b = det["best"]
    if not b:
        return "No photo to check."
    s = {"damaged": f"damage found ({b.get('defect_type')}, score {b.get('damage_score', b['confidence']):.2f})",
         "unsure": f"possible damage ({b.get('defect_type') or 'unclear'}, score {b.get('damage_score', 0):.2f})",
         "clean": "no damage found", "no_garment": "no garment found in the photo"}[b["verdict"]]
    return f"{b.get('backend', 'vision')}{' zoomed in' if det['mode'] == 'tiles' else ''}: {s}, {len(b.get('regions', []))} region(s)."


@_guarded
def detect_defects(state: State) -> State:
    t0 = time.time()
    r = _call(state, "detect_defects", photos=_usable(state))
    return {"detection": r, "calls": state["calls"], "trace": _step(state, "detect_defects", _describe(r), t0)}


@_guarded
def zoom_in(state: State) -> State:
    """Same tool, different parameters: four overlapping zoomed tiles per photo."""
    t0 = time.time()
    r = _call(state, "detect_defects", photos=_usable(state), tiles=True)
    first = state["detection"]
    keep = r if vision_rank(r) > vision_rank(first) else first
    msg = _describe(r) + ("" if keep is r else " Kept the first result.")
    return {"detection": keep, "calls": state["calls"], "trace": _step(state, "zoom_in", msg, t0)}


def vision_rank(det):
    b = det.get("best") or {}
    return ({"damaged": 3, "unsure": 2, "clean": 1, "no_garment": 0}.get(b.get("verdict"), 0), b.get("damage_score", 0))


@_guarded
def verify_with_vlm(state: State) -> State:
    t0 = time.time()
    b = state["detection"]["best"]
    r = _call(state, "verify_with_vlm", file=b["file"], regions=b.get("regions", []),
              claimed_class=(state.get("claim") or {}).get("defect_type"))
    msg = {"unavailable": "Vision check skipped: no Gemini key set.",
           "failed": "Vision check failed: " + "; ".join(r.get("errors", []))[:200],
           "no_damage_seen": "Vision model (blind) saw no damage in the marked area.",
           "damage_seen": "Vision model (blind) saw: " + ", ".join(r.get("types_seen", []))
                          + ("" if r.get("matches_claim", True) else " (different from the claim)") + "."}.get(r["status"], r["status"])
    return {"vision_check": r, "calls": state["calls"], "trace": _step(state, "verify_with_vlm", msg, t0)}


@_guarded
def retrieve_policy(state: State) -> State:
    t0 = time.time()
    c, b = state.get("claim") or {}, ((state.get("detection") or {}).get("best") or {})
    q = f"clothing {state['reason']} {c.get('defect_type') or ''} {b.get('defect_type') or ''} {(state.get('order') or {}).get('product_type', '')}"
    clauses = _call(state, "retrieve_policy", query=q, k=3, must_include=("R1",))
    return {"clauses": clauses, "calls": state["calls"],
            "trace": _step(state, "retrieve_policy", "Retrieved clauses " + ", ".join(k["id"] for k in clauses), t0)}


def decide(state: State) -> State:
    t0 = time.time()
    order, photos = state.get("order"), state.get("photos") or {}
    best = (state.get("detection") or {}).get("best")
    facts = {
        "order": order,
        "days_since_delivery": order["days_since_delivery"] if order else None,
        "return_window_days": config.RETURN_WINDOW_DAYS,
        "reason": state["reason"],
        "claim": state.get("claim"),
        "claims_defect": decision.claims_defect(state["reason"], state.get("claim")),
        "photo_quality": {k: photos.get(k) for k in ("ok_photos", "total", "instructions")} if photos else None,
        "integrity_flags": photos.get("integrity_flags", []),
        "detection": {k: best.get(k) for k in ("verdict", "label", "confidence", "damage_score", "defect_type", "backend", "mode")} if best else None,
        "vision_check": {k: v for k, v in (state.get("vision_check") or {}).items() if k != "observations"} or None,
        "evidence_rounds_used": state.get("evidence_rounds", 0),
        "clauses": state.get("clauses") or [],
    }
    if state.get("guardrail"):
        result = decision.finalize({**facts, "order": None}, None, "rules")  # escalate, then explain why
        result.update(decision=decision.ESCALATE, status=decision.STATUS[decision.ESCALATE], reason_code="GUARDRAIL",
                      resolution=None, recommendation=f"The agent stopped early ({state['guardrail']}). Check the case by hand.",
                      reasons=[f"The agent stopped early: {state['guardrail']}."], cited_clauses=["R7"],
                      customer_message=decision.customer_message(decision.ESCALATE, None))
        result["summary"] = result["reasons"][0]
        provider = "rules"
    else:
        proposal, provider, error = llm.propose(facts)
        result = decision.finalize(facts, proposal, provider)
        if error:
            result["notes"].append(f"Language model unavailable, used policy rules: {error}")
    msg = f"{result['decision']} via {result['source']} ({decision.REASONS.get(result['reason_code'], result['reason_code'])}); cites {', '.join(result['cited_clauses'])}"
    return {"facts": facts, "result": result, "trace": _step(state, "decide", msg, t0, {"agreement": result["agreement"]})}


def finalize(state: State) -> State:
    t0 = time.time()
    r = state["result"]
    ticket = tools.create_ticket(state["case_id"], r["decision"], r["summary"])
    return {"trace": _step(state, "finalize", f"Ticket {ticket['ticket_id']} in queue {ticket['queue']}", t0)}


# ---------------------------------------------------------------- routing
def _after_order(state: State) -> str:
    order = state.get("order")
    if not order or state.get("guardrail"):
        return "retrieve_policy"
    if config.AUTO_REJECT_OUTSIDE_WINDOW and not order["window_ok"]:
        return "retrieve_policy"  # the photos cannot change a late-return outcome, so skip them
    return "parse_claim"


def _after_photos(state: State) -> str:
    if state.get("guardrail") or not _usable(state):
        return "retrieve_policy"  # nothing to detect on; ask for new photos
    return "detect_defects"


def _needs_vlm(state: State) -> bool:
    b = (state.get("detection") or {}).get("best") or {}
    if not vlm.available() or b.get("verdict") in (None, "no_garment"):
        return False
    claims = decision.claims_defect(state["reason"], state.get("claim"))
    return claims or b["verdict"] in ("damaged", "unsure")  # an undamaged change-of-mind photo needs no second look


def _after_detect(state: State) -> str:
    if state.get("guardrail"):
        return "retrieve_policy"
    b = (state.get("detection") or {}).get("best") or {}
    claims = decision.claims_defect(state["reason"], state.get("claim"))
    weak = b.get("verdict") == "unsure" or (claims and b.get("verdict") == "clean")
    if weak and b.get("backend") == "clothes-yolo":
        return "zoom_in"
    return "verify_with_vlm" if _needs_vlm(state) else "retrieve_policy"


def _after_zoom(state: State) -> str:
    return "verify_with_vlm" if not state.get("guardrail") and _needs_vlm(state) else "retrieve_policy"


def build_graph():
    g = StateGraph(State)
    for fn in (get_order, parse_claim, check_photos, detect_defects, zoom_in, verify_with_vlm, retrieve_policy, decide, finalize):
        g.add_node(fn.__name__, fn)
    g.set_entry_point("get_order")
    g.add_conditional_edges("get_order", _after_order, {"parse_claim": "parse_claim", "retrieve_policy": "retrieve_policy"})
    g.add_edge("parse_claim", "check_photos")
    g.add_conditional_edges("check_photos", _after_photos, {"detect_defects": "detect_defects", "retrieve_policy": "retrieve_policy"})
    g.add_conditional_edges("detect_defects", _after_detect,
                            {"zoom_in": "zoom_in", "verify_with_vlm": "verify_with_vlm", "retrieve_policy": "retrieve_policy"})
    g.add_conditional_edges("zoom_in", _after_zoom, {"verify_with_vlm": "verify_with_vlm", "retrieve_policy": "retrieve_policy"})
    g.add_edge("verify_with_vlm", "retrieve_policy")
    g.add_edge("retrieve_policy", "decide")
    g.add_edge("decide", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def run_case(case_id: int) -> dict:
    """Run (or re-run, after new photos) the workflow for a stored case and persist the outcome."""
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    case = db.get_case(case_id)
    prior = case.get("trace") or []
    run_no = max((s.get("run", 1) for s in prior), default=0) + 1
    try:
        state = _GRAPH.invoke({
            "case_id": case_id, "run_no": run_no, "started": time.time(), "calls": 0,
            "order_id": case["order_id"], "reason": case["reason"], "evidence_rounds": case["evidence_rounds"] or 0,
            "trace": prior,
        })
    except Exception as exc:  # the graph itself failed: never leave the customer waiting
        db.update_case(case_id, status=decision.STATUS[decision.ESCALATE], decision=decision.ESCALATE, decided_by="system",
                       summary=f"Automatic processing failed ({type(exc).__name__}); a reviewer will check this case.",
                       customer_message=decision.customer_message(decision.ESCALATE, None),
                       recommendation="The agent failed. Check the case by hand.")
        db.add_event(case_id, "system", "agent_error", {"error": f"{type(exc).__name__}: {exc}"})
        return db.get_case(case_id)
    r = state["result"]
    best = (state.get("detection") or {}).get("best")
    db.update_case(
        case_id, status=r["status"], decision=r["decision"], resolution=r["resolution"], confidence=r["confidence"],
        verdict=r["verdict"], recommendation=r["recommendation"], summary=r["summary"],
        customer_message=r["customer_message"], request_message=r["request"] or None, decided_by="agent",
        parsed_claim=state.get("claim"), preanalysis=state.get("photos"), vision=best,
        vlm=state.get("vision_check"), citations=r["cited_clauses"],
        decision_detail={"reasons": r["reasons"], "source": r["source"], "agreement": r["agreement"], "notes": r["notes"],
                         "reason_code": r["reason_code"],
                         "clauses": [c for c in policy.all_clauses() if c["id"] in set(r["cited_clauses"])],
                         "retrieved": [c["id"] for c in state.get("clauses", [])],
                         "per_photo": [{k: p.get(k) for k in ("media_id", "verdict", "damage_score", "defect_type", "regions")}
                                       for p in (state.get("detection") or {}).get("per_photo", [])]},
        trace=state["trace"],
    )
    return db.get_case(case_id)
