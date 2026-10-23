"""LangGraph workflow that processes one return case.

    lookup_order --(order found)--> analyze_image --> retrieve_policy --> decide --> finalize
         \\--------(no order)--------------------------^

Every node writes an entry to the case trace and to the audit-event table, so the UI can show
exactly what the agent did and why.
"""
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import config, db, decision, llm, policy, tools


class State(TypedDict, total=False):
    case_id: int
    order_id: str
    reason: str
    image_path: str
    order: Any
    vision: Any
    clauses: Any
    facts: Any
    result: Any
    ticket: Any
    trace: list


def _log(state: State, step: str, summary: str, t0: float, detail=None) -> list:
    entry = {"step": step, "summary": summary, "ms": int((time.time() - t0) * 1000)}
    db.add_event(state["case_id"], "agent", step, {"summary": summary, **({"detail": detail} if detail else {})})
    return state.get("trace", []) + [entry]


def lookup_order(state: State) -> State:
    t0 = time.time()
    order = tools.lookup_order(state["order_id"])
    msg = (f"Found {order['product']} ({order['category']}), delivered {order['days_since_delivery']} days ago."
           if order else f"No order found for {state['order_id']}.")
    return {"order": order, "trace": _log(state, "lookup_order", msg, t0)}


def analyze_image(state: State) -> State:
    t0 = time.time()
    v = tools.run_vision(state["image_path"])
    msg = f"{v['backend']} backend: {v['label']} (confidence {v['confidence']:.2f})" + (
        f", {v['defect_type']}" if v.get("defect_type") else "")
    return {"vision": v, "trace": _log(state, "analyze_image", msg, t0, {"regions": len(v["regions"])})}


def retrieve_policy(state: State) -> State:
    t0 = time.time()
    v = state.get("vision") or {}
    q = f"{state['reason']} {v.get('defect_type') or ''} {(state.get('order') or {}).get('category', '')}"
    clauses = tools.retrieve_policy(q, k=3, must_include=("R1",))
    return {"clauses": clauses, "trace": _log(state, "retrieve_policy", "Retrieved clauses " + ", ".join(c["id"] for c in clauses), t0)}


def decide(state: State) -> State:
    t0 = time.time()
    order = state.get("order")
    facts = {
        "order": order,
        "days_since_delivery": order["days_since_delivery"] if order else None,
        "return_window_days": config.RETURN_WINDOW_DAYS,
        "reason": state["reason"],
        "claims_defect": decision.claims_defect(state["reason"]),
        "vision": state.get("vision"),
        "clauses": state.get("clauses") or [],
    }
    proposal, provider, error = llm.propose(facts)
    result = decision.finalize(facts, proposal, provider)
    if error:
        result["notes"].append(f"LLM unavailable, used policy rules: {error}")
    msg = f"{result['decision']} via {result['source']}; cites {', '.join(result['cited_clauses'])}"
    return {"facts": facts, "result": result, "trace": _log(state, "decide", msg, t0, {"agreement": result["agreement"]})}


def finalize(state: State) -> State:
    t0 = time.time()
    r = state["result"]
    ticket = tools.create_ticket(state["case_id"], r["decision"], r["summary"])
    return {"ticket": ticket, "trace": _log(state, "finalize", f"Ticket {ticket['ticket_id']} opened in {ticket['queue']}", t0)}


def _route_after_lookup(state: State) -> str:
    return "analyze_image" if state.get("order") else "retrieve_policy"


def build_graph():
    g = StateGraph(State)
    for fn in (lookup_order, analyze_image, retrieve_policy, decide, finalize):
        g.add_node(fn.__name__, fn)
    g.set_entry_point("lookup_order")
    g.add_conditional_edges("lookup_order", _route_after_lookup, {"analyze_image": "analyze_image", "retrieve_policy": "retrieve_policy"})
    g.add_edge("analyze_image", "retrieve_policy")
    g.add_edge("retrieve_policy", "decide")
    g.add_edge("decide", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def run_case(case_id: int) -> dict:
    """Run the workflow for a stored case and persist the outcome."""
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    case = db.get_case(case_id)
    state = _GRAPH.invoke({
        "case_id": case_id, "order_id": case["order_id"], "reason": case["reason"],
        "image_path": str(config.UPLOAD_DIR / case["image_file"]), "trace": [],
    })
    r = state["result"]
    db.update_case(
        case_id, status=r["status"], decision=r["decision"], resolution=r["resolution"], confidence=r["confidence"],
        summary=r["summary"], customer_message=r["customer_message"], decided_by="agent",
        vision=state.get("vision"), citations=r["cited_clauses"],
        decision_detail={"reasons": r["reasons"], "source": r["source"], "agreement": r["agreement"], "notes": r["notes"],
                         "clauses": [c for c in policy.all_clauses() if c["id"] in set(r["cited_clauses"])],
                         "retrieved": [c["id"] for c in state.get("clauses", [])]},
        trace=state["trace"],
    )
    return db.get_case(case_id)
