"""Tools the agent can call. The same functions are exposed over MCP by app/mcp_server.py."""
import datetime as dt

from . import db, policy, vision


def lookup_order(order_id: str) -> dict | None:
    """Find an order and compute how many days ago it was delivered."""
    order = db.get_order(order_id or "")
    if not order:
        return None
    delivered = dt.date.fromisoformat(order["delivered_date"])
    order["days_since_delivery"] = (dt.date.today() - delivered).days
    return order


def run_vision(image_path: str) -> dict:
    """Inspect the product photo and return defect evidence with regions."""
    return vision.analyze(image_path)


def retrieve_policy(query: str, k: int = 3, must_include=("R1",)) -> list[dict]:
    """Retrieve the policy clauses most relevant to the query."""
    return policy.retrieve(query, k=k, must_include=must_include)


def create_ticket(case_id: int, decision: str, summary: str) -> dict:
    """Open a ticket in the right queue and attach it to the case."""
    queue = {"approve": "refunds", "reject": "customer-care", "escalate": "human-review"}.get(decision, "human-review")
    ticket_id = f"TKT-{case_id:05d}"
    db.update_case(case_id, ticket_id=ticket_id)
    db.add_event(case_id, "agent", "ticket_created", {"ticket_id": ticket_id, "queue": queue, "summary": summary})
    return {"ticket_id": ticket_id, "queue": queue}


TOOLS = {
    "lookup_order": lookup_order,
    "run_vision": run_vision,
    "retrieve_policy": retrieve_policy,
    "create_ticket": create_ticket,
}
