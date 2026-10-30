"""Shared fixtures for the agentic tests. Every test here runs with the network blocked."""
import datetime as dt
import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise RuntimeError("network access is blocked in agentic tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def make_bundle():
    """Factory: run the whole pre-judge pipeline offline (parse -> T0 -> T8 -> CV plan -> evidence).

    Returns ``(bundle, runner)``. ``scenarios`` are mock scenario names, one per image; the image
    path carries the name so the mock toolbox picks it up.
    """
    from agentic.claim_parser import analyze_claim
    from agentic.config import AgenticConfig
    from agentic.cv import ToolRunner, build_mock_toolbox, run_cv_plan
    from agentic.evidence import build_evidence
    from agentic.models import CaseInput, ImageInput
    from agentic.tools_core import InMemoryOrderStore, OrderRecord, PolicyStore, fetch_order, fetch_policy

    def _make(scenarios=("stain_on_sleeve",), claim="There is a stain on the sleeve", *, raw_category="Footwear",
              price=2599.0, days=5, cfg=None, toolbox=None, order_id="ORD-1"):
        cfg = cfg or AgenticConfig()
        today = dt.date(2026, 10, 29)
        runner = ToolRunner(timeout_s=5)
        case = CaseInput(case_id="CASE-1", order_id=order_id, claim_text=claim,
                         images=[ImageInput(image_id=f"img{i + 1}", path=f"/p/{s}.jpg") for i, s in enumerate(scenarios)])
        store = InMemoryOrderStore([OrderRecord(order_id="ORD-1", sku="SKU-1", category=raw_category, price_inr=price,
                                                delivered_date=(today - dt.timedelta(days=days)).isoformat())])
        info = analyze_claim(claim)
        order = fetch_order(order_id, store, runner, cfg, today).output
        clauses = fetch_policy(PolicyStore(), runner).output.clauses
        sku = order.order.sku if order and order.order else None
        cv = run_cv_plan(case, sku, toolbox or build_mock_toolbox(), runner, cfg)
        bundle = build_evidence(case=case, claim=info, order=order, clauses=clauses, cv=cv, config=cfg, ledger=runner.ledger)
        return bundle, runner

    return _make
