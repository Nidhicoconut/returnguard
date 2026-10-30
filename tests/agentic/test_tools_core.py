import datetime as dt
import sqlite3

import pytest

from agentic.config import AgenticConfig
from agentic.cv.runner import ToolRunner
from agentic.schemas import Category, DefectClass
from agentic.tools_core import (
    InMemoryOrderStore, OrderRecord, PolicyStore, SqliteOrderStore, fetch_order, fetch_policy, lookup_order, map_category,
)

TODAY = dt.date(2026, 10, 29)


def rec(order_id="ORD-1", category="Footwear", price=4299.0, days=10, sku="SKU-1"):
    return OrderRecord(order_id=order_id, sku=sku, category=category, price_inr=price,
                       delivered_date=(TODAY - dt.timedelta(days=days)).isoformat())


def look(r, **kw):
    return lookup_order(r.order_id, InMemoryOrderStore([r]), today=TODAY, window_days=kw.get("window_days", 30))


@pytest.mark.parametrize("raw,expected", [
    ("Footwear", Category.apparel_footwear), ("bags", Category.apparel_footwear), ("Electronics", Category.electronics),
    ("Home", Category.drinkware_kitchen), (" drinkware kitchen ", Category.drinkware_kitchen),
    ("Personal Care", None), ("Software", None), ("", None),
])
def test_map_category(raw, expected):
    assert map_category(raw) == expected


@pytest.mark.parametrize("days,ok", [(0, True), (30, True), (31, False)])
def test_window_boundary_is_inclusive_at_30_days(days, ok):
    res = look(rec(days=days))
    assert res.status == "found" and res.order.window_ok is ok and res.days_since_delivery == days


def test_window_comes_from_config_value():
    assert look(rec(days=12), window_days=10).order.window_ok is False


def test_found_order_is_a_full_orderdetails():
    o = look(rec()).order
    assert (o.order_id, o.sku, o.category, o.price_inr) == ("ORD-1", "SKU-1", Category.apparel_footwear, 4299.0)


def test_not_found_and_blank_ids_are_valid_outcomes_not_errors():
    assert lookup_order("NOPE", InMemoryOrderStore(), today=TODAY, window_days=30).status == "not_found"
    assert lookup_order("  ", InMemoryOrderStore([rec()]), today=TODAY, window_days=30).status == "not_found"


def test_unsupported_category_keeps_price_and_label_for_later_policy():
    res = look(rec(category="Personal Care", price=1250))
    assert res.status == "unsupported_category" and res.order is None
    assert res.raw_category == "Personal Care" and res.price_inr == 1250 and res.days_since_delivery == 10


def test_fetch_order_is_audited_and_json_safe():
    r = ToolRunner()
    res = fetch_order("ORD-1", InMemoryOrderStore([rec()]), r, AgenticConfig(), TODAY)
    assert res.ok and res.output.status == "found"
    assert r.ledger[0].tool == "T0_order_lookup" and r.ledger[0].args["today"] == "2026-10-29"
    assert r.ledger[0].result["order"]["window_ok"] is True


@pytest.mark.parametrize("bad_date", ["not-a-date", (TODAY + dt.timedelta(days=3)).isoformat()])
def test_corrupt_dates_become_tool_errors(bad_date):
    bad = OrderRecord(order_id="ORD-1", sku="S", category="Footwear", price_inr=1, delivered_date=bad_date)
    res = fetch_order("ORD-1", InMemoryOrderStore([bad]), ToolRunner(max_retries=0), AgenticConfig(), TODAY)
    assert not res.ok and "ValueError" in res.call.error


def test_store_exception_never_escapes():
    class Boom:
        def get(self, _):
            raise OSError("db locked")

    r = ToolRunner(max_retries=0)
    res = fetch_order("ORD-1", Boom(), r, AgenticConfig(), TODAY)
    assert not res.ok and r.ledger[0].error == "OSError: db locked"


# ----------------------------------------------------------------------------- sqlite (read-only)
def make_db(path, with_sku=False):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer TEXT, product TEXT, category TEXT, price REAL, delivered_date TEXT"
                + (", sku TEXT" if with_sku else "") + ")")
    cols = "(order_id, customer, product, category, price, delivered_date" + (", sku)" if with_sku else ")")
    vals = ("ORD-9", "A", "Running Shoes", "Footwear", 4299, "2026-10-20") + (("SKU-RUN-9",) if with_sku else ())
    con.execute(f"INSERT INTO orders {cols} VALUES ({','.join('?' * len(vals))})", vals)
    con.commit()
    con.close()


def test_sqlite_store_reads_and_derives_sku(tmp_path):
    make_db(tmp_path / "o.db")
    s = SqliteOrderStore(tmp_path / "o.db")
    r = s.get("ORD-9")
    assert (r.sku, r.category, r.price_inr) == ("running-shoes", "Footwear", 4299.0)
    assert s.get("ORD-0") is None


def test_sqlite_store_prefers_a_sku_column(tmp_path):
    make_db(tmp_path / "o.db", with_sku=True)
    assert SqliteOrderStore(tmp_path / "o.db").get("ORD-9").sku == "SKU-RUN-9"


def test_sqlite_store_is_read_only_and_never_creates_files(tmp_path):
    make_db(tmp_path / "o.db")
    s = SqliteOrderStore(tmp_path / "o.db")
    con = sqlite3.connect(s._uri, uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM orders")
    con.close()
    missing = tmp_path / "missing.db"
    res = fetch_order("ORD-9", SqliteOrderStore(missing), ToolRunner(max_retries=0), AgenticConfig(), TODAY)
    assert not res.ok and not missing.exists()


# ----------------------------------------------------------------------------- policy
def test_policy_store_parses_the_real_policy_file():
    clauses = PolicyStore().all()
    assert [c.clause_id for c in clauses] == [f"R{i}" for i in range(1, 8)]
    by_id = {c.clause_id: c for c in clauses}
    assert "30 days" in by_id["R1"].text and "10000" in by_id["R7"].text
    assert set(by_id["R2"].covers_defects) == set(DefectClass) and by_id["R1"].covers_defects == []


def test_policy_get_subset_and_audit():
    r = ToolRunner()
    res = fetch_policy(PolicyStore(), r, ["R1", "R7", "R99"])
    assert [c.clause_id for c in res.output.clauses] == ["R1", "R7"]
    assert r.ledger[0].tool == "T8_policy" and r.ledger[0].args["ids"] == ["R1", "R7", "R99"]


def test_policy_failures_are_tool_errors(tmp_path):
    (tmp_path / "empty.md").write_text("no headings here")
    for path in (tmp_path / "missing.md", tmp_path / "empty.md"):
        res = fetch_policy(PolicyStore(path), ToolRunner(max_retries=0))
        assert not res.ok and res.output is None
