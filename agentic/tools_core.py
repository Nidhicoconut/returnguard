"""Non-CV tools: T0 order lookup and T8 policy retrieval.

Both run through the ToolRunner so they appear in the audit ledger like any CV tool.
Order data comes from an ``OrderStore`` (in-memory for tests, read-only SQLite for the real DB).
Nothing here writes to any database.
"""
import datetime as dt
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel

from agentic.config import AgenticConfig
from agentic.cv.runner import ToolRunner, ToolResult
from agentic.models import OrderLookupResult, PolicyBundle
from agentic.schemas import Category, DefectClass, OrderDetails, PolicyClause

T0, T8 = "T0_order_lookup", "T8_policy"

# store category label -> shared Category. Anything else (e.g. "Personal Care") is "unsupported_category"
# and is routed to a human by the policy engine (R4).
_CATEGORY_MAP = {
    "electronics": Category.electronics,
    "footwear": Category.apparel_footwear, "apparel": Category.apparel_footwear,
    "clothing": Category.apparel_footwear, "fashion": Category.apparel_footwear, "bags": Category.apparel_footwear,
    "apparel_footwear": Category.apparel_footwear,
    "home": Category.drinkware_kitchen, "kitchen": Category.drinkware_kitchen,
    "drinkware": Category.drinkware_kitchen, "drinkware_kitchen": Category.drinkware_kitchen,
}


def map_category(raw: str) -> Optional[Category]:
    return _CATEGORY_MAP.get(re.sub(r"[\s\-]+", "_", (raw or "").strip().lower()))


# ----------------------------------------------------------------------------- order stores
class OrderRecord(BaseModel):
    order_id: str
    sku: str
    category: str          # the store's own label
    price_inr: float
    delivered_date: str    # ISO date


class OrderStore(Protocol):
    def get(self, order_id: str) -> Optional[OrderRecord]: ...


class InMemoryOrderStore:
    def __init__(self, records: Iterable[OrderRecord] = ()):
        self._by_id = {r.order_id: r for r in records}

    def get(self, order_id: str) -> Optional[OrderRecord]:
        return self._by_id.get(order_id)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class SqliteOrderStore:
    """READ-ONLY view of the existing ``orders`` table (order_id, product, category, price, delivered_date).
    The table has no SKU column, so a SKU is derived from the product name unless a ``sku`` column exists."""

    def __init__(self, db_path: str | Path):
        self._uri = f"file:{Path(db_path).as_posix()}?mode=ro"

    def get(self, order_id: str) -> Optional[OrderRecord]:
        con = sqlite3.connect(self._uri, uri=True)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        keys = row.keys()
        sku = row["sku"] if "sku" in keys and row["sku"] else _slug(str(row["product"]))
        return OrderRecord(order_id=row["order_id"], sku=sku, category=str(row["category"] or ""),
                           price_inr=float(row["price"]), delivered_date=str(row["delivered_date"]))


# ----------------------------------------------------------------------------- T0
def lookup_order(order_id: str, store: OrderStore, *, today: dt.date, window_days: int) -> OrderLookupResult:
    """Pure lookup. Raises ValueError on corrupt data (bad/future date) so it surfaces as a tool error."""
    rec = store.get((order_id or "").strip()) if (order_id or "").strip() else None
    if rec is None:
        return OrderLookupResult(status="not_found", order_id=order_id or "")
    try:
        delivered = dt.date.fromisoformat(rec.delivered_date)
    except ValueError:
        raise ValueError(f"order {rec.order_id}: invalid delivered_date {rec.delivered_date!r}") from None
    days = (today - delivered).days
    if days < 0:
        raise ValueError(f"order {rec.order_id}: delivered_date {rec.delivered_date} is in the future")
    common = dict(order_id=rec.order_id, raw_category=rec.category, price_inr=rec.price_inr,
                  delivered_date=rec.delivered_date, days_since_delivery=days)
    category = map_category(rec.category)
    if category is None:
        return OrderLookupResult(status="unsupported_category", **common)
    details = OrderDetails(order_id=rec.order_id, sku=rec.sku, category=category, price_inr=rec.price_inr,
                           delivered_date=rec.delivered_date, window_ok=days <= window_days)
    return OrderLookupResult(status="found", order=details, **common)


def fetch_order(order_id: str, store: OrderStore, runner: ToolRunner, config: Optional[AgenticConfig] = None,
                today: Optional[dt.date] = None) -> ToolResult:
    cfg = config or AgenticConfig()
    day = today or dt.date.today()
    return runner.call(
        T0, lambda: lookup_order(order_id, store, today=day, window_days=cfg.return_window_days),
        args={"order_id": order_id, "today": day.isoformat(), "window_days": cfg.return_window_days},
        expect=OrderLookupResult,
    )


# ----------------------------------------------------------------------------- T8
DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "return_policy.md"
_COVERS: Mapping[str, list[DefectClass]] = {"R2": list(DefectClass)}  # R2 is the defect-refund clause
_HEADING = re.compile(r"^##\s+(R\d+)\s+(.+?)\s*$", re.M)


class PolicyStore:
    """Loads the store policy markdown (``## R1 Title`` sections) into PolicyClause objects."""

    def __init__(self, path: str | Path = DEFAULT_POLICY_PATH):
        self._path = Path(path)

    def all(self) -> list[PolicyClause]:
        text = self._path.read_text(encoding="utf-8")
        heads = list(_HEADING.finditer(text))
        if not heads:
            raise ValueError(f"no '## R<n> Title' clauses found in {self._path}")
        out = []
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = " ".join(text[m.end():end].split())
            out.append(PolicyClause(clause_id=m.group(1), text=f"{m.group(2)}: {body}",
                                    covers_defects=list(_COVERS.get(m.group(1), []))))
        return out

    def get(self, ids: Optional[Sequence[str]] = None) -> list[PolicyClause]:
        clauses = self.all()
        if ids is None:
            return clauses
        wanted = set(ids)
        return [c for c in clauses if c.clause_id in wanted]


def fetch_policy(store: PolicyStore, runner: ToolRunner, ids: Optional[Sequence[str]] = None) -> ToolResult:
    return runner.call(T8, lambda: PolicyBundle(clauses=store.get(ids)),
                       args={"ids": list(ids) if ids else "all"}, expect=PolicyBundle)
