"""SQLite persistence: orders, return cases and an audit trail of events."""
import datetime as dt
import json
import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, customer TEXT, product TEXT, category TEXT,
    price REAL, delivered_date TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT, reason TEXT, image_file TEXT,
    status TEXT, decision TEXT, resolution TEXT, confidence REAL,
    summary TEXT, customer_message TEXT, decided_by TEXT,
    vision TEXT, citations TEXT, decision_detail TEXT, trace TEXT,
    ticket_id TEXT, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER, ts TEXT, actor TEXT, action TEXT, detail TEXT
);
"""

JSON_FIELDS = ("vision", "citations", "decision_detail", "trace")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _days_ago(n: int) -> str:
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()


def seed_orders(reset: bool = False) -> None:
    rows = [
        ("ORD-1001", "Asha Menon", "Bluetooth Speaker", "Electronics", 3499, _days_ago(5)),
        ("ORD-1002", "Ravi Kumar", "Ceramic Mug Set", "Home", 899, _days_ago(12)),
        ("ORD-1003", "Meera Iyer", "Running Shoes", "Footwear", 4299, _days_ago(20)),
        ("ORD-1004", "Karthik S", "Desk Lamp", "Home", 1799, _days_ago(45)),
        ("ORD-1005", "Divya R", "Face Serum", "Personal Care", 1250, _days_ago(3)),
        ("ORD-1006", "Imran Khan", "Noise-Cancelling Headphones", "Electronics", 14999, _days_ago(6)),
        ("ORD-1007", "Sneha P", "Travel Backpack", "Bags", 2599, _days_ago(8)),
    ]
    with conn() as c:
        if reset:
            c.execute("DELETE FROM orders")
        if c.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
            c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", rows)


def init_db(reset: bool = False) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    with conn() as c:
        c.executescript(SCHEMA)
    seed_orders()


def get_order(order_id: str):
    with conn() as c:
        row = c.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)).fetchone()
    return dict(row) if row else None


def list_orders():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM orders ORDER BY order_id")]


def _decode(row):
    d = dict(row)
    for f in JSON_FIELDS:
        d[f] = json.loads(d[f]) if d.get(f) else None
    return d


def create_case(order_id: str, reason: str, image_file: str) -> int:
    ts = now()
    with conn() as c:
        cur = c.execute(
            "INSERT INTO cases (order_id, reason, image_file, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (order_id.strip().upper(), reason, image_file, "Submitted", ts, ts),
        )
        return cur.lastrowid


def update_case(case_id: int, **fields) -> None:
    for f in JSON_FIELDS:
        if f in fields and fields[f] is not None:
            fields[f] = json.dumps(fields[f])
    fields["updated_at"] = now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE cases SET {cols} WHERE id = ?", (*fields.values(), case_id))


def get_case(case_id: int):
    with conn() as c:
        row = c.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    return _decode(row) if row else None


def list_cases(limit: int = 50):
    with conn() as c:
        rows = c.execute("SELECT * FROM cases ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_decode(r) for r in rows]


def add_event(case_id: int, actor: str, action: str, detail=None) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO events (case_id, ts, actor, action, detail) VALUES (?,?,?,?,?)",
            (case_id, now(), actor, action, json.dumps(detail) if detail is not None else None),
        )


def events_for(case_id: int):
    with conn() as c:
        rows = c.execute("SELECT * FROM events WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["detail"] = json.loads(d["detail"]) if d["detail"] else None
        out.append(d)
    return out
