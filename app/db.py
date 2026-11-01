"""SQLite persistence: users and sessions, clothing orders, return cases, their photos,
every agent tool call, reviewer decisions and an audit trail of events."""
import datetime as dt
import json
import sqlite3
from contextlib import contextmanager

from . import config, security

SCHEMA_VERSION = 2  # 1 = the first prototype (no users, one photo per case)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL, name TEXT, role TEXT NOT NULL CHECK (role IN ('customer', 'reviewer')),
    password_hash TEXT NOT NULL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, user_id INTEGER, customer TEXT, product TEXT, product_type TEXT,
    category TEXT, price REAL, delivered_date TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, order_id TEXT, reason TEXT,
    status TEXT, decision TEXT, resolution TEXT, confidence REAL, verdict TEXT, recommendation TEXT,
    summary TEXT, customer_message TEXT, request_message TEXT, evidence_rounds INTEGER DEFAULT 0,
    parsed_claim TEXT, preanalysis TEXT, vision TEXT, vlm TEXT, citations TEXT, decision_detail TEXT, trace TEXT,
    ticket_id TEXT, decided_by TEXT, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER, round INTEGER, file TEXT, sha256 TEXT, dhash TEXT, width INTEGER, height INTEGER,
    quality TEXT, integrity TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER, run_no INTEGER, step_no INTEGER, tool TEXT, args TEXT, result TEXT,
    latency_ms INTEGER, error TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER, reviewer_id INTEGER, action TEXT, reason_code TEXT, note TEXT, vision_feedback TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER, ts TEXT, actor TEXT, action TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_cases_user ON cases (user_id);
CREATE INDEX IF NOT EXISTS idx_media_case ON media (case_id);
CREATE INDEX IF NOT EXISTS idx_calls_case ON tool_calls (case_id);
"""

CASE_JSON = ("parsed_claim", "preanalysis", "vision", "vlm", "citations", "decision_detail", "trace")
MEDIA_JSON = ("quality", "integrity")

# Demo accounts (passwords are for local demos only; see README). Reviewers handle escalated cases.
DEMO_USERS = [
    ("asha", "Asha Menon", "customer", "demo1234"),
    ("ravi", "Ravi Kumar", "customer", "demo1234"),
    ("neha", "Neha Sharma", "customer", "demo1234"),
    ("reviewer", "Return desk", "reviewer", "review1234"),
]


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


def _orders():
    # (order_id, username, product, product_type, price, days since delivery)
    return [
        ("ORD-2001", "asha", "Cotton T-Shirt", "t-shirt", 799, 3),
        ("ORD-2002", "asha", "Wool Sweater", "sweater", 2499, 5),
        ("ORD-2003", "asha", "Slim Fit Jeans", "jeans", 1999, 25),
        ("ORD-2004", "ravi", "Formal Trousers", "trousers", 1899, 4),
        ("ORD-2005", "ravi", "Linen Shirt", "shirt", 1499, 2),
        ("ORD-2006", "ravi", "Down Winter Jacket", "jacket", 6999, 6),
        ("ORD-2007", "neha", "Floral Midi Dress", "dress", 2199, 7),
        ("ORD-2008", "neha", "Cotton Socks, pack of 3", "socks", 399, 3),
        ("ORD-2009", "neha", "Fleece Hoodie", "hoodie", 1799, 1),
    ]


def seed(reset: bool = False) -> None:
    with conn() as c:
        if reset:
            c.execute("DELETE FROM orders")
        ids = {}
        for username, name, role, password in DEMO_USERS:
            row = c.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                cur = c.execute("INSERT INTO users (username, name, role, password_hash, created_at) VALUES (?,?,?,?,?)",
                                (username, name, role, security.hash_password(password), now()))
                ids[username] = (cur.lastrowid, name)
            else:
                ids[username] = (row["id"], name)
        # OR IGNORE: existing databases pick up orders added to this list later.
        c.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?)", [
            (oid, ids[u][0], ids[u][1], product, ptype, "Clothing", price, _days_ago(days))
            for oid, u, product, ptype, price, days in _orders()])


def init_db(reset: bool = False) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    with conn() as c:
        version = c.execute("PRAGMA user_version").fetchone()[0]
        has_tables = c.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='cases'").fetchone()[0]
        if has_tables and version < SCHEMA_VERSION:
            # The first prototype's tables have a different shape; its demo data is not kept.
            for t in ("orders", "cases", "events"):
                c.execute(f"DROP TABLE IF EXISTS {t}")
        c.executescript(SCHEMA)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    seed()


def _decode(row, fields):
    d = dict(row)
    for f in fields:
        d[f] = json.loads(d[f]) if d.get(f) else None
    return d


def _encode(fields: dict, json_fields) -> dict:
    return {k: (json.dumps(v) if k in json_fields and v is not None else v) for k, v in fields.items()}


# ---------------------------------------------------------------- users and sessions
def get_user_by_name(username: str):
    with conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    return dict(row) if row else None


def get_user(user_id: int):
    with conn() as c:
        row = c.execute("SELECT id, username, name, role FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_session(user_id: int) -> str:
    token = security.new_token()
    expires = (dt.datetime.now() + dt.timedelta(days=config.SESSION_DAYS)).isoformat(timespec="seconds")
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now(),))
        c.execute("INSERT INTO sessions VALUES (?,?,?)", (token, user_id, expires))
    return token


def session_user(token: str | None):
    if not token:
        return None
    with conn() as c:
        row = c.execute("SELECT u.id, u.username, u.name, u.role FROM sessions s JOIN users u ON u.id = s.user_id "
                        "WHERE s.token = ? AND s.expires_at > ?", (token, now())).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ---------------------------------------------------------------- orders
def get_order(order_id: str):
    with conn() as c:
        row = c.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.strip().upper(),)).fetchone()
    return dict(row) if row else None


def list_orders(user_id: int | None = None):
    with conn() as c:
        if user_id is None:
            return [dict(r) for r in c.execute("SELECT * FROM orders ORDER BY order_id")]
        return [dict(r) for r in c.execute("SELECT * FROM orders WHERE user_id = ? ORDER BY order_id", (user_id,))]


# ---------------------------------------------------------------- cases
def create_case(user_id: int, order_id: str, reason: str) -> int:
    ts = now()
    with conn() as c:
        cur = c.execute(
            "INSERT INTO cases (user_id, order_id, reason, status, evidence_rounds, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, order_id.strip().upper(), reason, "Checking", 0, ts, ts),
        )
        return cur.lastrowid


def update_case(case_id: int, **fields) -> None:
    fields = _encode(fields, CASE_JSON)
    fields["updated_at"] = now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE cases SET {cols} WHERE id = ?", (*fields.values(), case_id))


def get_case(case_id: int):
    with conn() as c:
        row = c.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    return _decode(row, CASE_JSON) if row else None


def list_cases(user_id: int | None = None, statuses: tuple[str, ...] | None = None, limit: int = 100):
    sql, args = "SELECT * FROM cases WHERE 1=1", []
    if user_id is not None:
        sql += " AND user_id = ?"; args.append(user_id)
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"; args += list(statuses)
    sql += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with conn() as c:
        return [_decode(r, CASE_JSON) for r in c.execute(sql, args).fetchall()]


# ---------------------------------------------------------------- media
def add_media(case_id: int, round_no: int, file: str, sha256: str, dhash: str, width: int, height: int,
              quality=None, integrity=None) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO media (case_id, round, file, sha256, dhash, width, height, quality, integrity, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (case_id, round_no, file, sha256, dhash, width, height,
             json.dumps(quality) if quality is not None else None, json.dumps(integrity) if integrity is not None else None, now()))
        return cur.lastrowid


def update_media(media_id: int, **fields) -> None:
    fields = _encode(fields, MEDIA_JSON)
    cols = ", ".join(f"{k} = ?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE media SET {cols} WHERE id = ?", (*fields.values(), media_id))


def media_for(case_id: int):
    with conn() as c:
        rows = c.execute("SELECT * FROM media WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
    return [_decode(r, MEDIA_JSON) for r in rows]


def media_by_file(file: str):
    with conn() as c:
        row = c.execute("SELECT * FROM media WHERE file = ?", (file,)).fetchone()
    return _decode(row, MEDIA_JSON) if row else None


def other_media_hashes(case_id: int):
    """(case_id, sha256, dhash) of photos uploaded to other cases, for duplicate checks."""
    with conn() as c:
        return [tuple(r) for r in c.execute("SELECT case_id, sha256, dhash FROM media WHERE case_id != ?", (case_id,))]


# ---------------------------------------------------------------- agent tool calls, reviews, events
def add_tool_call(case_id: int, run_no: int, step_no: int, tool: str, args: dict, result, latency_ms: int, error=None) -> None:
    with conn() as c:
        c.execute("INSERT INTO tool_calls (case_id, run_no, step_no, tool, args, result, latency_ms, error, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                  (case_id, run_no, step_no, tool, json.dumps(args, default=str), json.dumps(result, default=str), latency_ms, error, now()))


def tool_calls_for(case_id: int):
    with conn() as c:
        rows = c.execute("SELECT * FROM tool_calls WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
    return [_decode(r, ("args", "result")) for r in rows]


def add_review(case_id: int, reviewer_id: int, action: str, reason_code: str, note: str, vision_feedback) -> None:
    with conn() as c:
        c.execute("INSERT INTO reviews (case_id, reviewer_id, action, reason_code, note, vision_feedback, ts) VALUES (?,?,?,?,?,?,?)",
                  (case_id, reviewer_id, action, reason_code, note, vision_feedback, now()))


def reviews_for(case_id: int):
    with conn() as c:
        rows = c.execute("SELECT r.*, u.name AS reviewer FROM reviews r LEFT JOIN users u ON u.id = r.reviewer_id "
                         "WHERE case_id = ? ORDER BY r.id", (case_id,)).fetchall()
    return [dict(r) for r in rows]


def add_event(case_id: int, actor: str, action: str, detail=None) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO events (case_id, ts, actor, action, detail) VALUES (?,?,?,?,?)",
            (case_id, now(), actor, action, json.dumps(detail) if detail is not None else None),
        )


def events_for(case_id: int):
    with conn() as c:
        rows = c.execute("SELECT * FROM events WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
    return [_decode(r, ("detail",)) for r in rows]
