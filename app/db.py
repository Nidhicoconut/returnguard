"""SQLite persistence: users and sessions, clothing orders, return cases, their photos,
every agent tool call, reviewer decisions and an audit trail of events."""
import datetime as dt
import json
import sqlite3
from contextlib import contextmanager

from . import config, security

SCHEMA_VERSION = 4  # 1 = first prototype (no users); 2 = multi-page app; 3 = orders have sku and colour; 4 = doorstep pickups

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL, name TEXT, role TEXT NOT NULL CHECK (role IN ('customer', 'reviewer', 'courier')),
    password_hash TEXT NOT NULL, created_at TEXT, address TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, user_id INTEGER, customer TEXT, product TEXT, product_type TEXT,
    category TEXT, price REAL, delivered_date TEXT, sku TEXT, colour TEXT
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
CREATE TABLE IF NOT EXISTS pickups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER UNIQUE, courier_id INTEGER, status TEXT, scheduled_for TEXT, slot TEXT, code TEXT, attempts INTEGER DEFAULT 1,
    checks TEXT, note TEXT, photo TEXT, failure_reason TEXT, verified_at TEXT, refund_amount REAL, refunded_at TEXT,
    created_at TEXT, updated_at TEXT
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
    ("courier", "Arjun Rao", "courier", "pickup1234"),  # pickup partner: collects approved returns and checks them at the door
]
# Made-up pickup addresses for the demo customers.
ADDRESSES = {
    "asha": "Flat 4B, Lakeview Residency, 12th Main Road, Indiranagar, Bengaluru 560038",
    "ravi": "221 Palm Grove Apartments, Koramangala 5th Block, Bengaluru 560095",
    "neha": "House 18, 3rd Cross, Jayanagar 4th Block, Bengaluru 560011",
}


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
    # (order_id, username, product, product_type, price, days since delivery, sku, colour or None for any)
    # The demo products are real Amazon Fashion listings: the SKU is the product's parent ASIN, its listing photos are
    # in demo_amazon/catalog/<sku>/ and a damaged customer photo of each is in demo_amazon/customer_photos/
    # (Amazon Reviews 2023, research use only; see demo_amazon/README.md). The colour is the variant bought.
    return [
        ("ORD-3001", "asha", "Women's V-Neck Crop Sweater", "sweater", 1299, 3, "B08CRKG11D", "white"),
        ("ORD-3002", "asha", "SheIn Mock Neck Ribbed Knit Tee", "t-shirt", 899, 4, "B07MP31CQB", "army green"),
        ("ORD-3003", "ravi", "Cable Knit Sweater Set (hoodie crop top and sweatpants)", "sweater set", 2799, 2, "B0BK95GYJ2", "black"),
        ("ORD-3004", "ravi", "Zexxxy Striped Short-Sleeve Pyjama Set", "pyjama set", 1099, 5, "B07D2C7Q8S", "grey"),
        ("ORD-3005", "neha", "Hue Chevron Quilted Control-Top Tights", "tights", 1499, 3, "B01IW8LO0G", "navy"),
        ("ORD-3006", "neha", "Delia's Girls Thermal Knit Shirt and Pant Set, 2-pack", "thermal set", 1199, 6, "B0773WMP4K", "lavender and pink"),
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
        c.executemany("INSERT OR IGNORE INTO orders (order_id, user_id, customer, product, product_type, category, price, "
                      "delivered_date, sku, colour) VALUES (?,?,?,?,?,?,?,?,?,?)", [
            (oid, ids[u][0], ids[u][1], product, ptype, "Clothing", price, _days_ago(days), sku, colour)
            for oid, u, product, ptype, price, days, sku, colour in _orders()])
        c.executemany("UPDATE users SET address = ? WHERE username = ? AND address IS NULL", [(a, u) for u, a in ADDRESSES.items()])
        c.executemany("UPDATE orders SET sku = ?, colour = ? WHERE order_id = ? AND sku IS NULL",
                      [(sku, colour, oid) for oid, *_, sku, colour in _orders()])
        # The demo order list is defined here: drop orders no longer in it, unless a case refers to them.
        keep = [o[0] for o in _orders()]
        c.execute(f"DELETE FROM orders WHERE order_id NOT IN ({','.join('?' * len(keep))}) "
                  "AND order_id NOT IN (SELECT order_id FROM cases)", keep)


def init_db(reset: bool = False) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    with conn() as c:
        version = c.execute("PRAGMA user_version").fetchone()[0]
        has_tables = c.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='cases'").fetchone()[0]
        if has_tables and version < 2:
            # The first prototype's tables have a different shape; its demo data is not kept.
            for t in ("orders", "cases", "events"):
                c.execute(f"DROP TABLE IF EXISTS {t}")
        elif has_tables and version == 2:
            for col in ("sku", "colour"):
                c.execute(f"ALTER TABLE orders ADD COLUMN {col} TEXT")
        users = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
        if users and "courier" not in users[0]:
            # v4 adds the pickup-partner role and addresses. SQLite can't change a CHECK, so the table is rebuilt (ids kept).
            c.executescript("""
                ALTER TABLE users RENAME TO users_v3;
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL, name TEXT, role TEXT NOT NULL CHECK (role IN ('customer', 'reviewer', 'courier')),
                    password_hash TEXT NOT NULL, created_at TEXT, address TEXT);
                INSERT INTO users (id, username, name, role, password_hash, created_at)
                    SELECT id, username, name, role, password_hash, created_at FROM users_v3;
                DROP TABLE users_v3;""")
        c.executescript(SCHEMA)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    seed()


def reset_demo() -> dict:
    """Start the demo again: remove every return with its photos, agent steps, reviews and history, number cases
    from 1 again and re-seed the orders with fresh delivery dates. Users and sessions are kept."""
    tables = ("cases", "media", "tool_calls", "reviews", "events", "pickups")
    with conn() as c:
        files = [r["file"] for r in c.execute("SELECT file FROM media")]
        files += [r["photo"] for r in c.execute("SELECT photo FROM pickups WHERE photo IS NOT NULL")]
        n_cases = c.execute("SELECT count(*) FROM cases").fetchone()[0]
        for t in tables:
            c.execute(f"DELETE FROM {t}")
        c.execute(f"DELETE FROM sqlite_sequence WHERE name IN ({','.join('?' * len(tables))})", tables)
    seed(reset=True)
    removed = 0
    for f in files:  # only the stored file names from the media table, always inside the upload folder
        p = config.UPLOAD_DIR / f
        if p.parent == config.UPLOAD_DIR and p.is_file():
            p.unlink(); removed += 1
    return {"cases": n_cases, "photos": removed}


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
        row = c.execute("SELECT id, username, name, role, address FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def first_courier():
    with conn() as c:
        row = c.execute("SELECT id, username, name FROM users WHERE role = 'courier' ORDER BY id LIMIT 1").fetchone()
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


# ---------------------------------------------------------------- doorstep pickups
PICKUP_JSON = ("checks",)


def save_pickup(case_id: int, **fields) -> None:
    """Create the case's pickup, or start it again (a case has one pickup row; a re-approval reuses it)."""
    fields = _encode(fields, PICKUP_JSON)
    ts = now()
    with conn() as c:
        if c.execute("SELECT 1 FROM pickups WHERE case_id = ?", (case_id,)).fetchone():
            cols = ", ".join(f"{k} = ?" for k in fields)
            c.execute(f"UPDATE pickups SET {cols}, updated_at = ? WHERE case_id = ?", (*fields.values(), ts, case_id))
        else:
            cols = ", ".join(fields)
            c.execute(f"INSERT INTO pickups (case_id, {cols}, created_at, updated_at) VALUES (?, {','.join('?' * len(fields))}, ?, ?)",
                      (case_id, *fields.values(), ts, ts))


def update_pickup(pickup_id: int, **fields) -> None:
    fields = _encode(fields, PICKUP_JSON)
    fields["updated_at"] = now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE pickups SET {cols} WHERE id = ?", (*fields.values(), pickup_id))


def get_pickup(pickup_id: int):
    with conn() as c:
        row = c.execute("SELECT * FROM pickups WHERE id = ?", (pickup_id,)).fetchone()
    return _decode(row, PICKUP_JSON) if row else None


def pickup_for_case(case_id: int):
    with conn() as c:
        row = c.execute("SELECT * FROM pickups WHERE case_id = ?", (case_id,)).fetchone()
    return _decode(row, PICKUP_JSON) if row else None


def pickup_by_photo(file: str):
    with conn() as c:
        row = c.execute("SELECT * FROM pickups WHERE photo = ?", (file,)).fetchone()
    return _decode(row, PICKUP_JSON) if row else None


def list_pickups(courier_id: int):
    with conn() as c:
        rows = c.execute("SELECT * FROM pickups WHERE courier_id = ? ORDER BY scheduled_for, id", (courier_id,)).fetchall()
    return [_decode(r, PICKUP_JSON) for r in rows]
