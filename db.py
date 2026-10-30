"""SQLite data layer: users, chat history, feedback, tickets, leave balances and requests."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from app import config
from app.security import hash_password, verify_password

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL, full_name TEXT NOT NULL, role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
  question TEXT NOT NULL, answer TEXT NOT NULL, sources TEXT DEFAULT '[]', route TEXT DEFAULT 'rag',
  ticket_id INTEGER, latency_ms INTEGER DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL, rating INTEGER NOT NULL, comment TEXT DEFAULT '', created_at TEXT NOT NULL,
  UNIQUE(message_id, user_id));
CREATE TABLE IF NOT EXISTS tickets(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
  question TEXT NOT NULL, status TEXT DEFAULT 'open', reply TEXT DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS leave_balances(user_id INTEGER NOT NULL, leave_type TEXT NOT NULL,
  balance REAL NOT NULL, PRIMARY KEY(user_id, leave_type));
CREATE TABLE IF NOT EXISTS leave_requests(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
  leave_type TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL, days REAL NOT NULL,
  reason TEXT DEFAULT '', status TEXT DEFAULT 'pending', created_at TEXT NOT NULL);
"""

# Demo accounts - change or remove before any real deployment
DEMO_USERS = [
    ("employee", "emp123", "Aarav Sharma", "employee"),
    ("manager", "mgr123", "Meera Iyer", "manager"),
    ("hr", "hr123", "Rohan Gupta", "hr"),
]
DEFAULT_BALANCES = {"casual": 12, "sick": 10, "earned": 18}  # matches the leave policy


@contextmanager
def conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _now():
    return datetime.utcnow().isoformat(timespec="seconds")


def _rows(sql, params=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def _one(sql, params=()):
    rows = _rows(sql, params)
    return rows[0] if rows else None


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            for username, pw, name, role in DEMO_USERS:
                cur = c.execute("INSERT INTO users(username,password_hash,full_name,role) VALUES(?,?,?,?)",
                                (username, hash_password(pw), name, role))
                for lt, bal in DEFAULT_BALANCES.items():
                    c.execute("INSERT INTO leave_balances VALUES(?,?,?)", (cur.lastrowid, lt, bal))


# ---- users ----
def get_user(user_id):
    return _one("SELECT id,username,full_name,role FROM users WHERE id=?", (user_id,))


def get_user_by_username(username):
    return _one("SELECT id,username,full_name,role FROM users WHERE username=?", (username,))


def authenticate(username, password):
    row = _one("SELECT * FROM users WHERE username=?", (username,))
    if row and verify_password(password, row["password_hash"]):
        return {k: row[k] for k in ("id", "username", "full_name", "role")}
    return None


# ---- chat history & feedback ----
def add_message(user_id, question, answer, sources, route, latency_ms, ticket_id=None):
    with conn() as c:
        cur = c.execute("INSERT INTO messages(user_id,question,answer,sources,route,ticket_id,latency_ms,created_at)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (user_id, question, answer, json.dumps(sources, ensure_ascii=False), route,
                         ticket_id, latency_ms, _now()))
        return cur.lastrowid


def get_history(user_id, limit=20):
    rows = _rows("SELECT * FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    for r in rows:
        r["sources"] = json.loads(r["sources"] or "[]")
    return rows[::-1]


def add_feedback(message_id, user_id, rating, comment=""):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO feedback(message_id,user_id,rating,comment,created_at) VALUES(?,?,?,?,?)",
                  (message_id, user_id, rating, comment, _now()))


def negative_feedback(limit=20):
    return _rows("SELECT f.message_id, f.comment, f.created_at, m.question, m.answer, m.route FROM feedback f "
                 "JOIN messages m ON m.id=f.message_id WHERE f.rating<0 ORDER BY f.id DESC LIMIT ?", (limit,))


# ---- tickets ----
def create_ticket(user_id, question):
    with conn() as c:
        return c.execute("INSERT INTO tickets(user_id,question,created_at) VALUES(?,?,?)",
                         (user_id, question, _now())).lastrowid


def list_tickets(user_id=None):
    sql = "SELECT t.*, u.full_name FROM tickets t JOIN users u ON u.id=t.user_id"
    if user_id is None:
        return _rows(sql + " ORDER BY t.id DESC")
    return _rows(sql + " WHERE t.user_id=? ORDER BY t.id DESC", (user_id,))


def resolve_ticket(ticket_id, reply):
    with conn() as c:
        c.execute("UPDATE tickets SET status='resolved', reply=? WHERE id=?", (reply, ticket_id))


# ---- leave ----
def get_balances(user_id):
    return {r["leave_type"]: r["balance"] for r in
            _rows("SELECT leave_type,balance FROM leave_balances WHERE user_id=?", (user_id,))}


def adjust_balance(user_id, leave_type, delta):
    with conn() as c:
        c.execute("UPDATE leave_balances SET balance=balance+? WHERE user_id=? AND leave_type=?",
                  (delta, user_id, leave_type))


def add_leave_request(user_id, leave_type, start, end, days, reason):
    with conn() as c:
        return c.execute("INSERT INTO leave_requests(user_id,leave_type,start_date,end_date,days,reason,created_at)"
                         " VALUES(?,?,?,?,?,?,?)", (user_id, leave_type, start, end, days, reason, _now())).lastrowid


def list_leave_requests(user_id=None, status=None):
    sql, params, where = ("SELECT r.*, u.full_name FROM leave_requests r JOIN users u ON u.id=r.user_id", [], [])
    if user_id is not None:
        where.append("r.user_id=?"); params.append(user_id)
    if status:
        where.append("r.status=?"); params.append(status)
    if where:
        sql += " WHERE " + " AND ".join(where)
    return _rows(sql + " ORDER BY r.id DESC", params)


def get_leave_request(request_id):
    return _one("SELECT * FROM leave_requests WHERE id=?", (request_id,))


def set_leave_status(request_id, status):
    with conn() as c:
        c.execute("UPDATE leave_requests SET status=? WHERE id=?", (status, request_id))


# ---- analytics ----
def analytics():
    total = _one("SELECT COUNT(*) n, COALESCE(AVG(latency_ms),0) lat FROM messages")
    routes = {r["route"]: r["n"] for r in _rows("SELECT route, COUNT(*) n FROM messages GROUP BY route")}
    fb = _one("SELECT SUM(rating>0) up, SUM(rating<0) down FROM feedback")
    up, down = (fb["up"] or 0), (fb["down"] or 0)
    return {
        "total_questions": total["n"],
        "avg_latency_ms": round(total["lat"]),
        "routes": routes,
        "escalation_rate": round(routes.get("escalated", 0) / total["n"], 3) if total["n"] else 0,
        "helpful_rate": round(up / (up + down), 3) if (up + down) else None,
        "thumbs_up": up, "thumbs_down": down,
        "top_questions": _rows("SELECT LOWER(TRIM(question)) question, COUNT(*) count FROM messages "
                               "GROUP BY 1 ORDER BY count DESC LIMIT 10"),
        "unanswered": _rows("SELECT question, created_at FROM messages WHERE route='escalated' "
                            "ORDER BY id DESC LIMIT 10"),
    }
