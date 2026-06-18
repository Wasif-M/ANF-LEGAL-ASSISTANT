"""SQLite persistence layer: users, sessions, conversations, messages.

Single-file DB (app.db) next to the API. Connections are opened per call —
cheap for SQLite and avoids cross-thread issues with FastAPI's threadpool.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).parent / "app.db"

_PBKDF2_ITERATIONS = 200_000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                thinking TEXT,          -- JSON list of thinking steps (assistant only)
                sections TEXT,          -- JSON list of extracted section refs (assistant only)
                rating INTEGER,         -- 1-5 star user rating (assistant only)
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
            """
        )
        # Migration: add rating column (1-5 stars) to messages created before it existed
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        if "rating" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN rating INTEGER")


# ─── Password hashing ───

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)


# ─── Users / sessions ───

def create_user(username: str, email: str, password: str) -> dict:
    password_hash, salt = hash_password(password)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash, salt, _utcnow()),
        )
        user_id = cur.lastrowid
    return {"id": user_id, "username": username.strip(), "email": email.strip().lower()}


def authenticate_user(username_or_email: str, password: str) -> Optional[dict]:
    ident = username_or_email.strip()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?", (ident, ident.lower())
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"], row["salt"]):
        return None
    return {"id": row["id"], "username": row["username"], "email": row["email"]}


def create_session(user_id: int) -> str:
    token = secrets.token_hex(32)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _utcnow()),
        )
    return token


def get_user_by_token(token: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.username, u.email FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ─── Conversations ───

def _conv_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_conversation(user_id: int, title: str = "New Chat") -> dict:
    now = _utcnow()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
        conv_id = cur.lastrowid
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [_conv_dict(r) for r in rows]


def get_conversation(conv_id: int, user_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id)
        ).fetchone()
    return _conv_dict(row) if row else None


def rename_conversation(conv_id: int, user_id: int, title: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, _utcnow(), conv_id, user_id),
        )
    return cur.rowcount > 0


def delete_conversation(conv_id: int, user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id)
        )
    return cur.rowcount > 0


# ─── Messages ───

def add_message(
    conversation_id: int,
    role: str,
    content: str,
    thinking: Optional[list] = None,
    sections: Optional[list] = None,
) -> dict:
    now = _utcnow()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, thinking, sections, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                json.dumps(thinking) if thinking else None,
                json.dumps(sections) if sections else None,
                now,
            ),
        )
        msg_id = cur.lastrowid
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )
    return {"id": msg_id, "role": role, "content": content, "created_at": now}


def list_messages(conversation_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "thinking": json.loads(r["thinking"]) if r["thinking"] else [],
                "sections": json.loads(r["sections"]) if r["sections"] else [],
                "rating": r["rating"],
                "created_at": r["created_at"],
            }
        )
    return out


# ─── Ratings ───

def set_message_rating(message_id: int, user_id: int, rating: int) -> bool:
    """Store a 1-5 star rating on an assistant message owned by this user."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE messages SET rating = ?
            WHERE id = ? AND role = 'assistant'
              AND conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)
            """,
            (rating, message_id, user_id),
        )
    return cur.rowcount > 0


def get_rating_stats() -> dict:
    """Aggregate rating stats across all users — for accuracy/QA review."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS rated, AVG(rating) AS average FROM messages WHERE rating IS NOT NULL"
        ).fetchone()
        dist_rows = conn.execute(
            "SELECT rating, COUNT(*) AS n FROM messages WHERE rating IS NOT NULL GROUP BY rating"
        ).fetchall()
    distribution = {str(stars): 0 for stars in range(1, 6)}
    for r in dist_rows:
        distribution[str(r["rating"])] = r["n"]
    return {
        "rated_responses": row["rated"],
        "average_rating": round(row["average"], 2) if row["average"] is not None else None,
        "distribution": distribution,
    }
