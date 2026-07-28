"""Server-side chat storage (SQLite). Kept in its own DB so it never contends
with the corpus DB / backfill. Because it lives with the brain, moving the brain
to a data center moves the whole chat history with it."""
import json
import sqlite3
from datetime import datetime, timezone

from config import DATA

CHATS_DB = DATA / "chats.db"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    con = sqlite3.connect(CHATS_DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init(con):
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,
            created_at  TEXT,
            updated_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            role        TEXT,          -- 'user' | 'assistant'
            content     TEXT,
            sources     TEXT,          -- JSON array (assistant turns)
            created_at  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id);
        """
    )
    con.commit()


def create_chat(con, title="Νέα συζήτηση"):
    cur = con.execute(
        "INSERT INTO chats (title, created_at, updated_at) VALUES (?,?,?)",
        (title, _now(), _now()),
    )
    con.commit()
    return cur.lastrowid


def list_chats(con):
    rows = con.execute(
        """
        SELECT c.id, c.title, c.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS n
        FROM chats c ORDER BY c.updated_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_messages(con, chat_id):
    rows = con.execute(
        "SELECT role, content, sources FROM messages WHERE chat_id=? ORDER BY id",
        (chat_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = {"role": r["role"], "content": r["content"]}
        if r["sources"]:
            try:
                d["sources"] = json.loads(r["sources"])
            except Exception:
                d["sources"] = []
        out.append(d)
    return out


def add_message(con, chat_id, role, content, sources=None):
    con.execute(
        "INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
        (chat_id, role, content, json.dumps(sources or [], ensure_ascii=False), _now()),
    )
    con.execute("UPDATE chats SET updated_at=? WHERE id=?", (_now(), chat_id))
    con.commit()


def set_title(con, chat_id, title):
    con.execute("UPDATE chats SET title=? WHERE id=?", (title[:80], chat_id))
    con.commit()


def message_count(con, chat_id):
    return con.execute(
        "SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,)
    ).fetchone()[0]


def delete_chat(con, chat_id):
    con.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    con.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    con.commit()
