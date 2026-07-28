"""SQLite storage: decision metadata + embedded text chunks.

Vectors are stored as raw float32 bytes (numpy). We DON'T use a vector
extension because this Python's sqlite3 has no loadable-extension support;
numpy brute-force cosine over a few thousand docs is instant anyway.
"""
import sqlite3
import numpy as np
from config import DB_PATH, EMBED_DIM


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init(con):
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id           TEXT PRIMARY KEY,   -- wpDataTables row id
            number       TEXT,               -- decision number (from PDF name)
            year         TEXT,
            dtype        TEXT,               -- ΟΡΙΣΤΙΚΗ / ΠΡΟΣΩΡΙΝΗ / ...
            pdf_url      TEXT,
            source_date  TEXT,               -- created/modified date from source
            raw_json     TEXT,               -- original API row, for safety
            status       TEXT DEFAULT 'new', -- new|no_text|embedded|error
            text_chars   INTEGER DEFAULT 0,
            n_chunks     INTEGER DEFAULT 0,
            scraped_at   TEXT,
            ingested_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id  TEXT NOT NULL REFERENCES decisions(id),
            idx          INTEGER,
            page         INTEGER,
            text         TEXT,
            vec          BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_decision ON chunks(decision_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
        """
    )
    con.commit()


def upsert_decision(con, d):
    con.execute(
        """
        INSERT INTO decisions (id, number, year, dtype, pdf_url, source_date, raw_json, scraped_at)
        VALUES (:id, :number, :year, :dtype, :pdf_url, :source_date, :raw_json, :scraped_at)
        ON CONFLICT(id) DO UPDATE SET
            number=excluded.number, year=excluded.year, dtype=excluded.dtype,
            pdf_url=excluded.pdf_url, source_date=excluded.source_date,
            raw_json=excluded.raw_json, scraped_at=excluded.scraped_at
        """,
        d,
    )


def vec_to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def add_chunk(con, decision_id, idx, page, text, vec):
    con.execute(
        "INSERT INTO chunks (decision_id, idx, page, text, vec) VALUES (?,?,?,?,?)",
        (decision_id, idx, page, text, vec_to_blob(vec)),
    )


def load_matrix(con):
    """Return (ids, decision_ids, texts, M) where M is (n, EMBED_DIM) float32, L2-normalized."""
    rows = con.execute(
        "SELECT id, decision_id, text, vec FROM chunks ORDER BY id"
    ).fetchall()
    if not rows:
        return [], [], [], np.zeros((0, EMBED_DIM), dtype=np.float32)
    ids = [r["id"] for r in rows]
    dec_ids = [r["decision_id"] for r in rows]
    texts = [r["text"] for r in rows]
    M = np.stack([blob_to_vec(r["vec"]) for r in rows]).astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms
    return ids, dec_ids, texts, M
