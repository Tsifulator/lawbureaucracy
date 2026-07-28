"""Download decision PDFs, extract text, chunk, embed (bge-m3), store vectors.

Resumable: skips decisions already embedded. Bounded by CLI flags so you can
prove the loop on recent years first, then backfill the rest in the background.

  python ingest.py --year 2025 --limit 300
  python ingest.py --min-year 2023
  python ingest.py                       # everything not yet done
"""
import argparse
import fcntl
import io
import time
import urllib.request
from datetime import datetime, timezone

from pypdf import PdfReader

import db
from config import (
    PDF_DIR, DATA, USER_AGENT, CHUNK_CHARS, CHUNK_OVERLAP, MIN_TEXT_CHARS,
    HOST, PORT,
)
from embed import embed

LOCK_PATH = DATA / "ingest.lock"
RELOAD_EVERY = 100  # tell the running server to refresh its index this often


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def reload_server():
    """Best-effort: ask a running server to reload so new docs become searchable."""
    try:
        req = urllib.request.Request(f"http://{HOST}:{PORT}/reload", method="POST", data=b"")
        urllib.request.urlopen(req, timeout=180).read()
    except Exception:
        pass  # server may not be running; that's fine


def download_pdf(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def extract_pages(pdf_bytes):
    """Yield (page_number, text) for each page with usable text."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        yield i, txt


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out


def select_decisions(con, args):
    q = "SELECT * FROM decisions WHERE status IN ('new','error')"
    params = []
    if args.year:
        q += " AND year = ?"
        params.append(str(args.year))
    if args.min_year:
        q += " AND CAST(year AS INTEGER) >= ?"
        params.append(int(args.min_year))
    q += " ORDER BY CAST(year AS INTEGER) DESC, CAST(number AS INTEGER) DESC"
    if args.limit:
        q += f" LIMIT {int(args.limit)}"
    return con.execute(q, params).fetchall()


def ingest_one(con, d):
    pdf_path = PDF_DIR / f"Apofasi-{d['number']}-{d['year']}.pdf"
    try:
        download_pdf(d["pdf_url"], pdf_path)
        pdf_bytes = pdf_path.read_bytes()
    except Exception as e:
        con.execute("UPDATE decisions SET status='error' WHERE id=?", (d["id"],))
        con.commit()
        return f"download-error: {e}"

    # gather text per page, chunk, embed
    total_chars = 0
    idx = 0
    con.execute("DELETE FROM chunks WHERE decision_id=?", (d["id"],))  # idempotent re-ingest
    for page_no, page_text in extract_pages(pdf_bytes):
        total_chars += len(page_text)
        for chunk in chunk_text(page_text):
            if len(chunk.strip()) < 40:
                continue
            vec = embed(chunk)
            db.add_chunk(con, d["id"], idx, page_no, chunk, vec)
            idx += 1

    if total_chars < MIN_TEXT_CHARS:
        con.execute(
            "UPDATE decisions SET status='no_text', text_chars=?, n_chunks=0, ingested_at=? WHERE id=?",
            (total_chars, _now(), d["id"]),
        )
        con.commit()
        return f"no-text ({total_chars} chars) — likely scanned, needs OCR"

    con.execute(
        "UPDATE decisions SET status='embedded', text_chars=?, n_chunks=?, ingested_at=? WHERE id=?",
        (total_chars, idx, _now(), d["id"]),
    )
    con.commit()
    return f"embedded {idx} chunks ({total_chars} chars)"


def run(args):
    # single-writer lock: prevents the daily job and a long backfill from
    # embedding the same decisions at once (they'd race on chunks + Ollama).
    lock_f = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another ingest is already running; skipping this one.")
        lock_f.close()
        return

    try:
        con = db.connect()
        db.init(con)
        todo = select_decisions(con, args)
        print(f"{len(todo)} decisions to ingest")
        for n, d in enumerate(todo, 1):
            tag = f"{d['number']}/{d['year']}"
            try:
                msg = ingest_one(con, d)
            except Exception as e:
                con.execute("UPDATE decisions SET status='error' WHERE id=?", (d["id"],))
                con.commit()
                msg = f"ERROR: {e}"
            print(f"[{n}/{len(todo)}] {tag}: {msg}")
            if n % RELOAD_EVERY == 0:
                reload_server()  # make progress searchable during long backfills
                print(f"  … reloaded server index at {n}/{len(todo)}")
            time.sleep(0.3)  # be polite to the server
        con.close()
        reload_server()
        print("INGEST DONE")
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=str, help="only this year")
    ap.add_argument("--min-year", type=int, help="only decisions from this year onward")
    ap.add_argument("--limit", type=int, help="cap number of decisions")
    run(ap.parse_args())
