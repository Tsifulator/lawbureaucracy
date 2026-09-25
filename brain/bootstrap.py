"""First-boot asset fetch for hosted deploys.

The index (450 MB) and the bge-m3 ONNX model (1.1 GB) are far too big to bake
into a container image, and rebuilding them in the cloud would mean re-scraping
and re-embedding 15,639 PDFs. Instead they're pulled once onto the mounted
volume; every later boot finds them already there and starts immediately.

Safe to run on every boot — it's a no-op once the files exist. Downloads land on
a .part file and are renamed only on success, so a killed deploy can't leave a
half-written database that looks complete.
"""
import gzip
import os
import shutil
import sys
import urllib.request

import config

UA = {"User-Agent": "lawbureaucracy-bootstrap"}


def _log(msg):
    print(f"[bootstrap] {msg}", flush=True)


def _download(url, dest, gunzip=False):
    """Stream url -> dest, atomically. Returns True if it actually downloaded."""
    if dest.exists() and dest.stat().st_size > 0:
        _log(f"{dest.name} already present ({dest.stat().st_size / 1e6:.0f} MB) — skipping")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    _log(f"downloading {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        raw = gzip.GzipFile(fileobj=r) if gunzip else r
        done = 0
        step = 0
        with open(part, "wb") as f:
            while True:
                buf = raw.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if done // (50 << 20) > step:  # a line every ~50 MB
                    step = done // (50 << 20)
                    of = ""
                    if total:
                        of = f" / {total/1e6:.0f} MB {'gzipped' if gunzip else 'total'}"
                    _log(f"  … {done/1e6:.0f} MB written{of}")
    part.replace(dest)
    _log(f"{dest.name} ready ({dest.stat().st_size / 1e6:.0f} MB)")
    return True


def repair_numbers():
    """Re-derive number/year on an index built before decnum.py existed.

    The first published index came from a regex that only matched
    Apofasi-<digits>-<year>.pdf, so every prefixed decision (Α831/2025
    suspensions, E11/2021, AA55/2017) was stored with number='' — 4,306 of
    15,690, and 4,284 of the ~5k actually ingested. A volume that already holds
    that index never re-downloads it (the fetch above skips files that exist),
    so the repair has to happen in place. Metadata only, no re-embedding, and
    idempotent — a corrected index reports "already correct" and moves on.
    """
    import sqlite3

    import decnum

    con = sqlite3.connect(config.DB_PATH)
    try:
        rows = con.execute("SELECT id, number, year, pdf_url FROM decisions").fetchall()
        fixes = [
            (num, yr, did)
            for did, number, year, url in rows
            for num, yr in [decnum.from_pdf_url(url)]
            if num and (num != (number or "") or yr != (year or ""))
        ]
        if fixes:
            con.executemany("UPDATE decisions SET number=?, year=? WHERE id=?", fixes)
            con.commit()
            _log(f"repaired {len(fixes)} decision numbers")
        else:
            _log("decision numbers already correct")
    finally:
        con.close()


def ensure_assets():
    ok = True

    try:
        _download(config.DB_URL, config.DB_PATH, gunzip=True)
    except Exception as e:
        _log(f"FATAL: could not fetch the index: {e}")
        ok = False

    if config.EMBED_BACKEND == "onnx":
        try:
            _download(config.ONNX_TOKENIZER_URL, config.ONNX_TOKENIZER_PATH)
            _download(config.ONNX_MODEL_URL, config.ONNX_MODEL_PATH)
        except Exception as e:
            _log(f"FATAL: could not fetch the bge-m3 ONNX model: {e}")
            ok = False

    if ok:
        try:
            repair_numbers()
        except Exception as e:
            # A label problem must never stop the app from serving search.
            _log(f"WARNING: number repair failed ({e}) — serving index as-is")
        usage = shutil.disk_usage(config.DATA)
        _log(f"volume: {usage.used / 1e9:.1f} GB used / {usage.total / 1e9:.1f} GB total")
    return ok


if __name__ == "__main__":
    sys.exit(0 if ensure_assets() else 1)
