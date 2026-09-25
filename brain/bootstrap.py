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
        usage = shutil.disk_usage(config.DATA)
        _log(f"volume: {usage.used / 1e9:.1f} GB used / {usage.total / 1e9:.1f} GB total")
    return ok


if __name__ == "__main__":
    sys.exit(0 if ensure_assets() else 1)
