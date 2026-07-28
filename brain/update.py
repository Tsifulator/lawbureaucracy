"""Incremental refresh: pull any new decision metadata, embed anything not yet
done, then tell a running server to reload its index. Safe to run daily.

  python update.py            # scrape new + ingest everything still pending
  python update.py --new-only # scrape + ingest only current & last year
"""
import argparse
import urllib.request

import scrape
import ingest
from config import HOST, PORT


def reload_server():
    try:
        req = urllib.request.Request(f"http://{HOST}:{PORT}/reload", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=120) as r:
            print("reloaded:", r.read().decode())
    except Exception as e:
        print(f"(server not running or reload failed: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-only", action="store_true",
                    help="only ingest current+previous year (fast daily mode)")
    a = ap.parse_args()

    scrape.run()

    ing = argparse.Namespace(year=None, min_year=None, limit=None)
    if a.new_only:
        from datetime import datetime, timezone
        ing.min_year = datetime.now(timezone.utc).year - 1
    ingest.run(ing)

    reload_server()
