"""Scrape all EADHSY decision metadata from the wpDataTables server-side API.

Flow: load /apofaseis/ once to grab the frontend nonce, then POST paged
DataTables requests to admin-ajax.php and page through all ~15,600 rows.
Only metadata + PDF URL is captured here; PDFs are fetched by ingest.py.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import db
from config import APOFASEIS_PAGE, AJAX_URL, USER_AGENT

PAGE = 500  # rows per request
NONCE_RE = re.compile(r'id="wdtNonceFrontendServerSide_5"[^>]*value="([^"]+)"')
PDF_RE = re.compile(r"href='([^']+\.pdf[^']*)'", re.I)
# PDF filenames look like Apofasi-1614-2025.pdf -> (number, year)
NUMYEAR_RE = re.compile(r"Apofasi[-_](\d+)[-_](\d+)\.pdf", re.I)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_session():
    """Fetch the page, return (nonce, cookie_header)."""
    req = urllib.request.Request(APOFASEIS_PAGE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
        cookies = r.headers.get_all("Set-Cookie") or []
    cookie_hdr = "; ".join(c.split(";")[0] for c in cookies)
    m = NONCE_RE.search(html)
    if not m:
        raise RuntimeError("Could not find wpDataTables nonce on /apofaseis/")
    return m.group(1), cookie_hdr


def fetch_page(nonce, cookie_hdr, start, length=PAGE):
    data = {
        "draw": "1", "start": str(start), "length": str(length),
        "search[value]": "", "search[regex]": "false",
        "order[0][column]": "0", "order[0][dir]": "desc",
        "wdtNonce": nonce,
    }
    for i in range(11):  # table has 11 columns
        data[f"columns[{i}][data]"] = str(i)
        data[f"columns[{i}][name]"] = ""
        data[f"columns[{i}][searchable]"] = "true"
        data[f"columns[{i}][orderable]"] = "true"
        data[f"columns[{i}][search][value]"] = ""
        data[f"columns[{i}][search][regex]"] = "false"
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        AJAX_URL, data=body,
        headers={
            "User-Agent": USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": APOFASEIS_PAGE,
            "Cookie": cookie_hdr,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def parse_row(row):
    """Turn a raw API row (list of cells) into a decision dict, or None."""
    cells = [c if c is not None else "" for c in row]
    joined = " ".join(str(c) for c in cells)
    mpdf = PDF_RE.search(joined)
    if not mpdf:
        return None
    pdf_url = mpdf.group(1)
    num, year = "", ""
    mny = NUMYEAR_RE.search(pdf_url)
    if mny:
        num, year = mny.group(1), mny.group(2)
    # dtype = first cell that looks like a Greek all-caps status word
    dtype = ""
    for c in cells:
        s = str(c).strip()
        if s and s.isupper() and any("Α" <= ch <= "Ω" for ch in s):
            dtype = s
            break
    return {
        "id": str(cells[0]),
        "number": num,
        "year": year,
        "dtype": dtype,
        "pdf_url": pdf_url,
        "source_date": str(cells[2]) if len(cells) > 2 else "",
        "raw_json": json.dumps(cells, ensure_ascii=False),
        "scraped_at": _now(),
    }


def run():
    con = db.connect()
    db.init(con)
    nonce, cookie_hdr = get_session()
    print(f"nonce={nonce}")

    first = fetch_page(nonce, cookie_hdr, 0, 1)
    total = int(first.get("recordsTotal", 0))
    print(f"recordsTotal = {total}")

    saved = skipped = 0
    start = 0
    while start < total:
        try:
            j = fetch_page(nonce, cookie_hdr, start, PAGE)
        except Exception as e:
            print(f"  page {start} failed ({e}); refreshing nonce and retrying")
            nonce, cookie_hdr = get_session()
            time.sleep(2)
            j = fetch_page(nonce, cookie_hdr, start, PAGE)
        rows = j.get("data", [])
        if not rows:
            print(f"  empty page at start={start}; stopping")
            break
        for row in rows:
            d = parse_row(row)
            if d:
                db.upsert_decision(con, d)
                saved += 1
            else:
                skipped += 1
        con.commit()
        start += len(rows)
        print(f"  {start}/{total}  (saved={saved} skipped={skipped})")
        time.sleep(0.4)  # be polite

    con.close()
    print(f"DONE. saved={saved} skipped={skipped}")


if __name__ == "__main__":
    run()
