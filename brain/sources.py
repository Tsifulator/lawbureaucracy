"""Extra legal sources beyond the ΕΑΔΗΣΥ decisions, with exact citations.

  python sources.py probe                 # check every source, change nothing
  python sources.py run                   # fetch + split + embed everything new
  python sources.py run --only n4412-2016 # one source
  python sources.py list                  # what is in the database, per source

What it imports (configured in sources.json):
  * EU directives from EUR-Lex (Greek text), one document per article/annex:
        cite "Οδηγία 2014/24/ΕΕ, άρθρο 57"
  * Greek laws, one document per article (from a URL, a local file, or a
    consolidated text discovered on the ΕΑΔΗΣΥ site):
        cite "ν. 4412/2016, άρθρο 72"
  * Crawled documents (ΕΑΔΗΣΥ guidance & standard documents, other courts),
    one document per file:   cite "ΕΑΔΗΣΥ — Κατευθυντήρια Οδηγία 25"

Every chunk starts with its citation, so the model sees (and must quote) the
exact label. Re-runs skip anything already embedded with the same content.
Polite by design: honours robots.txt, one request per second, honest UA.
"""
import argparse
import fcntl
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import db
from config import BASE, CHUNK_CHARS, CHUNK_OVERLAP, DATA

MANIFEST = BASE / "sources.json"
CACHE = DATA / "sources"          # downloaded originals, so re-runs don't refetch
LOCK_PATH = DATA / "ingest.lock"  # shared with ingest.py: one embedder at a time
UA = "lawbureaucracy-bot/1.0 (+https://github.com/Tsifulator/lawbureaucracy)"
DELAY = 1.0
EMBED_BATCH = 16

CACHE.mkdir(parents=True, exist_ok=True)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(*a):
    print(*a, flush=True)


# =========================================================================
# fetching (polite)
# =========================================================================
_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
_last_hit: dict[str, float] = {}


def allowed(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    root = f"{p.scheme}://{p.netloc}"
    if root not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(root + "/robots.txt", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                rp.parse(r.read().decode("utf-8", "replace").splitlines())
        except Exception:
            rp = None  # no robots.txt (or unreachable) -> allowed
        _robots[root] = rp
    rp = _robots[root]
    return True if rp is None else rp.can_fetch(UA, url)


def fetch(url: str, timeout: int = 90) -> tuple[bytes, str]:
    """GET with a per-host delay. Returns (body, content-type)."""
    host = urllib.parse.urlparse(url).netloc
    wait = DELAY - (time.time() - _last_hit.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Language": "el,en;q=0.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers.get("Content-Type", "")
    finally:
        _last_hit[host] = time.time()


def cached_fetch(url: str) -> tuple[bytes, str]:
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    body_f, type_f = CACHE / f"{key}.bin", CACHE / f"{key}.type"
    if body_f.exists() and body_f.stat().st_size > 0:
        return body_f.read_bytes(), type_f.read_text() if type_f.exists() else ""
    if not allowed(url):
        raise RuntimeError(f"robots.txt disallows {url}")
    body, ctype = fetch(url)
    body_f.write_bytes(body)
    type_f.write_text(ctype)
    return body, ctype


# =========================================================================
# text extraction: HTML / PDF / DOCX -> lines (or pages)
# =========================================================================
_BLOCK = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5",
          "h6", "table", "section", "article", "dt", "dd", "blockquote", "hr"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.links, self._a = [], 0, [], None

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "head"):
            self.skip += 1
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            self._a = [href, []] if href else None

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "head"):
            self.skip = max(0, self.skip - 1)
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag == "a" and self._a:
            self.links.append((self._a[0], " ".join("".join(self._a[1]).split())))
            self._a = None

    def handle_data(self, data):
        if self.skip:
            return
        self.parts.append(data)
        if self._a is not None:
            self._a[1].append(data)


def html_parse(body: bytes) -> tuple[list[str], list[tuple[str, str]]]:
    """(non-empty text lines, [(href, link text)])"""
    p = _Text()
    p.feed(body.decode("utf-8", "replace"))
    text = html.unescape("".join(p.parts)).replace("\xa0", " ")
    lines = [" ".join(l.split()) for l in text.split("\n")]
    return [l for l in lines if l], p.links


def pdf_pages(body: bytes) -> list[str]:
    from pypdf import PdfReader  # lazy: only needed for PDFs
    out = []
    for page in PdfReader(io.BytesIO(body)).pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return out


def docx_lines(body: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
    out = []
    for p in paras:
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.S))
        t = " ".join(html.unescape(t).split())
        if t:
            out.append(t)
    return out


def kind_of(url: str, ctype: str, body: bytes) -> str:
    u = url.lower().split("?")[0]
    if body[:4] == b"%PDF" or "pdf" in ctype or u.endswith(".pdf"):
        return "pdf"
    if body[:2] == b"PK" or u.endswith(".docx"):
        return "docx"
    return "html"


def to_lines(url: str, ctype: str, body: bytes) -> list[str]:
    k = kind_of(url, ctype, body)
    if k == "pdf":
        lines = []
        for t in pdf_pages(body):
            lines.extend(" ".join(l.split()) for l in t.split("\n"))
        return [l for l in lines if l]
    if k == "docx":
        return docx_lines(body)
    return html_parse(body)[0]


# =========================================================================
# splitting legislation into articles
# =========================================================================
# "Άρθρο 72", "Αρθρο 72Α", "Ά ρ θ ρ ο 72", "Άρθρο 72 - Λόγοι αποκλεισμού"
_ART = re.compile(
    r"^[ΆΑA]\s*ρ\s*θ\s*ρ\s*ο\s+(\d{1,3})\s*([Α-ΩA-Z]?)\b(?:\s*[-–—:.]\s*|\s+)?(.{0,200})$")


def _plausible_heading(inline: str) -> bool:
    """'Άρθρο 2 Ορισμοί' is a heading; 'Άρθρο 72 του ν. 4412/2016 αντικαθίσταται:'
    is a sentence inside an amendment and must not start a new article."""
    t = inline.strip()
    if not t:
        return True
    if t.endswith(":") or t[0].islower() or re.search(r"\d+\s*/\s*\d{4}|\bτου\s+ν\.", t):
        return False
    return len(t) <= 160
_ANNEX = re.compile(r"^ΠΑΡΑΡΤΗΜΑ\s+([IVXLC]+[Α-ΩA-Z]?|\d{1,2}[Α-Ω]?)\b\s*(.{0,200})$")


def split_articles(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Split legislation into (preamble lines, [{"art","title","lines"}]).

    Robust to the two things that break naive splitting:
      * a table of contents (every heading appears twice; the ToC copy has a
        near-empty body) -> keep the occurrence with the longest body;
      * headings quoted inside amending provisions -> enforce article order.
    Annexes ("ΠΑΡΑΡΤΗΜΑ IV") are split too, after the last article.
    """
    cands = []
    for i, l in enumerate(lines):
        m = _ART.match(l)
        if m and _plausible_heading(m.group(3) or ""):
            cands.append((i, "art", m.group(1) + (m.group(2) or ""), int(m.group(1)), m.group(3) or ""))
            continue
        m = _ANNEX.match(l)
        if m:
            cands.append((i, "annex", m.group(1), 10_000, m.group(2) or ""))
    if not cands:
        return lines, []

    # body length of each candidate up to the next candidate
    body_len = []
    for j, c in enumerate(cands):
        end = cands[j + 1][0] if j + 1 < len(cands) else len(lines)
        body_len.append(sum(len(x) for x in lines[c[0] + 1:end]))
    best = {}
    for j, c in enumerate(cands):
        key = (c[1], c[2])
        if key not in best or body_len[j] > body_len[best[key]]:
            best[key] = j
    keep = sorted(best.values())

    # enforce order: article numbers never go backwards; annexes only after articles
    ordered, last = [], -1
    for j in keep:
        c = cands[j]
        if c[3] >= last:
            ordered.append(j)
            last = c[3] if c[1] == "art" else last
        elif c[1] == "annex":
            ordered.append(j)

    docs = []
    for n, j in enumerate(ordered):
        i, kind, label, _, inline_title = cands[j]
        end = cands[ordered[n + 1]][0] if n + 1 < len(ordered) else len(lines)
        body = lines[i + 1:end]
        title = inline_title.strip()
        if not title and body and len(body[0]) <= 160 and not body[0].endswith("."):
            title, body = body[0], body[1:]
        docs.append({"kind": kind, "art": label, "title": title, "lines": body})
    pre = lines[:cands[ordered[0]][0]] if ordered else lines
    return pre, docs


# =========================================================================
# chunk + embed + store
# =========================================================================
def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
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


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def store_doc(con, doc: dict, pieces: list[tuple[int | None, str]], force=False) -> str:
    """doc: id, source, cite, title, pdf_url, number, year, article, dtype.
    pieces: [(page or None, text)]. Embeds only if new or changed."""
    from embed import embed_batch

    full = "\n".join(t for _, t in pieces)
    sha = _sha(full)
    row = con.execute("SELECT status, raw_json FROM decisions WHERE id=?", (doc["id"],)).fetchone()
    if row and row["status"] == "embedded" and not force:
        try:
            if json.loads(row["raw_json"] or "{}").get("sha") == sha:
                return "unchanged"
        except Exception:
            pass

    header = f"[{doc['cite']}]" + (f" {doc['title']}" if doc.get("title") else "")
    chunks = []
    for page, text in pieces:
        for c in chunk_text(text, size=1200):
            if len(c.strip()) >= 40:
                chunks.append((page, f"{header}\n{c}"))

    db.upsert_doc(con, {**doc, "raw_json": json.dumps({"sha": sha, "src": doc.get("pdf_url")}),
                        "scraped_at": _now()})
    con.execute("DELETE FROM chunks WHERE decision_id=?", (doc["id"],))
    if not chunks:
        con.execute("UPDATE decisions SET status='no_text', text_chars=?, n_chunks=0, ingested_at=? "
                    "WHERE id=?", (len(full), _now(), doc["id"]))
        con.commit()
        return "no text"
    for s in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[s:s + EMBED_BATCH]
        vecs = embed_batch([t for _, t in batch])
        for k, ((page, text), vec) in enumerate(zip(batch, vecs)):
            db.add_chunk(con, doc["id"], s + k, page, text, vec)
    con.execute("UPDATE decisions SET status='embedded', text_chars=?, n_chunks=?, ingested_at=? "
                "WHERE id=?", (len(full), len(chunks), _now(), doc["id"]))
    con.commit()
    return f"{len(chunks)} chunks"


def legislation_docs(spec: dict, source: str, lines: list[str], url: str, anchor_fmt=None):
    """Turn split articles into (doc, pieces) pairs with proper citations."""
    pre, arts = split_articles(lines)
    out = []
    base_id = f"{source}:{spec['id']}"
    common = {"source": source, "number": spec["number"], "year": spec["year"],
              "dtype": "ΝΟΜΟΘΕΣΙΑ"}
    if pre and sum(len(l) for l in pre) > 400:
        label = "προοίμιο" if source == "law_eu" else "εισαγωγή"
        out.append(({**common, "id": f"{base_id}:pre", "article": label,
                     "cite": f"{spec['cite']}, {label}", "title": spec.get("title"),
                     "pdf_url": url}, [(None, "\n".join(pre))]))
    for a in arts:
        if a["kind"] == "art":
            cite = f"{spec['cite']}, άρθρο {a['art']}"
            anchor = anchor_fmt(a["art"]) if anchor_fmt else ""
        else:
            cite = f"{spec['cite']}, Παράρτημα {a['art']}"
            anchor = ""
        out.append(({**common, "id": f"{base_id}:{a['kind']}:{a['art']}",
                     "article": a["art"] if a["kind"] == "art" else f"παράρτημα {a['art']}",
                     "cite": cite, "title": a["title"] or None, "pdf_url": url + anchor},
                    [(None, "\n".join(a["lines"]))]))
    return out


# =========================================================================
# the three kinds of source
# =========================================================================
def eurlex_url(celex: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/EL/TXT/HTML/?uri=CELEX:{celex}"


def cellar_fetch(celex: str) -> tuple[bytes, str]:
    """The EU Publications Office's machine endpoint for the same text
    (content negotiation: Greek XHTML). Used when EUR-Lex's page is blocked
    or comes back without the articles (bot challenge page)."""
    url = f"https://publications.europa.eu/resource/celex/{celex}"
    key = hashlib.sha1(("cellar:" + url).encode()).hexdigest()[:16]
    f = CACHE / f"{key}.bin"
    if f.exists() and f.stat().st_size > 0:
        return f.read_bytes(), "text/html"
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/xhtml+xml, text/html;q=0.9",
        "Accept-Language": "ell, el"})
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read()
    f.write_bytes(body)
    return body, "text/html"


def eu_items(spec):
    url = eurlex_url(spec["celex"])
    lines = []
    try:
        body, ctype = cached_fetch(url)
        lines = to_lines(url, ctype, body)
    except Exception as e:
        log(f"   EUR-Lex: {e} — δοκιμή Cellar")
    if not split_articles(lines)[1]:
        body, ctype = cellar_fetch(spec["celex"])
        lines = to_lines(url, ctype, body)
    return legislation_docs(spec, "law_eu", lines, url,
                            anchor_fmt=lambda a: f"#art_{re.sub(r'[^0-9]', '', a)}")


def law_text(spec, discovered: dict) -> tuple[list[str], str]:
    if spec.get("file"):
        p = Path(spec["file"])
        p = p if p.is_absolute() else BASE / p
        body = p.read_bytes()
        return to_lines(p.name, "", body), spec.get("url") or p.as_uri()
    url = spec.get("url") or discovered.get(spec["id"])
    if not url:
        raise RuntimeError("no url/file set and nothing discovered — see `probe` output")
    body, ctype = cached_fetch(url)
    return to_lines(url, ctype, body), url


def law_items(spec, discovered):
    lines, url = law_text(spec, discovered)
    return legislation_docs(spec, "law_gr", lines, url)


_NUM_REF = re.compile(r"(?<!\d)([A-ZΑ-Ω]{0,2}\d{1,5})\s*/\s*((?:19|20)\d{2})(?!\d)")


def crawl(spec, max_pages=None, log_pages=False):
    """Breadth-first crawl of a site; returns [(url, link text)] of documents."""
    dom = re.compile(rf"(^|\.)({spec['domain']})$")
    doc_re = re.compile(spec["doc_match"])
    ex_re = re.compile(spec["exclude"]) if spec.get("exclude") else None
    max_pages = max_pages or spec.get("max_pages", 100)
    seen, docs, queue, pages = set(), {}, [(u, 0) for u in spec["start"]], 0
    while queue and pages < max_pages:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if not allowed(url):
            if log_pages:
                log(f"   robots.txt disallows {url}")
            continue
        try:
            body, ctype = fetch(url, timeout=30)
        except Exception as e:
            if log_pages:
                log(f"   ! {url}: {e}")
            continue
        pages += 1
        if kind_of(url, ctype, body) != "html":
            continue
        _, links = html_parse(body)
        for href, text in links:
            absu = urllib.parse.urljoin(url, href).split("#")[0]
            pu = urllib.parse.urlparse(absu)
            if pu.scheme not in ("http", "https") or not dom.search(pu.netloc):
                continue
            hay = urllib.parse.unquote(absu) + " " + text
            if re.search(r"\.(jpe?g|png|gif|svg|webp|zip|rar|xlsx?|pptx?|mp4|mp3|ics|xml|css|js)$", pu.path, re.I):
                continue
            is_file = re.search(r"\.(pdf|docx?)$", pu.path, re.I)
            if is_file:
                if doc_re.search(hay) and not (ex_re and ex_re.search(hay)):
                    docs.setdefault(absu, text)
            elif depth < spec.get("depth", 2) and absu not in seen:
                queue.append((absu, depth + 1))
    return list(docs.items())[:spec.get("max_docs", 300)], pages


def crawl_doc_meta(spec, url, text):
    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    label = text.strip() or re.sub(r"[-_]+", " ", name.rsplit(".", 1)[0])
    label = " ".join(label.split())[:90]
    m = _NUM_REF.search(text) or _NUM_REF.search(name)
    number, year = (m.group(1), m.group(2)) if m else (None, None)
    if spec["source"] in ("elsyn", "ste"):
        if not m:
            return None  # a court document we can't cite by number isn't useful
        cite = f"{spec['cite_prefix']} {number}/{year}"
    else:
        g = re.search(r"(?i)κατευθυντήρια\s+οδηγία\s*(?:αρ\.?\s*)?(\d+)", text)
        cite = (f"{spec['cite_prefix']} — Κατευθυντήρια Οδηγία {g.group(1)}" if g
                else f"{spec['cite_prefix']} — {label}")
    return {"id": f"{spec['source']}:{hashlib.sha1(url.encode()).hexdigest()[:12]}",
            "source": spec["source"], "cite": cite, "title": label, "pdf_url": url,
            "number": number, "year": year, "dtype": spec["source"].upper()}


def crawl_items(spec, found):
    for url, text in found:
        meta = crawl_doc_meta(spec, url, text)
        if not meta:
            continue
        try:
            body, ctype = cached_fetch(url)
        except Exception as e:
            log(f"   ! {url}: {e}")
            continue
        k = kind_of(url, ctype, body)
        if k == "pdf":
            pieces = [(i + 1, t) for i, t in enumerate(pdf_pages(body)) if t.strip()]
        else:
            pieces = [(None, "\n".join(to_lines(url, ctype, body)))]
        yield meta, pieces


def discover_laws(man, crawled: dict) -> dict:
    """Pick a consolidated text for each law without url/file from the crawl."""
    out = {}
    pool = [(u, t) for docs in crawled.values() for u, t in docs]
    for spec in man.get("laws", []):
        if spec.get("url") or spec.get("file") or not spec.get("discover"):
            continue
        rx = re.compile(spec["discover"])
        hits = [u for u, t in pool if rx.search(urllib.parse.unquote(u) + " " + t)]
        if hits:
            out[spec["id"]] = hits[0]
    return out


def crawl_for_discovery(man):
    """Crawl the ΕΑΔΗΣΥ site once with a broad filter to find law texts."""
    spec = next((c for c in man.get("crawl", []) if c["id"] == "eadhsy-guidance"), None)
    if not spec:
        return {}
    broad = {**spec, "doc_match": "(?i)(4412|4782|ν\\.?\\s?\\d{4}|nomos|νόμ)", "exclude": "(?i)apofasi"}
    docs, _ = crawl(broad, max_pages=spec.get("max_pages", 100))
    return {"discovery": docs}


# =========================================================================
# commands
# =========================================================================
def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def cmd_probe(args):
    man = load_manifest()
    only = args.only
    log("== EUR-Lex directives ==")
    for spec in man.get("eu", []):
        if only and spec["id"] != only:
            continue
        try:
            items = eu_items(spec)
            arts = sum(1 for d, _ in items if ":art:" in d["id"])
            ann = sum(1 for d, _ in items if ":annex:" in d["id"])
            ex = next((d["cite"] + " — " + (d["title"] or "") for d, _ in items if ":art:" in d["id"]), "-")
            log(f"  OK  {spec['cite']}: {arts} άρθρα, {ann} παραρτήματα  (π.χ. {ex[:70]})")
        except Exception as e:
            log(f"  ERR {spec['cite']}: {e}")

    crawled = {}
    log("== Crawled sites ==")
    for spec in man.get("crawl", []):
        if only and spec["id"] != only:
            continue
        try:
            docs, pages = crawl(spec, max_pages=min(spec.get("max_pages", 100), args.pages),
                                log_pages=True)
            crawled[spec["id"]] = docs
            usable = [crawl_doc_meta(spec, u, t) for u, t in docs]
            usable = [m for m in usable if m]
            log(f"  {'OK ' if usable else '-- '} {spec['id']}: {pages} σελίδες, {len(docs)} έγγραφα, "
                f"{len(usable)} με παραπομπή")
            for m in usable[:6]:
                log(f"       {m['cite'][:70]}  <- {m['pdf_url'][:90]}")
        except Exception as e:
            log(f"  ERR {spec['id']}: {e}")

    log("== Greek laws ==")
    need = [s for s in man.get("laws", []) if not (s.get("url") or s.get("file"))]
    disc = {}
    if need and not only:
        crawled.update(crawl_for_discovery(man))
        disc = discover_laws(man, crawled)
    for spec in man.get("laws", []):
        if only and spec["id"] != only:
            continue
        try:
            lines, url = law_text(spec, disc)
            _, arts = split_articles(lines)
            n = sum(1 for a in arts if a["kind"] == "art")
            first = arts[0]["art"] if arts else "-"
            last = arts[-1]["art"] if arts else "-"
            log(f"  OK  {spec['cite']}: {n} άρθρα ({first}…{last}) από {url[:90]}")
        except Exception as e:
            log(f"  --  {spec['cite']}: {e}")
    log("\nΤίποτα δεν αποθηκεύτηκε (probe). Για εισαγωγή: python sources.py run")


def cmd_run(args):
    man = load_manifest()
    lock_f = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("Another ingest is running; try again later.")
        return
    con = db.connect()
    db.init(con)
    total = {"new": 0, "same": 0, "err": 0}

    def put(doc, pieces):
        try:
            r = store_doc(con, doc, pieces, force=args.force)
        except Exception as e:
            total["err"] += 1
            log(f"   ! {doc['cite']}: {e}")
            return
        total["same" if r == "unchanged" else "new"] += 1
        if r != "unchanged":
            log(f"   {doc['cite'][:80]}: {r}")

    try:
        for spec in man.get("eu", []):
            if args.only and spec["id"] != args.only:
                continue
            log(f"== {spec['cite']} ==")
            try:
                for doc, pieces in eu_items(spec):
                    put(doc, pieces)
            except Exception as e:
                log(f"  ERR {e}")

        crawled = {}
        for spec in man.get("crawl", []):
            if args.only and spec["id"] != args.only:
                continue
            log(f"== {spec['id']} (crawl) ==")
            try:
                docs, pages = crawl(spec)
                crawled[spec["id"]] = docs
                log(f"  {pages} σελίδες, {len(docs)} έγγραφα")
                for doc, pieces in crawl_items(spec, docs):
                    put(doc, pieces)
            except Exception as e:
                log(f"  ERR {e}")

        laws = [s for s in man.get("laws", []) if not args.only or s["id"] == args.only]
        disc = {}
        if any(not (s.get("url") or s.get("file")) for s in laws):
            crawled.update(crawl_for_discovery(man))
            disc = discover_laws(man, crawled)
        for spec in laws:
            log(f"== {spec['cite']} ==")
            try:
                for doc, pieces in law_items(spec, disc):
                    put(doc, pieces)
            except Exception as e:
                log(f"  --  {e}")
    finally:
        con.close()
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()
    log(f"\nΤΕΛΟΣ: {total['new']} νέα/ενημερωμένα, {total['same']} αμετάβλητα, {total['err']} σφάλματα.")
    log("Επανεκκίνησε την εφαρμογή για να τα δει: sudo systemctl restart lawbureaucracy")


def cmd_list(args):
    con = db.connect()
    db.init(con)
    rows = con.execute(
        "SELECT COALESCE(source,'eadhsy') s, COUNT(*) docs, SUM(n_chunks) ch, "
        "SUM(status='embedded') ok FROM decisions GROUP BY s ORDER BY docs DESC").fetchall()
    for r in rows:
        log(f"  {r['s']:10s} {r['docs']:7d} έγγραφα  {r['ok'] or 0:7d} ενσωματωμένα  {r['ch'] or 0:8d} τμήματα")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="check every source without storing anything")
    p.add_argument("--only")
    p.add_argument("--pages", type=int, default=40, help="max pages per crawled site")
    r = sub.add_parser("run", help="fetch, split, embed and store")
    r.add_argument("--only")
    r.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    sub.add_parser("list", help="documents per source")
    a = ap.parse_args()
    {"probe": cmd_probe, "run": cmd_run, "list": cmd_list}[a.cmd](a)
