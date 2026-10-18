"""Hybrid search over embedded chunks from every source.

Semantic (numpy cosine over bge-m3 vectors) + a keyword/number boost, since
legal lookups often hinge on exact terms (a decision number, an article).
Results are aggregated to the DOCUMENT level (a decision, a law article, a
guidance note) and returned with the best snippet and a citation label.

Sources live in separate retrieval GROUPS (decisions / law / guidance / court)
with their own slots, so ~660k ΕΑΔΗΣΥ decision chunks can't crowd a
400-article law out of the LLM's context.
"""
import json
import re
import unicodedata

import numpy as np

import db
import decnum
from config import EMBED_DIM

_GREEK_NUM = re.compile(r"\b(\d{1,5})\b")
# An explicit decision reference: "1617/2025", "απόφαση 1617 / 2025", and the
# prefixed forms "Α831/2025" (suspensions) / "AA55/2017" — in either the Greek
# or the Latin spelling of the prefix, which decnum.normalize() folds together.
_DEC_REF = re.compile(r"\b([A-Za-zΑ-Ωα-ω]{0,2}\d{1,5})\s*/\s*((?:19|20)\d{2})\b")
# Article references, matched on norm()-ed text (no accents, lowercase):
# "άρθρο 72", "άρθρου 72α", "αρθ. 72", "άρθρα 72 και 73"
_ART_REF = re.compile(r"\bαρθ(?:ρ[οουαων]*)?\.?\s*(\d{1,3}[α-ω]?)\b")
# A law/directive reference: "ν. 4412/2016", "4412/2016", "2014/24"
_LAW_REF = re.compile(r"\b(\d{1,4})\s*/\s*((?:19|20)\d{2})\b|\b((?:19|20)\d{2})\s*/\s*(\d{1,3})\b")

GROUP_CODES = {"decision": 0, "law": 1, "guidance": 2, "court": 3, "other": 4}
# How many distinct documents each group may contribute to the LLM context.
QUOTA = {"decision": 8, "law": 5, "guidance": 3, "court": 3, "other": 2}
# Non-decision groups are included only when they are in the same league as
# the best hit (GROUP_GAP), and within a group only documents close to that
# group's own best (WITHIN_GAP) — so a relevant article always gets in, while
# five loosely related articles don't pad the context.
GROUP_GAP = 0.30
WITHIN_GAP = 0.10
# Bonus for documents the user named explicitly ("1617/2025", "άρθρο 72"):
# larger than any natural score (cosine + nudges stays well under 2).
FORCED = 3.0


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def norm(s: str) -> str:
    return strip_accents(s).lower().replace("ς", "σ")


def tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-zα-ω]+", norm(s)) if len(t) >= 3}


def cite_of(dec: dict) -> str:
    """The citation label for a document; ΕΑΔΗΣΥ decisions keep 'N/YYYY'."""
    if dec.get("cite"):
        return dec["cite"]
    return f"{dec.get('number')}/{dec.get('year')}"


def title_of(dec: dict) -> str:
    if dec.get("source") and dec.get("source") != "eadhsy":
        t = dec.get("title")
        return f"{cite_of(dec)} — {t}" if t else cite_of(dec)
    return (f"Απόφαση ΕΑΔΗΣΥ {dec.get('number')}/{dec.get('year')}"
            + (f" — {dec.get('dtype')}" if dec.get("dtype") else ""))


class Index:
    """Low-memory search index.

    The full corpus is ~660k chunks: 2.7 GB of float32 vectors plus ~0.8 GB of
    text. Holding all of that (and the temporary copies np.stack makes) needs
    8-10 GB of RAM, which a 4 GB Raspberry Pi can't give. So instead:

      * vectors live in an on-disk cache as int8 (one scale per row),
        memory-mapped: ~680 MB, paged in by the OS, never copied into Python;
      * a query scores every chunk in blocks against that cache (approximate),
        then RE-SCORES the candidate pool with the exact float32 vectors read
        back from SQLite, so final ranking matches an exact in-RAM index;
      * chunk text is fetched from SQLite only for the handful of rows needed.

    The cache follows the chunks table: new rows are APPENDED (fast), and it is
    rebuilt from scratch only if older chunks were deleted or changed.
    """

    BLOCK = 32768          # rows scored per numpy block (~128 MB float32 temp)
    POOL_FACTOR = 4        # approximate pool = this x exact pool, then rescore

    def __init__(self):
        import config

        con = db.connect()
        db.init(con)
        self.cache_dir = config.DATA / "index_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            r["id"]: dict(r)
            for r in con.execute(
                "SELECT id, number, year, dtype, pdf_url, source, cite, title, article "
                "FROM decisions"
            ).fetchall()
        }
        n, max_id = con.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
        con.close()

        self._ensure_cache(int(n), int(max_id))
        self.Q = np.load(self.cache_dir / "vecs_i8.npy", mmap_mode="r")
        self.scales = np.load(self.cache_dir / "scales.npy")
        self.chunk_ids = np.load(self.cache_dir / "chunk_ids.npy")
        self.dec_idx = np.load(self.cache_dir / "dec_idx.npy")
        self.dec_list = json.loads((self.cache_dir / "dec_list.json").read_text(encoding="utf-8"))
        self._dec_pos = {d: i for i, d in enumerate(self.dec_list)}

        # group code per document, then per chunk row (one int8 per row)
        dec_group = np.array(
            [GROUP_CODES[db.group_of(self.meta.get(d, {}).get("source"))] for d in self.dec_list]
            or [0], dtype=np.int8)
        self.row_group = dec_group[self.dec_idx] if len(self.dec_idx) else np.zeros(0, np.int8)
        self.groups_present = sorted(set(int(g) for g in np.unique(self.row_group)))

        # ΕΑΔΗΣΥ / court decisions by "N/YYYY"
        self.dec_by_ref = {}
        # legislation by (number, year, article): ("4412","2016","72") -> doc id
        self.law_by_ref = {}
        for did, m in self.meta.items():
            g = db.group_of(m.get("source"))
            if g in ("decision", "court") and m.get("number") and m.get("year"):
                key = (decnum.normalize(m["number"]), m["year"])
                self.dec_by_ref.setdefault(key, []).append(did)
            elif g == "law" and m.get("article"):
                key = (str(m.get("number")), str(m.get("year")), norm(str(m["article"])))
                self.law_by_ref[key] = did

    # ---------------------------------------------------------------- cache
    def _ensure_cache(self, n, max_id):
        stamp_f = self.cache_dir / "stamp.json"
        want = {"n": n, "max_id": max_id, "dim": EMBED_DIM, "v": 1}
        old = None
        if stamp_f.exists():
            try:
                old = json.loads(stamp_f.read_text())
            except Exception:
                old = None
        if old == want:
            return
        if old and self._try_append(old, want):
            stamp_f.write_text(json.dumps(want))
            return
        self._rebuild(n, want)

    def _quantize(self, batch):
        V = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in batch])
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        amax = np.maximum(np.abs(V).max(axis=1), 1e-12)
        return np.round(V / amax[:, None] * 127).astype(np.int8), amax / 127.0

    def _try_append(self, old, want) -> bool:
        """New chunks only (ids above the old max, none of the old ones touched):
        copy the existing cache and quantize just the new rows."""
        if old.get("dim") != EMBED_DIM or old.get("v") != want["v"]:
            return False
        if want["n"] < old["n"] or want["max_id"] < old["max_id"]:
            return False
        con = db.connect()
        try:
            same = con.execute("SELECT COUNT(*) FROM chunks WHERE id <= ?",
                               (old["max_id"],)).fetchone()[0]
            if same != old["n"]:
                return False  # something old was deleted/re-ingested -> full rebuild
            new_rows = con.execute(
                "SELECT id, decision_id, vec FROM chunks WHERE id > ? ORDER BY id",
                (old["max_id"],)).fetchall()
        finally:
            con.close()
        if len(new_rows) != want["n"] - old["n"]:
            return False
        print(f"appending {len(new_rows)} new chunks to the index cache…", flush=True)
        tmp = self.cache_dir / "building"
        tmp.mkdir(exist_ok=True)
        n_old, n = old["n"], want["n"]
        oldQ = np.load(self.cache_dir / "vecs_i8.npy", mmap_mode="r")
        Q = np.lib.format.open_memmap(tmp / "vecs_i8.npy", mode="w+", dtype=np.int8, shape=(n, EMBED_DIM))
        for s in range(0, n_old, self.BLOCK):
            e = min(s + self.BLOCK, n_old)
            Q[s:e] = oldQ[s:e]
        del oldQ
        scales = np.concatenate([np.load(self.cache_dir / "scales.npy"), np.zeros(n - n_old, np.float32)])
        chunk_ids = np.concatenate([np.load(self.cache_dir / "chunk_ids.npy"), np.zeros(n - n_old, np.int64)])
        dec_idx = np.concatenate([np.load(self.cache_dir / "dec_idx.npy"), np.zeros(n - n_old, np.int32)])
        dec_list = json.loads((self.cache_dir / "dec_list.json").read_text(encoding="utf-8"))
        dec_pos = {d: i for i, d in enumerate(dec_list)}
        row = n_old
        for s in range(0, len(new_rows), 20000):
            batch = new_rows[s:s + 20000]
            q8, sc = self._quantize(batch)
            Q[row:row + len(batch)] = q8
            scales[row:row + len(batch)] = sc
            for j, r in enumerate(batch):
                chunk_ids[row + j] = r["id"]
                d = r["decision_id"]
                if d not in dec_pos:
                    dec_pos[d] = len(dec_list)
                    dec_list.append(d)
                dec_idx[row + j] = dec_pos[d]
            row += len(batch)
        Q.flush()
        del Q
        self._commit_cache(tmp, scales, chunk_ids, dec_idx, dec_list)
        print("index cache ready", flush=True)
        return True

    def _rebuild(self, n, want):
        print(f"building low-memory index cache for {n} chunks (one-off)…", flush=True)
        tmp = self.cache_dir / "building"
        tmp.mkdir(exist_ok=True)
        Q = np.lib.format.open_memmap(tmp / "vecs_i8.npy", mode="w+", dtype=np.int8, shape=(n, EMBED_DIM))
        scales = np.zeros(n, dtype=np.float32)
        chunk_ids = np.zeros(n, dtype=np.int64)
        dec_idx = np.zeros(n, dtype=np.int32)
        dec_list, dec_pos = [], {}

        con = db.connect()
        cur = con.execute("SELECT id, decision_id, vec FROM chunks ORDER BY id")
        row = 0
        while True:
            batch = cur.fetchmany(20000)
            if not batch:
                break
            k = len(batch)
            Q[row:row + k], scales[row:row + k] = self._quantize(batch)
            for j, r in enumerate(batch):
                chunk_ids[row + j] = r["id"]
                d = r["decision_id"]
                if d not in dec_pos:
                    dec_pos[d] = len(dec_list)
                    dec_list.append(d)
                dec_idx[row + j] = dec_pos[d]
            row += k
            print(f"  … {row}/{n}", flush=True)
        con.close()
        Q.flush()
        del Q
        self._commit_cache(tmp, scales[:row], chunk_ids[:row], dec_idx[:row], dec_list)
        (self.cache_dir / "stamp.json").write_text(json.dumps(want))
        print("index cache ready", flush=True)

    def _commit_cache(self, tmp, scales, chunk_ids, dec_idx, dec_list):
        np.save(tmp / "scales.npy", scales)
        np.save(tmp / "chunk_ids.npy", chunk_ids)
        np.save(tmp / "dec_idx.npy", dec_idx)
        (tmp / "dec_list.json").write_text(json.dumps(dec_list), encoding="utf-8")
        for f in tmp.iterdir():
            f.replace(self.cache_dir / f.name)
        tmp.rmdir()

    # ---------------------------------------------------------------- helpers
    @property
    def n_chunks(self):
        return int(self.Q.shape[0])

    def _rows(self, idxs, with_vec=False):
        """Fetch text (and optionally exact vectors) for row indices."""
        idxs = [int(i) for i in idxs]
        if not idxs:
            return {}
        id_to_row = {int(self.chunk_ids[i]): i for i in idxs}
        cols = "id, text, vec" if with_vec else "id, text"
        con = db.connect()
        out = {}
        try:
            ids = list(id_to_row)
            for s in range(0, len(ids), 900):  # stay under SQLite's variable limit
                part = ids[s:s + 900]
                q = f"SELECT {cols} FROM chunks WHERE id IN ({','.join('?' * len(part))})"
                for r in con.execute(q, part):
                    out[id_to_row[r["id"]]] = r
        finally:
            con.close()
        return out

    def _approx_sims(self, q):
        sims = np.empty(self.n_chunks, dtype=np.float32)
        for s in range(0, self.n_chunks, self.BLOCK):
            blk = np.asarray(self.Q[s:s + self.BLOCK], dtype=np.float32)
            sims[s:s + self.BLOCK] = (blk @ q) * self.scales[s:s + self.BLOCK]
        return sims

    @staticmethod
    def _exact(row, q):
        v = np.frombuffer(row["vec"], dtype=np.float32)
        return float(v @ q / (np.linalg.norm(v) or 1.0))

    def _doc_of(self, i):
        return self.dec_list[self.dec_idx[i]]

    def _force_doc(self, did, q, scored, texts, bonus=FORCED):
        """Put a document's best chunk at the front (explicit user reference)."""
        p = self._dec_pos.get(did)
        if p is None:
            return
        idxs = np.flatnonzero(self.dec_idx == p)
        if len(idxs) == 0:
            return
        drows = self._rows(idxs, with_vec=True)
        if not drows:
            return
        best_i = max(drows, key=lambda j: self._exact(drows[j], q))
        texts[best_i] = drows[best_i]["text"]
        scored[:] = [(s, j) for s, j in scored if j != best_i]
        scored.append((self._exact(drows[best_i], q) + bonus, best_i))

    def _rank(self, query: str, pool: int = 200, side_pool: int = 40):
        """Hybrid-score a candidate pool: cosine + keyword nudge + number boost.
        Candidates are drawn per retrieval group (decisions get `pool`, every
        other group `side_pool`) so small sources always get a look-in.
        Returns ([(score, row_index)] best-first, {row_index: text})."""
        from embed import embed

        if self.n_chunks == 0:
            return [], {}

        q = np.asarray(embed(query), dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        approx = self._approx_sims(q)

        cand = []
        for g in self.groups_present:
            gp = pool if g == GROUP_CODES["decision"] else side_pool
            if len(self.groups_present) == 1:
                rows_g = None  # everything is one group: skip the mask
                size = self.n_chunks
            else:
                rows_g = np.flatnonzero(self.row_group == g)
                size = len(rows_g)
            wide = min(size, gp * self.POOL_FACTOR)
            if wide <= 0:
                continue
            src = approx if rows_g is None else approx[rows_g]
            top = np.argpartition(-src, wide - 1)[:wide]
            idx = top if rows_g is None else rows_g[top]
            # exact rescoring happens below; keep the best `gp` per group
            cand.append((gp, idx))

        all_idx = np.unique(np.concatenate([c for _, c in cand]))
        rows = self._rows(all_idx, with_vec=True)
        exact = {i: self._exact(r, q) for i, r in rows.items()}
        pool_idx = []
        for gp, idx in cand:
            ranked = sorted((int(i) for i in idx if int(i) in exact), key=lambda i: -exact[i])
            pool_idx.extend(ranked[:gp])
        pool_idx = list(dict.fromkeys(pool_idx))

        qn = norm(query)
        qtokens = tokens(query)
        qnums = set(_GREEK_NUM.findall(query))
        texts, scored = {}, []
        for i in pool_idx:
            score = exact[i]
            texts[i] = rows[i]["text"]
            if qtokens:
                nt = norm(texts[i])
                score += 0.03 * sum(1 for t in qtokens if t in nt)  # small lexical nudge
            dec = self.meta.get(self._doc_of(i), {})
            if (qnums and db.group_of(dec.get("source")) in ("decision", "court")
                    and decnum.digits(dec.get("number")) in qnums):
                score += 0.25  # strong boost: exact decision-number match
            scored.append((score, i))

        # "1617/2025" -> that decision; "άρθρο 72 ν. 4412/2016" -> that article.
        laws = {(a, b) for a, b, _, _ in _LAW_REF.findall(qn) if a} | \
               {(b, a) for _, _, a, b in _LAW_REF.findall(qn) if a}
        arts = _ART_REF.findall(qn)
        if arts:
            for art in arts:
                hits = [did for (num, yr, a), did in self.law_by_ref.items()
                        if a == art and (not laws or (num, yr) in laws)]
                if len(hits) <= 3:  # an article number alone is ambiguous across laws
                    for did in hits:
                        self._force_doc(did, q, scored, texts, bonus=FORCED + 0.2)
        for num, year in _DEC_REF.findall(query):
            for did in self.dec_by_ref.get((decnum.normalize(num), year), []):
                self._force_doc(did, q, scored, texts)

        scored.sort(key=lambda x: -x[0])
        return scored, texts

    def _pick(self, scored, k, quotas=None):
        """Collapse chunks to DISTINCT documents (each one's best chunk), filling
        each group up to its quota. `k` caps the decision group."""
        quotas = dict(quotas or QUOTA)
        quotas["decision"] = k
        # reference points: the best *semantic* hit overall and per group
        # (explicit references carry a +FORCED bonus and are always kept)
        natural = [s for s, _ in scored if s < FORCED]
        best_score = max(natural) if natural else 0.0
        group_best = {}
        for score, i in scored:
            if score < FORCED:
                g = db.group_of(self.meta.get(self._doc_of(i), {}).get("source"))
                group_best.setdefault(g, score)  # scored is sorted best-first
        taken, counts = {}, {}
        for score, i in scored:
            did = self._doc_of(i)
            if did in taken:
                continue
            g = db.group_of(self.meta.get(did, {}).get("source"))
            if counts.get(g, 0) >= quotas.get(g, 0):
                continue
            if g != "decision" and score < FORCED:
                gb = group_best.get(g, 0.0)
                if gb < best_score - GROUP_GAP or score < gb - WITHIN_GAP:
                    continue
            taken[did] = (score, i)
            counts[g] = counts.get(g, 0) + 1
        return sorted(taken.items(), key=lambda x: -x[1][0])

    def _record(self, did, score, text):
        dec = self.meta.get(did, {})
        return {
            "doc_id": did,
            "source": dec.get("source") or "eadhsy",
            "group": db.group_of(dec.get("source")),
            "cite": cite_of(dec),
            "number": dec.get("number"),
            "year": dec.get("year"),
            "type": dec.get("dtype"),
            "pdf_url": dec.get("pdf_url"),
            "score": float(score),
            "title": title_of(dec),
            "text": text,
        }

    def search(self, query: str, k: int = 8, pool: int = 60):
        scored, texts = self._rank(query, pool=max(pool, k * 25))
        results = []
        for did, (score, i) in self._pick(scored, k):
            r = self._record(did, round(score, 4), None)
            snippet = texts[i]
            if len(snippet) > 320:
                snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
            r.pop("text")
            r["snippet"] = snippet
            results.append(r)
        return results

    def top_chunks(self, query: str, k: int = 10):
        """Full-text context for the LLM: the best chunk from each of the
        DISTINCT documents picked per group (no truncation)."""
        scored, texts = self._rank(query, pool=max(200, k * 25))
        return [self._record(did, score, texts[i]) for did, (score, i) in self._pick(scored, k)]


if __name__ == "__main__":
    import sys

    idx = Index()
    print(f"index: {idx.n_chunks} chunks over {len(idx.meta)} documents")
    q = " ".join(sys.argv[1:]) or "διαγωνισμός καθαριότητας νοσοκομείου"
    for r in idx.search(q):
        print(f"\n[{r['score']}] {r['title']}\n  {r['pdf_url']}\n  {r['snippet']}")
