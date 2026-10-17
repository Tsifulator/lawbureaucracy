"""Hybrid search over embedded decision chunks.

Semantic (numpy cosine over bge-m3 vectors) + a keyword/number boost, since
legal lookups often hinge on exact terms (a decision number, a specific word).
Results are aggregated to the DECISION level and returned with the best snippet.
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


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def norm(s: str) -> str:
    return strip_accents(s).lower().replace("ς", "σ")


def tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-zα-ω]+", norm(s)) if len(t) >= 3}


class Index:
    """Low-memory search index.

    The full corpus is ~660k chunks: 2.7 GB of float32 vectors plus ~0.8 GB of
    text. Holding all of that (and the temporary copies np.stack makes) needs
    8-10 GB of RAM, which a 4 GB Raspberry Pi can't give. So instead:

      * vectors live in a one-off on-disk cache as int8 (one scale per row),
        memory-mapped: ~680 MB, paged in by the OS, never copied into Python;
      * a query scores every chunk in blocks against that cache (approximate),
        then RE-SCORES the candidate pool with the exact float32 vectors read
        back from SQLite, so final ranking matches the old in-RAM index;
      * chunk text is fetched from SQLite only for the handful of rows needed.

    The cache is rebuilt automatically whenever the chunks table changes
    (row count or max rowid), e.g. after update.py ingests new decisions.
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
                "SELECT id, number, year, dtype, pdf_url FROM decisions"
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

        self.dec_by_ref = {}
        for did, m in self.meta.items():
            if m.get("number") and m.get("year"):
                key = (decnum.normalize(m["number"]), m["year"])
                self.dec_by_ref.setdefault(key, []).append(did)
        self._dec_pos = {d: i for i, d in enumerate(self.dec_list)}

    # ---------------------------------------------------------------- cache
    def _ensure_cache(self, n, max_id):
        stamp_f = self.cache_dir / "stamp.json"
        want = {"n": n, "max_id": max_id, "dim": EMBED_DIM, "v": 1}
        if stamp_f.exists():
            try:
                if json.loads(stamp_f.read_text()) == want:
                    return
            except Exception:
                pass
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
            V = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in batch])
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            amax = np.maximum(np.abs(V).max(axis=1), 1e-12)
            k = len(batch)
            Q[row:row + k] = np.round(V / amax[:, None] * 127).astype(np.int8)
            scales[row:row + k] = amax / 127.0
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
        np.save(tmp / "scales.npy", scales[:row])
        np.save(tmp / "chunk_ids.npy", chunk_ids[:row])
        np.save(tmp / "dec_idx.npy", dec_idx[:row])
        (tmp / "dec_list.json").write_text(json.dumps(dec_list), encoding="utf-8")
        for f in tmp.iterdir():
            f.replace(self.cache_dir / f.name)
        tmp.rmdir()
        stamp_f.write_text(json.dumps(want))
        print("index cache ready", flush=True)

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

    def _rank(self, query: str, pool: int = 200):
        """Hybrid-score a candidate pool: cosine + keyword nudge + number boost.
        Returns [(score, row_index)] sorted best-first; self._texts holds the
        text of every returned row for the caller."""
        from embed import embed

        if self.n_chunks == 0:
            return [], {}

        q = np.asarray(embed(query), dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        approx = self._approx_sims(q)

        wide = min(self.n_chunks, pool * self.POOL_FACTOR)
        cand = np.argpartition(-approx, wide - 1)[:wide]
        rows = self._rows(cand, with_vec=True)
        exact = {i: self._exact(r, q) for i, r in rows.items()}
        pool_idx = sorted(exact, key=lambda i: -exact[i])[:pool]

        qtokens = tokens(query)
        qnums = set(_GREEK_NUM.findall(query))
        texts = {}
        scored = []
        for i in pool_idx:
            score = exact[i]
            texts[i] = rows[i]["text"]
            if qtokens:
                nt = norm(texts[i])
                score += 0.03 * sum(1 for t in qtokens if t in nt)  # small lexical nudge
            dec = self.meta.get(self.dec_list[self.dec_idx[i]], {})
            if qnums and decnum.digits(dec.get("number")) in qnums:
                score += 0.25  # strong boost: exact decision-number match
            scored.append((score, i))

        # An explicit "1617/2025" is an exact request: look it up by metadata and
        # force its best chunk to the front (it rarely ranks in the semantic pool).
        for num, year in _DEC_REF.findall(query):
            for did in self.dec_by_ref.get((decnum.normalize(num), year), []):
                p = self._dec_pos.get(did)
                if p is None:
                    continue
                idxs = np.flatnonzero(self.dec_idx == p)
                if len(idxs) == 0:
                    continue
                drows = self._rows(idxs, with_vec=True)
                best_i = max(drows, key=lambda j: self._exact(drows[j], q))
                texts[best_i] = drows[best_i]["text"]
                scored = [(s, j) for s, j in scored if j != best_i]
                scored.append((self._exact(drows[best_i], q) + 1.0, best_i))

        scored.sort(key=lambda x: -x[0])
        return scored, texts

    def _best_per_decision(self, scored, k):
        """Collapse chunks to DISTINCT decisions, keeping each one's best chunk."""
        best = {}
        for score, i in scored:
            did = self.dec_list[self.dec_idx[i]]
            if did not in best:
                best[did] = (score, i)
            if len(best) >= k:
                break
        return sorted(best.items(), key=lambda x: -x[1][0])[:k]

    def search(self, query: str, k: int = 8, pool: int = 60):
        scored, texts = self._rank(query, pool=max(pool, k * 25))
        if not scored:
            return []
        results = []
        for did, (score, i) in self._best_per_decision(scored, k):
            dec = self.meta.get(did, {})
            snippet = texts[i]
            if len(snippet) > 320:
                snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
            results.append(
                {
                    "number": dec.get("number"),
                    "year": dec.get("year"),
                    "type": dec.get("dtype"),
                    "pdf_url": dec.get("pdf_url"),
                    "score": round(score, 4),
                    "snippet": snippet,
                    "title": f"Απόφαση {dec.get('number')}/{dec.get('year')}"
                    + (f" — {dec.get('dtype')}" if dec.get("dtype") else ""),
                }
            )
        return results

    def top_chunks(self, query: str, k: int = 10):
        """Full-text context for the LLM: the best chunk from each of k DISTINCT
        decisions (no truncation). Same hybrid ranking as search()."""
        scored, texts = self._rank(query, pool=max(200, k * 25))
        if not scored:
            return []
        out = []
        for did, (score, i) in self._best_per_decision(scored, k):
            dec = self.meta.get(did, {})
            out.append({
                "text": texts[i],
                "score": float(score),
                "number": dec.get("number"),
                "year": dec.get("year"),
                "type": dec.get("dtype"),
                "pdf_url": dec.get("pdf_url"),
                "title": f"Απόφαση {dec.get('number')}/{dec.get('year')}"
                         + (f" — {dec.get('dtype')}" if dec.get("dtype") else ""),
            })
        return out


if __name__ == "__main__":
    import sys

    idx = Index()
    print(f"index: {idx.n_chunks} chunks over {len(idx.meta)} decisions")
    q = " ".join(sys.argv[1:]) or "διαγωνισμός καθαριότητας νοσοκομείου"
    for r in idx.search(q):
        print(f"\n[{r['score']}] {r['title']}\n  {r['pdf_url']}\n  {r['snippet']}")
