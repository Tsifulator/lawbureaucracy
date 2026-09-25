"""Hybrid search over embedded decision chunks.

Semantic (numpy cosine over bge-m3 vectors) + a keyword/number boost, since
legal lookups often hinge on exact terms (a decision number, a specific word).
Results are aggregated to the DECISION level and returned with the best snippet.
"""
import re
import unicodedata

import numpy as np

import db
import decnum

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
    """In-memory search index; built once, reused per query."""

    def __init__(self):
        self.con = db.connect()
        db.init(self.con)
        self.ids, self.dec_ids, self.texts, self.M = db.load_matrix(self.con)
        self.norm_texts = [norm(t) for t in self.texts]
        # decision metadata lookup
        self.meta = {
            r["id"]: dict(r)
            for r in self.con.execute(
                "SELECT id, number, year, dtype, pdf_url FROM decisions"
            ).fetchall()
        }
        # Reverse lookups so an explicit "1617/2025" can be fetched by metadata
        # instead of hoping it lands in the semantic pool (it often doesn't —
        # a decision's own number is barely present in its text).
        self.chunks_by_dec = {}
        for i, did in enumerate(self.dec_ids):
            self.chunks_by_dec.setdefault(did, []).append(i)
        self.dec_by_ref = {}
        for did, m in self.meta.items():
            if m.get("number") and m.get("year"):
                key = (decnum.normalize(m["number"]), m["year"])
                self.dec_by_ref.setdefault(key, []).append(did)

    @property
    def n_chunks(self):
        return len(self.ids)

    def _rank(self, query: str, pool: int = 200):
        """Hybrid-score a candidate pool: cosine + keyword nudge + number boost.

        Shared by search() AND top_chunks(), so the chat box ranks exactly like
        the search endpoint — before this, asking "1617/2025" in chat silently
        missed the decision-number boost that /search applies.
        Returns [(score, chunk_index)] sorted best-first.
        """
        from embed import embed

        if self.M.shape[0] == 0:
            return []

        q = np.asarray(embed(query), dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        sims = self.M @ q  # cosine (M already normalized)

        qtokens = tokens(query)
        qnums = set(_GREEK_NUM.findall(query))

        # take a candidate pool by semantic score, then re-rank with keyword boost
        pool_idx = np.argsort(-sims)[:pool]
        scored = []
        for i in pool_idx:
            i = int(i)
            score = float(sims[i])
            nt = self.norm_texts[i]
            if qtokens:
                hits = sum(1 for t in qtokens if t in nt)
                score += 0.03 * hits  # small lexical nudge
            dec = self.meta.get(self.dec_ids[i], {})
            if qnums and decnum.digits(dec.get("number")) in qnums:
                score += 0.25  # strong boost: exact decision-number match
            scored.append((score, i))

        # An explicit "1617/2025" is an exact request, not a fuzzy one. Look the
        # decision up by metadata and force it to the front — relying on the
        # semantic pool fails, because a decision rarely quotes its own number
        # (1617/2025's best chunk ranks ~370th by cosine, well outside the pool).
        for num, year in _DEC_REF.findall(query):
            for did in self.dec_by_ref.get((decnum.normalize(num), year), []):
                idxs = self.chunks_by_dec.get(did, [])
                if not idxs:
                    continue
                best_i = max(idxs, key=lambda j: float(sims[j]))
                scored = [(s, j) for s, j in scored if j != best_i]
                scored.append((float(sims[best_i]) + 1.0, best_i))

        scored.sort(key=lambda x: -x[0])
        return scored

    def _best_per_decision(self, scored, k):
        """Collapse chunks to DISTINCT decisions, keeping each one's best chunk.

        Counting decisions instead of chunks is what guarantees k different
        cases — several chunks of one long decision used to crowd everything
        else out, which is why /ask surfaced only 4-5 cases.
        """
        best = {}
        for score, i in scored:
            did = self.dec_ids[i]
            if did not in best:
                best[did] = (score, i)
            if len(best) >= k:
                break
        return sorted(best.items(), key=lambda x: -x[1][0])[:k]

    def search(self, query: str, k: int = 8, pool: int = 60):
        # pool scales with k so asking for more results actually searches wider
        scored = self._rank(query, pool=max(pool, k * 25))
        if not scored:
            return []

        results = []
        for did, (score, i) in self._best_per_decision(scored, k):
            dec = self.meta.get(did, {})
            snippet = self.texts[i]
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
        decisions (no truncation). Uses the same hybrid ranking as search(), so
        chat and search agree on what's relevant."""
        scored = self._rank(query, pool=max(200, k * 25))
        if not scored:
            return []
        out = []
        for did, (score, i) in self._best_per_decision(scored, k):
            dec = self.meta.get(did, {})
            out.append({
                "text": self.texts[i],
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
