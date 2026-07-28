"""Hybrid search over embedded decision chunks.

Semantic (numpy cosine over bge-m3 vectors) + a keyword/number boost, since
legal lookups often hinge on exact terms (a decision number, a specific word).
Results are aggregated to the DECISION level and returned with the best snippet.
"""
import re
import unicodedata

import numpy as np

import db

_GREEK_NUM = re.compile(r"\b(\d{1,5})\b")


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

    @property
    def n_chunks(self):
        return len(self.ids)

    def search(self, query: str, k: int = 8, pool: int = 60):
        from embed import embed

        if self.M.shape[0] == 0:
            return []

        q = np.asarray(embed(query), dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        sims = self.M @ q  # cosine (M already normalized)

        qtokens = tokens(query)
        qnums = set(_GREEK_NUM.findall(query))

        # take a candidate pool by semantic score, then re-rank with keyword boost
        pool_idx = np.argsort(-sims)[: max(pool, k)]
        scored = []
        for i in pool_idx:
            i = int(i)
            score = float(sims[i])
            nt = self.norm_texts[i]
            if qtokens:
                hits = sum(1 for t in qtokens if t in nt)
                score += 0.03 * hits  # small lexical nudge
            dec = self.meta.get(self.dec_ids[i], {})
            if qnums and dec.get("number") in qnums:
                score += 0.25  # strong boost: exact decision-number match
            scored.append((score, i))

        # aggregate to best chunk per decision
        best = {}
        for score, i in sorted(scored, key=lambda x: -x[0]):
            did = self.dec_ids[i]
            if did not in best:
                best[did] = (score, i)

        results = []
        for did, (score, i) in sorted(best.items(), key=lambda x: -x[1][0])[:k]:
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

    def top_chunks(self, query: str, k: int = 6):
        """Full-text top-k chunks for feeding an LLM (no dedupe, no truncation)."""
        from embed import embed

        if self.M.shape[0] == 0:
            return []
        q = np.asarray(embed(query), dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        sims = self.M @ q
        out = []
        for i in np.argsort(-sims)[:k]:
            i = int(i)
            dec = self.meta.get(self.dec_ids[i], {})
            out.append({
                "text": self.texts[i],
                "score": float(sims[i]),
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
