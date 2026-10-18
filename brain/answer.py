"""RAG orchestration: retrieve (bge-m3) -> grounded prompt -> stream the answer.

The answer engine is swappable via config.ANSWER_ENGINE:
  'ollama'        -> local Llama-Krikri (Greek-native, free)   [default]
  'openai_compat' -> any OpenAI-style API (Groq/OpenRouter/Together/…)
  'gemini'        -> cloud Gemini Flash
  'none'          -> search-only, no LLM
Both engines expose the same interface (available / unavailable_msg /
rewrite_query / stream_answer), so this file doesn't care which is used.

Yields event dicts for the SSE endpoint:
  {"t": "..."}        a token of the answer
  {"sources": [...]}  the decisions the answer is grounded in
  {"done": true}
  {"error": "..."}    something went wrong (shown to the user)
"""
import config

if config.ANSWER_ENGINE == "gemini":
    import gemini as engine
elif config.ANSWER_ENGINE == "openai_compat":
    import openai_compat as engine
elif config.ANSWER_ENGINE == "none":
    import none_gen as engine
else:
    import ollama_gen as engine

SYSTEM = (
    "Είσαι νομικός βοηθός για το δίκαιο των δημοσίων συμβάσεων στην Ελλάδα. "
    "Απάντησε στα ελληνικά, με σαφήνεια, ακρίβεια και επαρκές βάθος.\n"
    "Πηγές: σου δίνονται αποσπάσματα από νομοθεσία (ελληνική και ενωσιακή), "
    "κατευθυντήριες οδηγίες και αποφάσεις (ΕΑΔΗΣΥ και δικαστηρίων). Κάθε απόσπασμα "
    "ξεκινά με την ετικέτα παραπομπής του σε αγκύλες, π.χ. [ν. 4412/2016, άρθρο 72] ή [1617/2025].\n"
    "Κανόνες:\n"
    "1. ΚΑΘΕ ουσιαστικός ισχυρισμός συνοδεύεται από παραπομπή με την ΑΚΡΙΒΗ ετικέτα "
    "του αποσπάσματος σε αγκύλες. Μην επινοείς ετικέτες, αριθμούς άρθρων ή αποφάσεων.\n"
    "2. Ξεκίνα από τον κανόνα δικαίου (νόμος/οδηγία) όπου υπάρχει και μετά δείξε πώς "
    "τον εφαρμόζουν οι αποφάσεις.\n"
    "3. Αν χρησιμοποιήσεις κάτι που ΔΕΝ υπάρχει στα αποσπάσματα (γενική γνώση), "
    "σημείωσέ το ρητά με «(γενική γνώση — χωρίς πηγή στη βάση)».\n"
    "4. Αν τα αποσπάσματα δεν αρκούν, πες ακριβώς τι λείπει αντί να μαντέψεις.\n"
    "5. Οι κατευθυντήριες οδηγίες δεν είναι δεσμευτικές όπως ο νόμος· οι αποφάσεις "
    "ΕΑΔΗΣΥ κρίνουν συγκεκριμένες υποθέσεις — πες το όπου έχει σημασία."
)

_GROUP_LABEL = {"law": "νομοθεσία", "guidance": "κατευθυντήρια οδηγία",
                "court": "απόφαση δικαστηρίου", "decision": "απόφαση ΕΑΔΗΣΥ"}


def build_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    blocks = []
    # legislation first, then guidance, then case law: mirrors how a lawyer reads
    order = {"law": 0, "guidance": 1, "court": 2, "decision": 3}
    for c in sorted(chunks, key=lambda c: order.get(c.get("group"), 4)):
        tag = c.get("cite") or f"{c.get('number')}/{c.get('year')}"
        kind = _GROUP_LABEL.get(c.get("group"), "")
        extra = f" — {c['type']}" if c.get("group") == "decision" and c.get("type") else ""
        text = c.get("text", "") or ""
        if text.startswith(f"[{tag}]"):  # sources.py chunks carry their label already
            text = text.split("\n", 1)[1] if "\n" in text else ""
            if c.get("title"):
                kind += f": {c['title'].split(' — ', 1)[-1]}"
        text = text[:1200]  # cap per-chunk for faster first token
        blocks.append(f"[{tag}] ({kind}{extra})\n{text}")
    context = "\n\n---\n\n".join(blocks) if blocks else "(κανένα σχετικό απόσπασμα)"

    hist = ""
    if history:
        lines = []
        for m in history[-6:]:  # last ~3 exchanges for continuity
            who = "Χρήστης" if m.get("role") == "user" else "Βοηθός"
            lines.append(f"{who}: {(m.get('content') or '')[:500]}")
        hist = "=== ΠΡΟΗΓΟΥΜΕΝΗ ΣΥΝΟΜΙΛΙΑ ===\n" + "\n".join(lines) + "\n\n"

    return (
        f"{SYSTEM}\n\n"
        f"{hist}"
        f"=== ΑΠΟΣΠΑΣΜΑΤΑ ΠΗΓΩΝ ===\n{context}\n\n"
        f"=== ΕΡΩΤΗΣΗ ===\n{question}\n\n=== ΑΠΑΝΤΗΣΗ ==="
    )


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for c in chunks:
        key = c.get("doc_id") or (c.get("number"), c.get("year"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"number": c.get("number"), "year": c.get("year"), "type": c.get("type"),
                        "pdf_url": c.get("pdf_url"), "title": c.get("title"),
                        "cite": c.get("cite"), "group": c.get("group")})
    # same order as the prompt: law, guidance, courts, ΕΑΔΗΣΥ
    order = {"law": 0, "guidance": 1, "court": 2, "decision": 3}
    return sorted(sources, key=lambda s: order.get(s.get("group"), 4))


def answer_stream(index, question: str, k: int = 8, history: list[dict] | None = None):
    have_engine = engine.available()

    # 1) figure out what to actually search for, then retrieve chunks:
    #    - follow-up in a conversation -> rewrite to a standalone query (uses memory)
    #    - first turn -> optional query sharpening (off by default for speed)
    search_q = question
    if history and have_engine:
        try:
            search_q = engine.contextualize(question, history)
        except Exception:
            search_q = question
    elif config.REWRITE and have_engine:
        try:
            search_q = engine.rewrite_query(question)
        except Exception:
            search_q = question
    chunks = index.top_chunks(search_q, k=k)

    # 2) emit the matched decisions IMMEDIATELY so the popup shows them
    #    within ~1s while the answer is still being written
    yield {"sources": _dedupe_sources(chunks)}

    # 3) engine not ready -> explain how to switch it on (matches already shown)
    if not have_engine:
        yield {"t": engine.unavailable_msg()}
        yield {"done": True}
        return

    # 4) stream the grounded answer (with conversation memory)
    prompt = build_prompt(question, chunks, history=history)
    try:
        for tok in engine.stream_answer(prompt):
            yield {"t": tok}
    except Exception as e:
        yield {"error": f"{engine.NAME}: {e}"}
        yield {"done": True}
        return

    yield {"done": True}
