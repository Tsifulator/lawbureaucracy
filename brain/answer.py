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
    "Είσαι νομικός βοηθός για τις αποφάσεις προδικαστικών προσφυγών της ΕΑΔΗΣΥ "
    "(δημόσιες συμβάσεις). Απάντησε στα ελληνικά, με σαφήνεια και ακρίβεια. "
    "Χρησιμοποίησε ΜΟΝΟ τα παρακάτω αποσπάσματα αποφάσεων. Παρέθεσε τις αποφάσεις "
    "που επικαλείσαι στη μορφή [αριθμός/έτος]. Αν τα αποσπάσματα δεν αρκούν για "
    "να απαντηθεί η ερώτηση, πες το ρητά αντί να μαντέψεις."
)


def build_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    blocks = []
    for c in chunks:
        tag = f"{c.get('number')}/{c.get('year')}"
        text = (c.get("text", "") or "")[:1200]  # cap per-chunk for faster first token
        blocks.append(f"[{tag}] ({c.get('type','')})\n{text}")
    context = "\n\n---\n\n".join(blocks) if blocks else "(καμία σχετική απόφαση)"

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
        f"=== ΑΠΟΣΠΑΣΜΑΤΑ ΑΠΟΦΑΣΕΩΝ ===\n{context}\n\n"
        f"=== ΕΡΩΤΗΣΗ ===\n{question}\n\n=== ΑΠΑΝΤΗΣΗ ==="
    )


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for c in chunks:
        key = (c.get("number"), c.get("year"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"number": c["number"], "year": c["year"], "type": c["type"],
                        "pdf_url": c["pdf_url"], "title": c["title"]})
    return sources


def answer_stream(index, question: str, k: int = 10, history: list[dict] | None = None):
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
