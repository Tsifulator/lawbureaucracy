"""Gemini client (stdlib only): query rewrite + streaming answer.

Retrieval stays on local bge-m3; Gemini is used for its Greek fluency to
(a) sharpen the search query and (b) write the grounded, streamed answer.
"""
import json
import urllib.request
import urllib.error

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_BASE


UA = "lawbureaucracy/1.0 (+https://github.com/Tsifulator/lawbureaucracy)"


class GeminiError(RuntimeError):
    pass


NAME = "Gemini"


def available() -> bool:
    return bool(GEMINI_API_KEY)


def unavailable_msg() -> str:
    return ("🔎 **Λειτουργία αναζήτησης.** Οι αυτόματες απαντήσεις AI δεν είναι "
            "ρυθμισμένες ακόμη (λείπει το GEMINI_API_KEY) — παρακάτω θα βρεις τις "
            "πιο σχετικές αποφάσεις της ΕΑΔΗΣΥ με σύνδεσμο στο PDF.")


def _endpoint(method: str) -> str:
    return f"{GEMINI_BASE}/models/{GEMINI_MODEL}:{method}"


def _headers() -> dict:
    # Key goes in a header (works with both AIza… and newer AQ.… keys,
    # and keeps it out of URLs/logs).
    return {"Content-Type": "application/json", "User-Agent": UA,
            "x-goog-api-key": GEMINI_API_KEY}


def rewrite_query(q: str, timeout: int = 12) -> str:
    """Turn a loose Greek question into a tight search phrase. Falls back to q."""
    if not GEMINI_API_KEY:
        return q
    prompt = (
        "Είσαι βοηθός νομικής αναζήτησης για αποφάσεις δημοσίων συμβάσεων (ΕΑΔΗΣΥ). "
        "Μετάτρεψε την ερώτηση σε μια σύντομη φράση αναζήτησης με τους βασικούς νομικούς "
        "όρους, χωρίς εισαγωγικά ή επεξήγηση. Ερώτηση: " + q
    )
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
    }).encode()
    req = urllib.request.Request(
        _endpoint("generateContent"), data=body,
        headers=_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
        text = j["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text or q
    except Exception:
        return q  # never let query-rewrite break the flow


def contextualize(question: str, history: list[dict], timeout: int = 20) -> str:
    """Rewrite a follow-up into a standalone search query using the conversation."""
    if not history or not GEMINI_API_KEY:
        return question
    convo = "\n".join(
        (("Χρήστης: " if m.get("role") == "user" else "Βοηθός: ") + (m.get("content") or "")[:300])
        for m in history[-4:]
    )
    prompt = (
        "Με βάση τη συνομιλία, ξαναγράψε την τελευταία ερώτηση ως αυτοτελή φράση "
        "αναζήτησης (μία πρόταση, χωρίς εισαγωγικά). "
        f"Συνομιλία:\n{convo}\n\nΤελευταία ερώτηση: {question}"
    )
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
    }).encode()
    req = urllib.request.Request(_endpoint("generateContent"), data=body,
                                 headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read())
        return j["candidates"][0]["content"]["parts"][0]["text"].strip() or question
    except Exception:
        return question


def stream_answer(prompt: str, timeout: int = 300):
    """Yield answer text chunks from Gemini's SSE stream."""
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY not set")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }).encode()
    url = _endpoint("streamGenerateContent") + "?alt=sse"
    req = urllib.request.Request(url, data=body, headers=_headers())
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise GeminiError(f"Gemini HTTP {e.code}: {detail}")
    finish, blocked, got_text = None, None, False
    with resp:
        for raw in resp:  # SSE lines
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                print(f"gemini: unparsable SSE line ({len(payload)} chars)", flush=True)
                continue
            fb = chunk.get("promptFeedback") or {}
            if fb.get("blockReason"):
                blocked = fb["blockReason"]
            for cand in chunk.get("candidates") or []:
                finish = cand.get("finishReason") or finish
                for p in (cand.get("content") or {}).get("parts") or []:
                    if "text" in p and not p.get("thought"):
                        got_text = True
                        yield p["text"]
    # Say so when the answer didn't end normally, instead of silently
    # showing half a sentence.
    if blocked:
        print(f"gemini: prompt blocked ({blocked})", flush=True)
        yield f"\n\n⚠️ Το Gemini αρνήθηκε να απαντήσει ({blocked})."
    elif finish and finish != "STOP":
        print(f"gemini: finishReason={finish}", flush=True)
        why = {"MAX_TOKENS": "έφτασε το όριο μήκους"}.get(finish, finish)
        yield f"\n\n⚠️ (Η απάντηση διακόπηκε: {why}.)"
    elif not got_text:
        yield "⚠️ Το Gemini δεν επέστρεψε κείμενο. Δοκίμασε ξανά."
