"""Gemini client (stdlib only): query rewrite + streaming answer.

Retrieval stays on local bge-m3; Gemini is used for its Greek fluency to
(a) sharpen the search query and (b) write the grounded, streamed answer.
"""
import json
import urllib.request
import urllib.error

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_BASE


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
    return f"{GEMINI_BASE}/models/{GEMINI_MODEL}:{method}?key={GEMINI_API_KEY}"


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
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60},
    }).encode()
    req = urllib.request.Request(
        _endpoint("generateContent"), data=body,
        headers={"Content-Type": "application/json"},
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
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60},
    }).encode()
    req = urllib.request.Request(_endpoint("generateContent"), data=body,
                                 headers={"Content-Type": "application/json"})
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
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }).encode()
    url = _endpoint("streamGenerateContent") + "&alt=sse"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise GeminiError(f"Gemini HTTP {e.code}: {detail}")
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
                parts = chunk["candidates"][0]["content"]["parts"]
                for p in parts:
                    if "text" in p:
                        yield p["text"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
