"""Answer engine for any OpenAI-compatible /chat/completions API.

Deliberately provider-agnostic: Groq, OpenRouter, Together, DeepSeek, Cerebras,
Fireworks and friends all speak this dialect, so switching provider is three env
vars (LLM_BASE / LLM_MODEL / LLM_API_KEY) rather than a new module. Defaults
point at Groq's free tier.

Same interface as ollama_gen.py / gemini.py / none_gen.py, stdlib only.
"""
import json
import urllib.error
import urllib.request

from config import LLM_API_KEY, LLM_BASE, LLM_MODEL, LLM_LABEL


class LLMError(RuntimeError):
    pass


NAME = LLM_LABEL

# Groq (and several other providers) sit behind Cloudflare, which 403s the
# default "Python-urllib/3.x" signature with error 1010. Any honest User-Agent
# gets through — this is not evasion, just identifying the client properly.
UA = "lawbureaucracy/1.0 (+https://github.com/Tsifulator/lawbureaucracy)"


def available() -> bool:
    return bool(LLM_API_KEY)


def unavailable_msg() -> str:
    return ("🔎 **Λειτουργία αναζήτησης.** Οι αυτόματες απαντήσεις AI δεν είναι "
            "ρυθμισμένες ακόμη (λείπει το LLM_API_KEY) — παρακάτω θα βρεις τις "
            "πιο σχετικές αποφάσεις της ΕΑΔΗΣΥ με σύνδεσμο στο PDF.")


def _request(payload: dict, timeout: int):
    req = urllib.request.Request(
        f"{LLM_BASE.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
            "User-Agent": UA,
        },
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"{NAME} HTTP {e.code}: {detail}")


def _complete(prompt: str, max_tokens: int, timeout: int) -> str:
    """One-shot, non-streaming. Used for the short query-rewrite helpers."""
    if not LLM_API_KEY:
        return ""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    with _request(payload, timeout) as r:
        j = json.loads(r.read())
    return (j["choices"][0]["message"]["content"] or "").strip()


def rewrite_query(q: str, timeout: int = 15) -> str:
    prompt = (
        "Μετάτρεψε την ερώτηση σε μια σύντομη φράση αναζήτησης με τους βασικούς "
        "νομικούς όρους (δημόσιες συμβάσεις / ΕΑΔΗΣΥ). Δώσε ΜΟΝΟ τη φράση, χωρίς "
        "εισαγωγικά ή επεξήγηση.\nΕρώτηση: " + q
    )
    try:
        out = _complete(prompt, max_tokens=60, timeout=timeout)
        return out.splitlines()[0].strip() if out else q
    except Exception:
        return q  # never let rewrite break the flow


def contextualize(question: str, history: list[dict], timeout: int = 25) -> str:
    """Rewrite a follow-up into a standalone search query, so retrieval can
    resolve pronouns ("το ύψος της" -> "... της εγγυητικής επιστολής")."""
    if not history or not LLM_API_KEY:
        return question
    convo = "\n".join(
        (("Χρήστης: " if m.get("role") == "user" else "Βοηθός: ")
         + (m.get("content") or "")[:300])
        for m in history[-4:]
    )
    prompt = (
        "Με βάση τη συνομιλία, ξαναγράψε την τελευταία ερώτηση ως ΑΥΤΟΤΕΛΗ φράση "
        "αναζήτησης (μία πρόταση, χωρίς εισαγωγικά ή επεξήγηση), κατανοητή χωρίς "
        f"τη συνομιλία.\n\nΣυνομιλία:\n{convo}\n\nΤελευταία ερώτηση: {question}\n\n"
        "Αυτοτελής φράση:"
    )
    try:
        return _complete(prompt, max_tokens=60, timeout=timeout) or question
    except Exception:
        return question


def stream_answer(prompt: str, timeout: int = 300):
    """Yield answer text chunks from the provider's SSE stream."""
    if not LLM_API_KEY:
        raise LLMError("LLM_API_KEY not set")
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048,
        "stream": True,
    }
    with _request(payload, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if not payload_str or payload_str == "[DONE]":
                continue
            try:
                delta = json.loads(payload_str)["choices"][0]["delta"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            tok = delta.get("content")
            if tok:
                yield tok
