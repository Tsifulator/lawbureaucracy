"""Local answer engine via Ollama — Llama-Krikri (Greek-native, free, no key).

Same interface as gemini.py so answer.py can swap engines:
  available(), unavailable_msg(), rewrite_query(q), stream_answer(prompt)
"""
import json
import urllib.request
import urllib.error

from config import OLLAMA_URL, ANSWER_MODEL

NAME = "Ollama/Krikri"


def _model_present() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            tags = json.loads(r.read()).get("models", [])
        names = {m.get("name", "") for m in tags}
        # match "krikri", "krikri:latest", etc.
        return any(n == ANSWER_MODEL or n.startswith(ANSWER_MODEL + ":") for n in names)
    except Exception:
        return False


def available() -> bool:
    return _model_present()


def unavailable_msg() -> str:
    return (f"⚠️ Το μοντέλο «{ANSWER_MODEL}» δεν βρέθηκε στο Ollama — δείχνω μόνο τις "
            f"σχετικές αποφάσεις. Τρέξε: ollama pull hf.co/ilsp/"
            f"Llama-Krikri-8B-Instruct-GGUF:Q4_K_M  (και ollama cp … {ANSWER_MODEL}).")


def rewrite_query(q: str, timeout: int = 30) -> str:
    prompt = (
        "Μετάτρεψε την ερώτηση σε μια σύντομη φράση αναζήτησης με τους βασικούς "
        "νομικούς όρους (δημόσιες συμβάσεις / ΕΑΔΗΣΥ). Δώσε ΜΟΝΟ τη φράση, χωρίς "
        "εισαγωγικά ή επεξήγηση.\nΕρώτηση: " + q
    )
    body = json.dumps({
        "model": ANSWER_MODEL, "prompt": prompt, "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": 40},
    }).encode()
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read()).get("response", "").strip()
        return out.splitlines()[0].strip() if out else q
    except Exception:
        return q  # never let rewrite break the flow


def contextualize(question: str, history: list[dict], timeout: int = 40) -> str:
    """Rewrite a follow-up into a standalone search query using the conversation,
    so retrieval finds the right decisions (e.g. 'το ύψος της' -> '... της
    εγγυητικής επιστολής συμμετοχής'). Falls back to the raw question."""
    if not history:
        return question
    convo = "\n".join(
        (("Χρήστης: " if m.get("role") == "user" else "Βοηθός: ") + (m.get("content") or "")[:300])
        for m in history[-4:]
    )
    prompt = (
        "Με βάση τη συνομιλία, ξαναγράψε την τελευταία ερώτηση ως ΑΥΤΟΤΕΛΗ φράση "
        "αναζήτησης (μία πρόταση, χωρίς εισαγωγικά ή επεξήγηση), κατανοητή χωρίς τη "
        f"συνομιλία.\n\nΣυνομιλία:\n{convo}\n\nΤελευταία ερώτηση: {question}\n\nΑυτοτελής φράση:"
    )
    body = json.dumps({
        "model": ANSWER_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": 60},
    }).encode()
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read()).get("response", "").strip()
        return out.splitlines()[0].strip() if out else question
    except Exception:
        return question


def stream_answer(prompt: str, timeout: int = 300):
    """Yield answer chunks from Ollama's streaming chat API."""
    body = json.dumps({
        "model": ANSWER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": "30m",  # keep Krikri warm between questions
        "options": {"temperature": 0.2, "num_ctx": 12288},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:  # newline-delimited JSON
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tok = (obj.get("message") or {}).get("content", "")
            if tok:
                yield tok
            if obj.get("done"):
                break
