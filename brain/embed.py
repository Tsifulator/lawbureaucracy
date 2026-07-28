"""Embeddings via local Ollama (bge-m3, 1024-dim, strong multilingual/Greek)."""
import json
import time
import urllib.request
import urllib.error
from config import OLLAMA_URL, EMBED_MODEL


def embed(text: str, model: str = EMBED_MODEL, retries: int = 3) -> list[float]:
    body = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["embedding"]
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ollama embed failed after {retries} tries: {last}")
