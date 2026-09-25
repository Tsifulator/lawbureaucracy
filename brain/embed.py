"""Query/document embeddings — bge-m3, 1024-dim (strong multilingual/Greek).

Two interchangeable backends, both producing vectors in the SAME space:

  'ollama' (default) — local Ollama. What ingest.py always uses to build the index.
  'onnx'             — bundled bge-m3 fp16 ONNX, for hosts with no Ollama (the
                       cloud deploy). CLS pooling + L2 norm, same as bge-m3 dense.

Verified equivalent on the real 62k-chunk corpus: mean query-vector cosine 1.0000
and 100% top-8 overlap vs Ollama across 8 Greek legal queries. (int8 quantization
was measurably worse — 0.986 / 89% — so this deliberately uses fp16.)
"""
import json
import time
import urllib.request
import urllib.error

from config import OLLAMA_URL, EMBED_MODEL, EMBED_BACKEND


# --------------------------------------------------------------------------
# ollama backend
# --------------------------------------------------------------------------
def _embed_ollama(text: str, model: str = EMBED_MODEL, retries: int = 3) -> list[float]:
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


# --------------------------------------------------------------------------
# onnx backend (lazy — heavy imports only when actually selected)
# --------------------------------------------------------------------------
_SESSION = None
_TOKENIZER = None
_INPUT_NAMES = None


def _onnx_load():
    global _SESSION, _TOKENIZER, _INPUT_NAMES
    if _SESSION is not None:
        return
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from config import ONNX_MODEL_PATH, ONNX_TOKENIZER_PATH

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2  # small boxes: don't oversubscribe
    _SESSION = ort.InferenceSession(
        str(ONNX_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    _INPUT_NAMES = {i.name for i in _SESSION.get_inputs()}
    _TOKENIZER = Tokenizer.from_file(str(ONNX_TOKENIZER_PATH))
    _TOKENIZER.enable_truncation(max_length=512)


def _embed_onnx(text: str) -> list[float]:
    import numpy as np

    _onnx_load()
    enc = _TOKENIZER.encode(text)
    feed = {
        "input_ids": np.asarray([enc.ids], dtype=np.int64),
        "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64),
    }
    feed = {k: v for k, v in feed.items() if k in _INPUT_NAMES}
    out = _SESSION.run(None, feed)[0]
    vec = out[0, 0] if out.ndim == 3 else out[0]  # CLS token = bge-m3 dense
    vec = np.asarray(vec, dtype=np.float32)
    vec /= np.linalg.norm(vec) or 1.0
    return vec.tolist()


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed many texts in one call. ~8x faster than looping embed() on Ollama,
    which matters when ingesting 15k PDFs (~12 chunks each).

    Ollama's newer /api/embed takes an array; verified identical to the legacy
    single-shot /api/embeddings (cosine 1.000000 on Greek legal text), so
    batched and previously-stored vectors share one space and can be mixed.
    ONNX has no batched path here and falls back to a loop — the cloud only
    ever embeds one query at a time, so it gains nothing from batching.
    """
    if not texts:
        return []
    if EMBED_BACKEND == "onnx":
        return [_embed_onnx(t) for t in texts]

    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read()).get("embeddings")
        if out and len(out) == len(texts):
            return out
    except (urllib.error.URLError, TimeoutError, ValueError):
        pass
    # Older Ollama, or a partial reply -> one-at-a-time still works.
    return [_embed_ollama(t, model=model) for t in texts]


def warmup():
    """Pay the ONNX load cost at boot instead of on the first user's query."""
    if EMBED_BACKEND == "onnx":
        _embed_onnx("προθέρμανση")


def embed(text: str, model: str = EMBED_MODEL, retries: int = 3) -> list[float]:
    if EMBED_BACKEND == "onnx":
        return _embed_onnx(text)
    return _embed_ollama(text, model=model, retries=retries)
