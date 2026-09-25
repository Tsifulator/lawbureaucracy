"""Central config for the lawbureaucracy brain."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
WEB_DIR = BASE.parent / "web"


def _load_env():
    """Tiny .env loader (no dependency). Real env vars win over the file."""
    envf = BASE / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()  # must run BEFORE anything below reads os.environ

# --- Paths (DATA_DIR points the cloud deploy at a mounted volume, e.g. /data) ---
DATA = Path(os.environ.get("DATA_DIR") or (BASE / "data"))
PDF_DIR = DATA / "pdfs"
DB_PATH = DATA / "decisions.db"

DATA.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- EADHSY source (wpDataTables server-side API) ---
SITE = "https://eadhsy.gr"
APOFASEIS_PAGE = f"{SITE}/apofaseis/"
AJAX_URL = f"{SITE}/wp-admin/admin-ajax.php?action=get_wdtable&table_id=5"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

# --- Embeddings ---
# Backend for turning a QUERY into a vector. Both produce bge-m3 1024-dim vectors
# in the same space as the stored ones (verified: cosine 1.0000, identical top-8):
#   'ollama' -> local Ollama bge-m3            [default; what ingest.py always uses]
#   'onnx'   -> bundled bge-m3 fp16 ONNX       [cloud: no Ollama, no API key]
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

# ONNX backend assets (downloaded to DATA on first boot by bootstrap.py)
ONNX_MODEL_PATH = Path(os.environ.get("ONNX_MODEL_PATH") or (DATA / "bge-m3-fp16.onnx"))
ONNX_TOKENIZER_PATH = Path(os.environ.get("ONNX_TOKENIZER_PATH") or (DATA / "bge-m3-tokenizer.json"))
ONNX_MODEL_URL = os.environ.get(
    "ONNX_MODEL_URL", "https://huggingface.co/Xenova/bge-m3/resolve/main/onnx/model_fp16.onnx"
)
ONNX_TOKENIZER_URL = os.environ.get(
    "ONNX_TOKENIZER_URL", "https://huggingface.co/Xenova/bge-m3/resolve/main/tokenizer.json"
)

# --- Chunking ---
CHUNK_CHARS = 1400      # ~ a few paragraphs of legal text
CHUNK_OVERLAP = 200
MIN_TEXT_CHARS = 200    # below this a PDF is treated as scanned/empty (needs OCR later)

# --- Answer engine (retrieval always stays on bge-m3) ---
# 'ollama'        = local Llama-Krikri (Greek-native, free, no key)
# 'openai_compat' = any OpenAI-style API (Groq/OpenRouter/Together/DeepSeek/…)
# 'gemini'        = Google Gemini Flash
# 'none'          = search-only, no LLM
ANSWER_ENGINE = os.environ.get("ANSWER_ENGINE", "ollama")

# --- OpenAI-compatible provider (used when ANSWER_ENGINE=openai_compat) ---
# Defaults target Groq's free tier. Point these three at any other provider that
# speaks /chat/completions and nothing else has to change.
LLM_BASE = os.environ.get("LLM_BASE", "https://api.groq.com/openai/v1")
# Groq dropped Llama 3.1 8B / 3.3 70B from free + Developer accounts on
# 2026-08-16. gpt-oss-120b is the strongest Greek writer left on the free tier
# (checked against qwen3.8-27b, which slips into private-law register:
# "συμβολαίου" where procurement law says "σύμβασης").
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_LABEL = os.environ.get("LLM_LABEL", "GPT-OSS 120B")  # shown in the UI
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "krikri")  # Ollama tag for the answer LLM
REWRITE = os.environ.get("REWRITE", "0") == "1"          # sharpen query before search (off: faster)

# --- Gemini (only used when ANSWER_ENGINE=gemini) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# 'gemini-flash-latest' auto-tracks the newest Flash so it survives model churn
# (Gemini 2.5 retires 2026-10-16). Override with GEMINI_MODEL in .env if needed.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# --- Search server ---
# Cloud (Railway) injects PORT and needs 0.0.0.0; locally stay on loopback.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8787"))

# --- Prebuilt index (so the cloud deploy skips re-scraping + re-embedding 15.6k PDFs) ---
DB_URL = os.environ.get(
    "DB_URL",
    "https://github.com/Tsifulator/lawbureaucracy/releases/download/db-2026-08-03/decisions.db.gz",
)
# Fetch the index/model onto the volume at boot. Off locally — you already have them.
BOOTSTRAP = os.environ.get("BOOTSTRAP", "0") == "1"

# Chat history visibility. Default OFF = every browser gets its own history,
# which is what a shared URL needs (secure by default — a hosted deploy that
# forgets to set anything still isolates). A single-user install sets
# SHARED_HISTORY=1 in brain/.env so ALL chats stay visible, including ones
# created before per-visitor ownership existed.
SHARED_HISTORY = os.environ.get("SHARED_HISTORY", "0") == "1"

# Shared PIN for the hosted app. Empty = no gate (the default, and what you want
# on localhost). This is standalone — its own PIN, unrelated to any other app.
# /health stays open regardless, or Railway's healthcheck could never pass.
APP_PIN = os.environ.get("APP_PIN", "").strip()


# --- Multi-user accounts (username+password) ---------------------------------
# APP_USERS="vasilis:pw1,eleni:pw2,nikos:pw3". When set, every request must log
# in and each user sees only their OWN chats (owner = username). Empty = open
# (localhost). SECRET_KEY signs the session cookie; keep it stable across
# restarts (set it in brain/.env) or everyone gets logged out on each restart.
def _parse_users(raw: str) -> dict:
    users = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        u, _, p = pair.partition(":")
        u, p = u.strip(), p.strip()
        if u and p:
            users[u] = p
    return users


APP_USERS = _parse_users(os.environ.get("APP_USERS", ""))
SECRET_KEY = os.environ.get("SECRET_KEY", "").strip() or "lawbureaucracy-dev-secret"
