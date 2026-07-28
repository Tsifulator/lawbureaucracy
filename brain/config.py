"""Central config for the lawbureaucracy brain."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
PDF_DIR = DATA / "pdfs"
DB_PATH = DATA / "decisions.db"

DATA.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)


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


_load_env()

# --- EADHSY source (wpDataTables server-side API) ---
SITE = "https://eadhsy.gr"
APOFASEIS_PAGE = f"{SITE}/apofaseis/"
AJAX_URL = f"{SITE}/wp-admin/admin-ajax.php?action=get_wdtable&table_id=5"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

# --- Embeddings (local Ollama) ---
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

# --- Chunking ---
CHUNK_CHARS = 1400      # ~ a few paragraphs of legal text
CHUNK_OVERLAP = 200
MIN_TEXT_CHARS = 200    # below this a PDF is treated as scanned/empty (needs OCR later)

# --- Answer engine (retrieval always stays local on bge-m3) ---
# 'ollama' = local Llama-Krikri (Greek-native, free, no key); 'gemini' = cloud.
ANSWER_ENGINE = os.environ.get("ANSWER_ENGINE", "ollama")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "krikri")  # Ollama tag for the answer LLM
REWRITE = os.environ.get("REWRITE", "0") == "1"          # sharpen query before search (off: faster)

# --- Gemini (only used when ANSWER_ENGINE=gemini) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# 'gemini-flash-latest' auto-tracks the newest Flash so it survives model churn
# (Gemini 2.5 retires 2026-10-16). Override with GEMINI_MODEL in .env if needed.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# --- Search server ---
HOST = "127.0.0.1"
PORT = 8787
