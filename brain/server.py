"""API + web UI: search, streaming chat with memory, and server-side chat
storage (so history survives refresh and moves with the brain).

Also serves web/ itself, so the whole thing is one origin — locally that means
no second static server, and hosted it means one Railway service.

Stdlib only (no FastAPI/pydantic — dodges the Rust wheel problem on py3.14).
"""
import hashlib
import hmac
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import chatdb
from config import HOST, PORT, WEB_DIR
from search import Index

INDEX = None

# Routes reachable without the PIN. /health must stay open or Railway's
# healthcheck can never pass and every deploy fails.
OPEN_PATHS = {"/health", "/login"}

LOGIN_HTML = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Νομικός βοηθός ΕΑΔΗΣΥ</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;height:100vh;display:grid;place-items:center;background:#111315;
       color:#e9edf1;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  form{background:#191c1f;padding:34px 30px;border-radius:16px;width:min(90vw,340px);
       border:1px solid #262b30}
  h1{font-size:17px;margin:0 0 6px;text-align:center}
  p{margin:0 0 22px;color:#8b959e;font-size:13px;text-align:center}
  label{display:block;font-size:12px;color:#8b959e;margin:14px 2px 6px}
  input{width:100%;box-sizing:border-box;padding:12px 14px;font-size:16px;
        border-radius:10px;border:1px solid #2d3339;background:#0e1012;color:#e9edf1}
  input:focus{outline:none;border-color:#3f8f6f}
  button{width:100%;margin-top:22px;padding:12px;border:0;border-radius:10px;
         background:#2f7d5f;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .err{color:#e0705f;font-size:13px;height:18px;margin-top:12px;text-align:center}
</style>
<form id="f">
  <h1>Νομικός βοηθός ΕΑΔΗΣΥ</h1>
  <p>Σύνδεση στον λογαριασμό σας</p>
  <label for="u">Όνομα χρήστη</label>
  <input id="u" autocomplete="username" autocapitalize="none" autofocus>
  <label for="pw">Κωδικός</label>
  <input id="pw" type="password" autocomplete="current-password">
  <button id="b">Είσοδος</button>
  <div class="err" id="e"></div>
</form>
<script>
const f=document.getElementById("f"),u=document.getElementById("u"),
      pw=document.getElementById("pw"),e=document.getElementById("e"),b=document.getElementById("b");
f.onsubmit=async ev=>{
  ev.preventDefault(); b.disabled=true; e.textContent="";
  try{
    const r=await fetch("/login",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({username:u.value.trim(),password:pw.value})});
    if(r.ok){ location.replace("/"); return; }
    e.textContent="Λάθος στοιχεία";
  }catch{ e.textContent="Σφάλμα σύνδεσης"; }
  b.disabled=false; pw.value=""; pw.focus();
};
</script>"""


def get_index(rebuild=False):
    global INDEX
    if INDEX is None or rebuild:
        INDEX = Index()
    return INDEX


def _engine_label():
    """Display name for whatever is actually answering, so the UI never has to
    hardcode a model name (it once claimed 'Krikri' on a box with no LLM)."""
    import config
    if config.ANSWER_ENGINE == "none":
        return "Αναζήτηση"
    try:
        from answer import engine
        return getattr(engine, "NAME", config.ANSWER_ENGINE)
    except Exception:
        return config.ANSWER_ENGINE


_READY_CACHE = (0.0, False)


def _answers_ready(ttl=30):
    """Is the answer engine actually usable? Cached — the Ollama backend probes
    the daemon over HTTP, and /health is hit constantly by the platform."""
    global _READY_CACHE
    import config
    if config.ANSWER_ENGINE == "none":
        return False
    now = time.monotonic()
    ts, val = _READY_CACHE
    if now - ts < ttl:
        return val
    try:
        from answer import engine
        val = bool(engine.available())
    except Exception:
        val = False
    _READY_CACHE = (now, val)
    return val


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    # --- helpers -------------------------------------------------------
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def _send(self, obj, code=200):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def _parts(self):
        return urlparse(self.path).path.strip("/").split("/")

    # --- Auth (username + password) -------------------------------------
    @staticmethod
    def _sign(username):
        import config
        return hmac.new(
            config.SECRET_KEY.encode(), username.encode(), hashlib.sha256
        ).hexdigest()

    def _user(self):
        """The logged-in username from a valid session cookie, else None."""
        import config
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "lb_session":
                uname, _, sig = v.partition("|")
                if uname in config.APP_USERS and hmac.compare_digest(sig, self._sign(uname)):
                    return uname
        return None

    def _owner(self):
        """Chat owner for this request: the logged-in user when accounts exist,
        otherwise the per-browser id (open localhost mode)."""
        import config
        return self._user() if config.APP_USERS else self._client_id()

    def _authed(self):
        import config
        if not config.APP_USERS:
            return True  # no accounts configured -> open (localhost default)
        return self._user() is not None

    def _require_auth(self, path):
        """True if the request may proceed. Otherwise it has already replied."""
        if path in OPEN_PATHS or self._authed():
            return True
        # Browsers navigating get the login page; API callers get a clean 401.
        if "text/html" in (self.headers.get("Accept") or ""):
            body = LOGIN_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send({"error": "unauthorized"}, 401)
        return False

    def _set_session(self, token, max_age):
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
        self.send_response(204)
        self.send_header(
            "Set-Cookie",
            f"lb_session={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}{secure}",
        )
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _do_login(self):
        import config
        body = self._body()
        uname = str(body.get("username") or "").strip()
        pw = str(body.get("password") or "")
        expected = config.APP_USERS.get(uname)
        if not expected or not hmac.compare_digest(pw, expected):
            time.sleep(1.0)  # blunt the brute-force rate
            return self._send({"error": "bad credentials"}, 401)
        self._set_session(f"{uname}|{self._sign(uname)}", 2592000)

    def _do_logout(self):
        self._set_session("", 0)

    def _client_id(self):
        """Per-browser id, so visitors don't share one global chat history.
        Header first; query param is the fallback for plain <a>/EventSource."""
        cid = (self.headers.get("X-Client-Id") or "").strip()
        if not cid:
            cid = (parse_qs(urlparse(self.path).query).get("cid") or [""])[0].strip()
        return cid[:64] or None

    def _serve_static(self, url_path):
        """Serve web/. Anything that escapes WEB_DIR is a 404, not a file read."""
        rel = unquote(url_path).lstrip("/") or "index.html"
        target = (WEB_DIR / rel).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            return self._send({"error": "not found"}, 404)
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            return self._send({"error": "not found"}, 404)

        body = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # --- CORS preflight ------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Client-Id")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # --- GET -----------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        p = self._parts()
        qs = parse_qs(u.query)

        if not self._require_auth(u.path):
            return

        if u.path == "/health":
            import config
            idx = get_index()
            return self._send({
                "ok": True,
                "chunks": idx.n_chunks,
                "decisions": len(idx.meta),
                "embed_backend": config.EMBED_BACKEND,
                "answer_engine": config.ANSWER_ENGINE,
                "engine_label": _engine_label(),
                # whether an answer will actually be written — the UI labels
                # itself off this, so a missing API key can't make it claim
                # an AI that never speaks
                "answers_ready": _answers_ready(),
                "user": self._user(),  # who's logged in (None if open/localhost)
            })

        if u.path == "/search":
            q = (qs.get("q") or [""])[0].strip()
            n = int((qs.get("n") or ["8"])[0])
            if not q:
                return self._send({"error": "missing q"}, 400)
            return self._send({"query": q, "results": get_index().search(q, k=n)})

        if u.path == "/chats":
            con = chatdb.connect(); chatdb.init(con)
            chats = chatdb.list_chats(con, self._owner()); con.close()
            return self._send({"chats": chats})

        # /chats/{id}/messages
        if len(p) == 3 and p[0] == "chats" and p[2] == "messages":
            con = chatdb.connect(); chatdb.init(con)
            msgs = chatdb.get_messages(con, int(p[1]), owner=self._owner())
            con.close()
            return self._send({"messages": msgs})

        if u.path == "/ask":
            q = (qs.get("q") or [""])[0].strip()
            chat_id = (qs.get("chat_id") or [""])[0].strip()
            if not q:
                return self._send({"error": "missing q"}, 400)
            return self._stream_ask(
                q,
                int(chat_id) if chat_id.isdigit() else None,
                owner=self._owner(),
            )

        return self._serve_static(u.path)

    # --- POST ----------------------------------------------------------
    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/login":
            return self._do_login()
        if u.path == "/logout":
            return self._do_logout()
        if not self._require_auth(u.path):
            return
        if u.path == "/reload":
            idx = get_index(rebuild=True)
            return self._send({"ok": True, "chunks": idx.n_chunks})
        if u.path == "/chats":
            title = (self._body().get("title") or "Νέα συζήτηση").strip() or "Νέα συζήτηση"
            con = chatdb.connect(); chatdb.init(con)
            cid = chatdb.create_chat(con, title, owner=self._owner()); con.close()
            return self._send({"id": cid, "title": title})
        return self._send({"error": "not found"}, 404)

    # --- DELETE --------------------------------------------------------
    def do_DELETE(self):
        p = self._parts()
        if not self._require_auth(urlparse(self.path).path):
            return
        if len(p) == 2 and p[0] == "chats" and p[1].isdigit():
            con = chatdb.connect(); chatdb.init(con)
            ok = chatdb.delete_chat(con, int(p[1]), owner=self._owner())
            con.close()
            return self._send({"ok": ok}, 200 if ok else 403)
        return self._send({"error": "not found"}, 404)

    # --- streaming chat with memory + persistence ----------------------
    def _stream_ask(self, q, chat_id, owner=None):
        from answer import answer_stream

        con = chatdb.connect(); chatdb.init(con)
        # An id we don't own (stale link, guessed number) must never be appended
        # to — start a fresh chat for this visitor instead.
        if chat_id is not None and not chatdb.owns(con, chat_id, owner):
            chat_id = None
        if chat_id is None:
            chat_id = chatdb.create_chat(con, q[:60], owner=owner)
        history = chatdb.get_messages(con, chat_id, owner=owner)  # turns BEFORE this one
        if not history:                               # first turn -> title the chat
            chatdb.set_title(con, chat_id, q[:60])
        chatdb.add_message(con, chat_id, "user", q)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        answer_text, sources = "", []
        try:
            emit({"chat_id": chat_id, "title": q[:60] if not history else None})
            for ev in answer_stream(get_index(), q, history=history):
                if ev.get("t"):
                    answer_text += ev["t"]
                elif ev.get("sources"):
                    sources = ev["sources"]
                emit(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client closed the tab mid-stream
        finally:
            if answer_text:
                chatdb.add_message(con, chat_id, "assistant", answer_text, sources)
            con.close()


def main():
    import config
    import embed

    if config.BOOTSTRAP:
        import bootstrap
        if not bootstrap.ensure_assets():
            raise SystemExit("bootstrap failed — refusing to start without an index")

    print("loading index…", flush=True)
    idx = get_index()
    con = chatdb.connect(); chatdb.init(con); con.close()
    print(f"index ready: {idx.n_chunks} chunks / {len(idx.meta)} decisions", flush=True)

    # pay the ONNX model load now, not on the first visitor's query
    embed.warmup()

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"lawbureaucracy listening on http://{HOST}:{PORT} "
        f"(embed={config.EMBED_BACKEND}, answers={config.ANSWER_ENGINE})",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
