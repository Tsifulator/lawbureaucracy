"""Localhost API for the web UI: search, streaming chat with memory, and
server-side chat storage (so history survives refresh and moves with the brain).

Stdlib only (no FastAPI/pydantic — dodges the Rust wheel problem on py3.14).
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import chatdb
from config import HOST, PORT
from search import Index

INDEX = None


def get_index(rebuild=False):
    global INDEX
    if INDEX is None or rebuild:
        INDEX = Index()
    return INDEX


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

    # --- CORS preflight ------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # --- GET -----------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        p = self._parts()
        qs = parse_qs(u.query)

        if u.path == "/health":
            idx = get_index()
            return self._send({"ok": True, "chunks": idx.n_chunks, "decisions": len(idx.meta)})

        if u.path == "/search":
            q = (qs.get("q") or [""])[0].strip()
            n = int((qs.get("n") or ["8"])[0])
            if not q:
                return self._send({"error": "missing q"}, 400)
            return self._send({"query": q, "results": get_index().search(q, k=n)})

        if u.path == "/chats":
            con = chatdb.connect(); chatdb.init(con)
            chats = chatdb.list_chats(con); con.close()
            return self._send({"chats": chats})

        # /chats/{id}/messages
        if len(p) == 3 and p[0] == "chats" and p[2] == "messages":
            con = chatdb.connect(); chatdb.init(con)
            msgs = chatdb.get_messages(con, int(p[1])); con.close()
            return self._send({"messages": msgs})

        if u.path == "/ask":
            q = (qs.get("q") or [""])[0].strip()
            chat_id = (qs.get("chat_id") or [""])[0].strip()
            if not q:
                return self._send({"error": "missing q"}, 400)
            return self._stream_ask(q, int(chat_id) if chat_id.isdigit() else None)

        return self._send({"error": "not found"}, 404)

    # --- POST ----------------------------------------------------------
    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/reload":
            idx = get_index(rebuild=True)
            return self._send({"ok": True, "chunks": idx.n_chunks})
        if u.path == "/chats":
            title = (self._body().get("title") or "Νέα συζήτηση").strip() or "Νέα συζήτηση"
            con = chatdb.connect(); chatdb.init(con)
            cid = chatdb.create_chat(con, title); con.close()
            return self._send({"id": cid, "title": title})
        return self._send({"error": "not found"}, 404)

    # --- DELETE --------------------------------------------------------
    def do_DELETE(self):
        p = self._parts()
        if len(p) == 2 and p[0] == "chats" and p[1].isdigit():
            con = chatdb.connect(); chatdb.init(con)
            chatdb.delete_chat(con, int(p[1])); con.close()
            return self._send({"ok": True})
        return self._send({"error": "not found"}, 404)

    # --- streaming chat with memory + persistence ----------------------
    def _stream_ask(self, q, chat_id):
        from answer import answer_stream

        con = chatdb.connect(); chatdb.init(con)
        new_chat = chat_id is None
        if new_chat:
            chat_id = chatdb.create_chat(con, q[:60])
        history = chatdb.get_messages(con, chat_id)   # turns BEFORE this one
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
    print("loading index…")
    idx = get_index()
    con = chatdb.connect(); chatdb.init(con); con.close()
    print(f"index ready: {idx.n_chunks} chunks / {len(idx.meta)} decisions")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"lawbureaucracy brain listening on http://{HOST}:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
