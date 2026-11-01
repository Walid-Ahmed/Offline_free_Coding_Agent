import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from offline_agent.agent import run
from offline_agent.model import MODEL, Ollama
from offline_agent.workspace import Workspace


def action(kind, path="", content="", message="Working"):
    return dict(action=kind, path=path, content=content, message=message)


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, messages, schema):
        return next(self.replies)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)

    def test_reject_escapes_and_hidden_paths(self):
        for path in ["../outside", "/tmp/outside", ".git/config", ".env", "a/../../x", ""]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.ws.path(path)

    def test_symlinks_blocked(self):
        (self.root / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.ws.read("link/example.py")
        self.assertEqual(self.ws.files(), [])

    def test_preview_and_confirmed_write(self):
        replies = [action("write", "hello.py", "print('hello')\n"), action("finish")]
        run(FakeModel(replies), self.ws, "Create hello", output=lambda _: None)
        self.assertFalse((self.root / "hello.py").exists())
        run(FakeModel(replies), self.ws, "Create hello", apply=True,
            confirm=lambda _: True, output=lambda _: None)
        self.assertEqual(self.ws.read("hello.py"), "print('hello')\n")

    def test_decline_write(self):
        run(FakeModel([action("write", "x", "text"), action("finish")]), self.ws,
            "task", apply=True, confirm=lambda _: False, output=lambda _: None)
        self.assertFalse((self.root / "x").exists())

    def test_stale_preview(self):
        (self.root / "x").write_text("original")
        before, _ = self.ws.propose("x", "new")
        (self.root / "x").write_text("user edit")
        with self.assertRaises(ValueError):
            self.ws.write("x", "new", before)
        self.assertEqual(self.ws.read("x"), "user edit")

    def test_size_binary_and_hardlink(self):
        (self.root / "big").write_bytes(b"x" * 64001)
        (self.root / "binary").write_bytes(b"\0")
        for name in ["big", "binary"]:
            with self.assertRaises(ValueError):
                self.ws.read(name)
        (self.root / "x").write_text("text")
        (self.root / "y").hardlink_to(self.root / "x")
        with self.assertRaises(ValueError):
            self.ws.propose("y", "edit")

    def test_step_limit_and_invalid_action(self):
        self.assertEqual(run(FakeModel([action("read", "missing")]), self.ws,
                             "task", max_steps=1, output=lambda _: None), 2)
        with self.assertRaises(ValueError):
            run(FakeModel([{"action": "shell"}]), self.ws, "task")


class TransportTests(unittest.TestCase):
    def test_remote_endpoints_rejected(self):
        for endpoint in ["https://example.com", "http://localhost:11434", "http://10.0.0.1",
                         "http://127.0.0.1/evil", "http://user@127.0.0.1", "http://127.0.0.1:bad"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                Ollama(endpoint=endpoint)
        with self.assertRaises(ValueError):
            Ollama("model:cloud")

    def test_only_selected_model_supported(self):
        self.assertEqual(Ollama().model, "qwen2.5-coder:7b")
        for name in ["other", "qwen2.5-coder:14b", "qwen2.5-coder:latest"]:
            with self.subTest(model=name), self.assertRaises(ValueError):
                Ollama(name)

    def test_real_http_contract(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_payload({"models": [{"name": MODEL}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, payload))
                if self.path == "/api/show":
                    self.send_payload({})
                else:
                    self.send_payload({"message": {"content": json.dumps(action("finish"))}})

            def send_payload(self, value):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                client = Ollama(endpoint=f"http://127.0.0.1:{server.server_port}")
                client.check_model()
                self.assertEqual(client.chat([], {} )["action"], "finish")
                self.assertFalse(calls[-1][1]["stream"])
                self.assertTrue(all(payload["model"] == MODEL for _, payload in calls))
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
