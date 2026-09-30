import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from starlette.testclient import TestClient

from coderag.chunker import chunk_repo
from coderag.index import CodeIndex
from coderag.qa import GeneratorError, OllamaGenerator, answer_question
from coderag.service import create_app
from tests.helpers import make_repo


class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        self.server.last = json.loads(self.rfile.read(n))
        body = json.dumps({"response": self.server.reply}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CodeIndex.build(chunk_repo(make_repo(), "", hide_docstrings=False, exclude_tests=True))
        cls.client = TestClient(create_app(cls.index))

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_search(self):
        r = self.client.post("/v1/search", json={"query": "check a password against its digest", "k": 3})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["results"][0]["name"], "verify_password")

    def test_validation_errors(self):
        self.assertEqual(self.client.post("/v1/search", json={"query": "x"}).status_code, 422)
        self.assertEqual(self.client.post("/v1/search", json={"query": "valid query", "mode": "zzz"}).status_code, 422)
        self.assertEqual(self.client.post("/v1/search", content=b"not json").status_code, 400)

    def test_ask(self):
        r = self.client.post("/v1/ask", json={"question": "how do I hash a password?"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["citations"]["grounded"])


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.index = CodeIndex.build(chunk_repo(make_repo(), "", hide_docstrings=False, exclude_tests=True))
        self.srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
        self.srv.reply = "Passwords are hashed with sha256 [1]. Also see [7]."
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.gen = OllamaGenerator("fake", f"http://127.0.0.1:{self.srv.server_port}", timeout=5)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_prompt_contains_code_and_invented_citation_is_flagged(self):
        r = answer_question(self.index, "how are passwords hashed", self.gen, k=2)
        self.assertIn("hash_password", self.srv.last["prompt"])
        self.assertEqual(self.srv.last["options"]["temperature"], 0)
        self.assertEqual(r["citations"]["invalid"], [7])
        self.assertFalse(r["citations"]["grounded"])

    def test_unreachable_server_raises_clear_error(self):
        bad = OllamaGenerator("fake", "http://127.0.0.1:9", timeout=2)
        with self.assertRaises(GeneratorError):
            bad.generate("q", self.index.search("password", k=1))

    def test_service_returns_502_when_generator_down(self):
        bad = OllamaGenerator("fake", "http://127.0.0.1:9", timeout=2)
        c = TestClient(create_app(self.index, bad))
        self.assertEqual(c.post("/v1/ask", json={"question": "hash password"}).status_code, 502)


if __name__ == "__main__":
    unittest.main()
