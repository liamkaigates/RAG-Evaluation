import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from rag_eval.api import create_app
from rag_eval.data import load_documents, load_questions
from rag_eval.retriever import build_index, load_index, save_index


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("RAG_EMBEDDING_PROVIDER", "local")
        os.environ.setdefault("RAG_GENERATION_PROVIDER", "local")
        cls._tmp = TemporaryDirectory()
        index = build_index(load_documents("data/corpus.jsonl"), embedding_provider="local")
        save_index(index, Path(cls._tmp.name) / "index")
        cls.index = load_index(Path(cls._tmp.name) / "index", embedding_provider="local")
        cls.questions = load_questions("data/eval_questions.jsonl")
        cls.client = TestClient(create_app(cls.index, cls.questions, static_dir="static"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_auth_required_when_token_set(self):
        os.environ["RAG_API_TOKEN"] = "secret-token"
        try:
            client = TestClient(create_app(self.index, self.questions, static_dir="static"))
            denied = client.get("/api/summary")
            wrong_key = client.get("/api/summary", headers={"X-API-Key": "wrong"})
            with_key = client.get("/api/summary", headers={"X-API-Key": "secret-token"})
            with_bearer = client.get("/api/summary", headers={"Authorization": "Bearer secret-token"})
            health = client.get("/healthz")
        finally:
            del os.environ["RAG_API_TOKEN"]
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(wrong_key.status_code, 401)
        self.assertEqual(with_key.status_code, 200)
        self.assertEqual(with_bearer.status_code, 200)
        self.assertEqual(health.status_code, 200)

    def test_healthz(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_summary(self):
        response = self.client.get("/api/summary")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["chunks"], len(self.index.documents))
        self.assertEqual(payload["questions"], len(self.questions))

    def test_search(self):
        response = self.client.get(
            "/api/search", params={"question": "Which metrics evaluate retrieval quality?", "top_k": 3}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["answer"])
        self.assertTrue(payload["results"])
        self.assertTrue(payload["citations"])

    def test_evaluate_is_cached(self):
        first = self.client.get("/api/evaluate", params={"top_k": 4})
        second = self.client.get("/api/evaluate", params={"top_k": 4})
        refreshed = self.client.get("/api/evaluate", params={"top_k": 4, "refresh": "true"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json()["summary"], refreshed.json()["summary"])
        self.assertEqual(first.json()["summary"]["question_count"], float(len(self.questions)))


if __name__ == "__main__":
    unittest.main()
