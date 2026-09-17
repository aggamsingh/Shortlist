"""Request validation and auth tests for the /screen endpoint.

Bad input should be rejected at the edge with a 422 rather than reaching the
embedding model or, worse, burning an LLM call. These were written after a live
server accepted an empty job_description and returned 200.
"""

import os
import unittest
from unittest.mock import MagicMock

import api.main as main_module
from fastapi.testclient import TestClient

# No global patching here. Importing api.main does not construct the embedding
# model or a Qdrant client -- those are built in the lifespan handler, which
# TestClient only runs when used as a context manager. Module-level patchers
# would leak into every later test module (and CVEmbedder is a singleton, so a
# mocked model would persist), which is exactly what broke the e2e suite.

HEADERS = {"X-API-Key": "unit-test-key"}


def setUpModule():
    """Set the auth key for this module; main.py reads it per request."""
    os.environ["API_KEY"] = "unit-test-key"


class RequestValidationTest(unittest.TestCase):
    def setUp(self):
        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.retriever = MagicMock()
        main_module.retriever.search_candidates.return_value = []
        main_module.reranker = MagicMock()
        main_module.reranker.rerank.return_value = []
        self.client = TestClient(main_module.app)

    def post(self, payload):
        return self.client.post("/api/v1/screen", json=payload, headers=HEADERS)

    def test_valid_request_is_accepted(self):
        self.assertEqual(self.post({"job_description": "Python engineer"}).status_code, 200)

    def test_missing_job_description_rejected(self):
        self.assertEqual(self.post({}).status_code, 422)

    def test_empty_job_description_rejected(self):
        self.assertEqual(self.post({"job_description": ""}).status_code, 422)

    def test_whitespace_only_job_description_rejected(self):
        """min_length alone would accept '   ' and embed meaningless input."""
        self.assertEqual(self.post({"job_description": "    "}).status_code, 422)

    def test_embedder_not_called_for_invalid_input(self):
        """Validation must reject before any expensive work happens."""
        main_module.embedder.embed_text.reset_mock()
        self.post({"job_description": "   "})
        main_module.embedder.embed_text.assert_not_called()

    def test_top_k_bounds(self):
        self.assertEqual(self.post({"job_description": "x", "top_k": 0}).status_code, 422)
        self.assertEqual(self.post({"job_description": "x", "top_k": -1}).status_code, 422)
        self.assertEqual(self.post({"job_description": "x", "top_k": 99999}).status_code, 422)
        self.assertEqual(self.post({"job_description": "x", "top_k": 5}).status_code, 200)

    def test_top_k_wrong_type_rejected(self):
        self.assertEqual(self.post({"job_description": "x", "top_k": "many"}).status_code, 422)

    def test_negative_min_experience_rejected(self):
        payload = {"job_description": "x", "filters": {"min_experience": -5}}
        self.assertEqual(self.post(payload).status_code, 422)

    def test_absurd_min_experience_rejected(self):
        payload = {"job_description": "x", "filters": {"min_experience": 500}}
        self.assertEqual(self.post(payload).status_code, 422)

    def test_oversized_job_description_rejected(self):
        self.assertEqual(self.post({"job_description": "x" * 20001}).status_code, 422)

    def test_job_description_is_stripped(self):
        main_module.embedder.embed_text.reset_mock()
        self.post({"job_description": "  Python engineer  "})
        main_module.embedder.embed_text.assert_called_once_with("Python engineer")


class AuthTest(unittest.TestCase):
    def setUp(self):
        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.retriever = MagicMock()
        main_module.retriever.search_candidates.return_value = []
        main_module.reranker = MagicMock()
        self.client = TestClient(main_module.app)
        self.payload = {"job_description": "Python engineer"}

    def test_missing_key_rejected(self):
        self.assertEqual(
            self.client.post("/api/v1/screen", json=self.payload).status_code, 401
        )

    def test_wrong_key_rejected(self):
        response = self.client.post(
            "/api/v1/screen", json=self.payload, headers={"X-API-Key": "wrong"}
        )
        self.assertEqual(response.status_code, 401)

    def test_correct_key_accepted(self):
        response = self.client.post("/api/v1/screen", json=self.payload, headers=HEADERS)
        self.assertEqual(response.status_code, 200)

    def test_auth_checked_before_validation(self):
        """An unauthenticated caller should not learn about schema details."""
        response = self.client.post("/api/v1/screen", json={})
        self.assertEqual(response.status_code, 401)


class ResponseContractTest(unittest.TestCase):
    def setUp(self):
        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.retriever = MagicMock()
        main_module.reranker = MagicMock()
        self.client = TestClient(main_module.app)

    def test_empty_result_is_a_valid_response(self):
        main_module.retriever.search_candidates.return_value = []
        response = self.client.post(
            "/api/v1/screen", json={"job_description": "x"}, headers=HEADERS
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["candidates"], [])
        self.assertIn("job_id", body)
        self.assertTrue(body["screened_at"].endswith("Z"))

    def test_out_of_range_score_is_rejected_by_the_schema(self):
        """The response model is the last guard against a bad score leaking out."""
        main_module.retriever.search_candidates.return_value = [
            {"candidate_id": "c1", "name": "A", "cv_path": "/a.pdf",
             "score": 0.5, "resume_summary": "x"}
        ]
        main_module.reranker.rerank.return_value = [
            {"candidate_id": "c1", "name": "A", "score": 7.5,
             "match_reasoning": "bad score", "cv_path": "/a.pdf"}
        ]
        with self.assertRaises(Exception):
            self.client.post(
                "/api/v1/screen", json={"job_description": "x"}, headers=HEADERS
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
