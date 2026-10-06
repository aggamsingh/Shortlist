"""What POST /screen tells the caller about how the ranking was produced.

The project's recurring failure class is the silent one: a ranking that looks
fine but was produced by a degraded path. The reranker can be down, over quota,
or switched off, and in every case it falls back to retrieval order. That is the
right behaviour, but until the response carried `reranked` the caller had no way
to tell the two outcomes apart -- the flag existed only on the stored screening.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api.main as main_module
from api.cross_encoder import CrossEncoderReranker
from api.main import NoOpReranker, app
from api.reranker import CVReranker

KEY = "screen-response-test-key"
HEADERS = {"X-API-Key": KEY}
PAYLOAD = {"job_description": "Senior Python backend engineer with FastAPI.", "top_k": 3}

RETRIEVED = [
    {"candidate_id": "c1", "name": "Ada", "cv_path": "/a.pdf", "score": 0.9,
     "years_of_experience": 5, "location": "Pune", "resume_summary": "python"},
    {"candidate_id": "c2", "name": "Bo", "cv_path": "/b.pdf", "score": 0.7,
     "years_of_experience": 3, "location": "Delhi", "resume_summary": "react"},
]

JUDGED = "Strong match on FastAPI."
FALLBACK = "Vector-similarity match (not scored by reranker)."


def row(cid, name, score, reasoning):
    return {"candidate_id": cid, "name": name, "score": score,
            "match_reasoning": reasoning, "cv_path": f"/{cid}.pdf"}


class ScreenResponseTest(unittest.TestCase):
    GLOBALS = ("embedder", "retriever", "reranker", "store")

    def setUp(self):
        # Restore the real singletons afterwards: other modules replace these too,
        # and a leak here would hand them a mock.
        self._saved = {name: getattr(main_module, name) for name in self.GLOBALS}
        self._saved_key = os.environ.get("API_KEY")
        os.environ["API_KEY"] = KEY

        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.retriever = MagicMock()
        main_module.retriever.search_candidates.return_value = list(RETRIEVED)
        main_module.store = MagicMock()
        self.client = TestClient(app)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(main_module, name, value)
        if self._saved_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = self._saved_key

    def screen(self):
        response = self.client.post("/api/v1/screen", json=PAYLOAD, headers=HEADERS)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def use_reranker(self, rows):
        main_module.reranker = MagicMock()
        main_module.reranker.rerank.return_value = rows

    # ---- reranked ----

    def test_reranked_true_when_the_reranker_scored_candidates(self):
        self.use_reranker([row("c1", "Ada", 0.95, JUDGED), row("c2", "Bo", 0.4, JUDGED)])
        self.assertIs(self.screen()["reranked"], True)

    def test_reranked_false_when_every_candidate_fell_back(self):
        """The case the flag exists for: a provider outage or exhausted quota."""
        self.use_reranker([row("c1", "Ada", 0.9, FALLBACK), row("c2", "Bo", 0.7, FALLBACK)])
        self.assertIs(self.screen()["reranked"], False)

    def test_reranked_true_when_only_some_were_scored(self):
        """Partial success is still reranking: the judged block leads the list."""
        self.use_reranker([row("c1", "Ada", 0.95, JUDGED), row("c2", "Bo", 0.7, FALLBACK)])
        self.assertIs(self.screen()["reranked"], True)

    def test_reranked_false_with_the_none_backend(self):
        """Retrieval-only is a supported configuration, and must say so."""
        main_module.reranker = NoOpReranker()
        body = self.screen()
        self.assertIs(body["reranked"], False)
        self.assertEqual([c["name"] for c in body["candidates"]], ["Ada", "Bo"])

    def test_reranked_false_when_nothing_matched(self):
        main_module.retriever.search_candidates.return_value = []
        main_module.reranker = MagicMock()
        body = self.screen()
        self.assertIs(body["reranked"], False)
        self.assertEqual(body["candidates"], [])
        main_module.reranker.rerank.assert_not_called()

    def test_the_response_agrees_with_the_stored_screening(self):
        """One definition of 'reranked', not two that can drift apart."""
        self.use_reranker([row("c1", "Ada", 0.9, FALLBACK), row("c2", "Bo", 0.7, FALLBACK)])
        body = self.screen()
        stored = main_module.store.save_screening.call_args.kwargs["reranked"]
        self.assertEqual(body["reranked"], stored)

    # ---- timings ----

    def test_timings_cover_every_stage(self):
        self.use_reranker([row("c1", "Ada", 0.95, JUDGED)])
        timings = self.screen()["timings"]
        for stage in ("embed_ms", "retrieve_ms", "rerank_ms", "total_ms"):
            self.assertIn(stage, timings)
            self.assertIsInstance(timings[stage], (int, float))
            self.assertGreaterEqual(timings[stage], 0)

    def test_total_is_the_sum_of_the_stages(self):
        self.use_reranker([row("c1", "Ada", 0.95, JUDGED)])
        t = self.screen()["timings"]
        parts = t["embed_ms"] + t["retrieve_ms"] + t["rerank_ms"]
        self.assertAlmostEqual(t["total_ms"], parts, delta=0.5)

    # ---- backwards compatibility ----

    def test_existing_fields_are_unchanged(self):
        """Additive change: nothing a current client reads may move."""
        self.use_reranker([row("c1", "Ada", 0.95, JUDGED)])
        body = self.screen()
        for field in ("job_id", "candidates", "screened_at"):
            self.assertIn(field, body)
        candidate = body["candidates"][0]
        for field in ("candidate_id", "name", "score", "match_reasoning", "cv_path"):
            self.assertIn(field, candidate)


class HealthBackendTest(unittest.TestCase):
    """`llm_configured` has to mean an LLM, not 'some reranker loaded'."""

    def setUp(self):
        self._saved = {n: getattr(main_module, n) for n in ("embedder", "retriever", "reranker")}
        main_module.embedder = MagicMock()
        main_module.retriever = MagicMock()
        main_module.retriever.client.get_collections.return_value = None
        self.client = TestClient(app)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(main_module, name, value)

    def health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_llm_backend_reports_llm_configured(self):
        with patch.object(CVReranker, "is_configured", new=property(lambda self: True)):
            main_module.reranker = CVReranker()
            body = self.health()
        self.assertIs(body["llm_configured"], True)
        self.assertEqual(body["reranker_backend"], "CVReranker")

    def test_cross_encoder_does_not_claim_an_llm(self):
        """The bug found by running the real flow: with the cross-encoder live,
        /health said llm_configured true, telling an operator a key existed when
        none was involved."""
        with patch.object(
            CrossEncoderReranker, "is_configured", new=property(lambda self: True)
        ):
            main_module.reranker = CrossEncoderReranker()
            body = self.health()
        self.assertIs(body["llm_configured"], False)
        self.assertEqual(body["reranker_backend"], "CrossEncoderReranker")

    def test_none_backend_does_not_claim_an_llm(self):
        main_module.reranker = NoOpReranker()
        body = self.health()
        self.assertIs(body["llm_configured"], False)
        self.assertEqual(body["reranker_backend"], "NoOpReranker")


if __name__ == "__main__":
    unittest.main(verbosity=2)
