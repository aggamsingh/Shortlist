"""Tests for the evaluation harness itself.

The harness produces the quality numbers quoted in the README, so its own
plumbing needs covering. In particular the --rerank branch cannot be exercised
without an API key, which is exactly how a code path rots: it would break at the
moment a key is first added. Here it runs against a stubbed provider.

A stub is used to prove the PLUMBING only. No stub-derived number is ever
reported as a quality result -- the README's reranker section stays explicitly
unmeasured until a real key runs it.
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient

from api.reranker import CVReranker
from evaluation import run_eval
from evaluation.corpus import CANDIDATES, HARD_QUERIES, QUERIES
from indexer.embedder import CVEmbedder


class EvalHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.embedder = CVEmbedder()
        except Exception as e:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"embedding model unavailable: {e}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="resume_evalharness_"))
        cls.client = QdrantClient(path=str(cls.tmp / "qdrant"))
        cls.n_chunks = run_eval.build_index(
            cls.client, cls.embedder, run_eval.CHUNKERS["section"]
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_index_covers_every_candidate(self):
        self.assertEqual(self.n_chunks, 4 * len(CANDIDATES))

    def test_grouped_retrieval_returns_distinct_candidates(self):
        vector = self.embedder.embed_text(QUERIES[0]["job_description"])
        ranked = run_eval.retrieve_grouped(self.client, vector, budget=10)
        ids = [cid for cid, _ in ranked]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 10)

    def test_grouped_beats_flat_on_pool_recall(self):
        """The core claim behind the grouping change, asserted as a test.

        Grouping must surface at least as many relevant candidates as flat
        chunk retrieval at the same budget. If this ever regresses, the README's
        justification for grouped retrieval is no longer true.
        """
        flat = run_eval.evaluate_config(
            self.client, self.embedder, "flat", budget=10, k=5, queries=HARD_QUERIES
        )
        grouped = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5, queries=HARD_QUERIES
        )
        self.assertGreater(grouped["pool_recall"], flat["pool_recall"])
        self.assertGreaterEqual(
            grouped["avg_candidates_retrieved"], flat["avg_candidates_retrieved"]
        )

    def test_metrics_are_in_valid_ranges(self):
        result = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5
        )
        for key in ("recall@5", "precision@5", "ndcg@5", "mrr", "pool_recall"):
            self.assertGreaterEqual(result[key], 0.0, key)
            self.assertLessEqual(result[key], 1.0, key)

    def test_within_role_queries_are_harder_than_cross_role(self):
        """If the hard set stops discriminating, the benchmark is saturated.

        A benchmark that scores ~1.0 everywhere cannot tell a good pipeline from
        a bad one, which is why the corpus carries a dense within-role cluster.
        """
        cross = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5, queries=QUERIES
        )
        within = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5, queries=HARD_QUERIES
        )
        self.assertLess(within["ndcg@5"], cross["ndcg@5"])
        self.assertLess(within["ndcg@5"], 0.95, "hard set has saturated")

    def test_rerank_branch_runs_with_a_stubbed_provider(self):
        """Covers the --rerank path that no API key is available to exercise."""
        reranker = CVReranker()
        reranker.gemini_key = "stub"

        def fake_call(jd, candidates):
            # Rank by candidate id so the ordering is deterministic and differs
            # from the vector ordering, proving the rerank branch is applied.
            body = {
                "rankings": [
                    {
                        "candidate_id": c["candidate_id"],
                        "score": round(1.0 - i / 100, 3),
                        "match_reasoning": "stubbed judgement",
                    }
                    for i, c in enumerate(sorted(candidates, key=lambda c: c["candidate_id"]))
                ]
            }
            return reranker._parse_rankings(json.dumps(body), "Gemini")

        reranker._rerank_with_gemini = fake_call

        result = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5,
            reranker=reranker, queries=HARD_QUERIES,
        )
        for key in ("recall@5", "ndcg@5", "pool_recall"):
            self.assertIn(key, result)
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    def test_rerank_does_not_change_the_retrieved_pool(self):
        """Reranking reorders; it must not add or drop candidates.

        pool_recall is the retrieval ceiling, so it has to be identical with and
        without a reranker. If it moves, the two stages are entangled.
        """
        reranker = CVReranker()
        reranker.gemini_key = "stub"
        reranker._rerank_with_gemini = lambda jd, c: reranker._parse_rankings(
            json.dumps({"rankings": [
                {"candidate_id": x["candidate_id"], "score": 0.5,
                 "match_reasoning": "stub"} for x in c
            ]}), "Gemini"
        )

        without = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5, queries=HARD_QUERIES
        )
        with_rerank = run_eval.evaluate_config(
            self.client, self.embedder, "grouped", budget=10, k=5,
            reranker=reranker, queries=HARD_QUERIES,
        )
        self.assertAlmostEqual(without["pool_recall"], with_rerank["pool_recall"])
        self.assertEqual(
            without["avg_candidates_retrieved"], with_rerank["avg_candidates_retrieved"]
        )

    def test_chunkers_all_produce_output(self):
        text = "Jane Doe\n\nSummary\nEngineer with 5 years of experience.\n\nSkills\nPython"
        for name, chunker in run_eval.CHUNKERS.items():
            chunks = chunker(text)
            self.assertGreater(len(chunks), 0, name)
            self.assertTrue(all(c.strip() for c in chunks), name)


class CheckLlmTest(unittest.TestCase):
    """The key-verification helper must fail cleanly rather than crash."""

    def test_reports_failure_when_no_provider_configured(self):
        import evaluation.check_llm as check_llm

        original = CVReranker.__init__

        def unconfigured(self):
            original(self)
            self.gemini_key = None
            self.groq_key = None

        CVReranker.__init__ = unconfigured
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_llm.main(), 1)
        finally:
            CVReranker.__init__ = original

    def test_reports_success_with_a_working_stub(self):
        import evaluation.check_llm as check_llm

        original = CVReranker.__init__

        def stubbed(self):
            original(self)
            self.gemini_key = "stub"
            self.groq_key = None
            self._rerank_with_gemini = lambda jd, c: self._parse_rankings(
                json.dumps({"rankings": [
                    {"candidate_id": "c_vector", "score": 0.95, "match_reasoning": "vector db"},
                    {"candidate_id": "c_django", "score": 0.40, "match_reasoning": "no vectors"},
                    {"candidate_id": "c_frontend", "score": 0.05, "match_reasoning": "frontend"},
                ]}), "Gemini"
            )

        CVReranker.__init__ = stubbed
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_llm.main(), 0)
        finally:
            CVReranker.__init__ = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
