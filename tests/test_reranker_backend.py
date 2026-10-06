"""Selecting the reranking backend.

Three backends now satisfy the same contract, so the risk shifts from "does
reranking work" to "is the service running the backend the operator asked for".
A silent fallback is the dangerous failure here: it produces plausible results
from the wrong component, which no amount of inspecting the output would reveal.
"""

import unittest
from unittest.mock import patch

from api.cross_encoder import CrossEncoderReranker
from api.main import NoOpReranker, build_reranker
from api.reranker import CVReranker


class BackendSelectionTest(unittest.TestCase):
    def setUp(self):
        CrossEncoderReranker._model = None
        CrossEncoderReranker._load_failed = False

    tearDown = setUp

    def build(self, env):
        with patch.dict("os.environ", env, clear=False):
            return build_reranker()

    def test_default_is_the_llm(self):
        """The service used the LLM before this existed. A changed default
        would silently alter every existing deployment's ordering."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("RERANKER_BACKEND", None)
            self.assertIsInstance(build_reranker(), CVReranker)

    def test_explicit_llm(self):
        self.assertIsInstance(self.build({"RERANKER_BACKEND": "llm"}), CVReranker)

    def test_cross_encoder_selected(self):
        with patch.object(
            CrossEncoderReranker, "is_configured", new=property(lambda self: True)
        ):
            built = self.build({"RERANKER_BACKEND": "cross-encoder"})
        self.assertIsInstance(built, CrossEncoderReranker)

    def test_cross_encoder_aliases(self):
        with patch.object(
            CrossEncoderReranker, "is_configured", new=property(lambda self: True)
        ):
            for alias in ("cross", "cross-encoder", "cross_encoder", "CROSS-ENCODER"):
                self.assertIsInstance(
                    self.build({"RERANKER_BACKEND": alias}),
                    CrossEncoderReranker,
                    alias,
                )

    def test_none_backend_is_supported_not_an_error(self):
        """Reranking hurts on easy queries, so retrieval-only is a legitimate
        configuration rather than a misconfiguration."""
        for alias in ("none", "off", "retrieval"):
            self.assertIsInstance(
                self.build({"RERANKER_BACKEND": alias}), NoOpReranker, alias
            )

    def test_unknown_backend_falls_back_to_llm_loudly(self):
        with self.assertLogs("api.main", level="WARNING") as logs:
            built = self.build({"RERANKER_BACKEND": "magic"})
        self.assertIsInstance(built, CVReranker)
        self.assertTrue(any("magic" in line for line in logs.output))

    def test_cross_encoder_failure_does_not_silently_become_the_llm(self):
        """An operator who asked for the cross-encoder and got LLM results
        would be comparing the wrong two things and have no way to tell."""
        with patch(
            "sentence_transformers.CrossEncoder", side_effect=OSError("no network")
        ):
            with self.assertLogs("api.main", level="ERROR"):
                built = self.build({"RERANKER_BACKEND": "cross"})
        self.assertIsInstance(built, CrossEncoderReranker)
        self.assertNotIsInstance(built, CVReranker)


class NoOpRerankerTest(unittest.TestCase):
    def rows(self):
        return [
            {"candidate_id": "c1", "name": "A", "score": 0.3, "cv_path": "/a.pdf"},
            {"candidate_id": "c2", "name": "B", "score": 0.9, "cv_path": "/b.pdf"},
        ]

    def test_orders_by_retrieval_score(self):
        results = NoOpReranker().rerank("jd", self.rows(), top_k=2)
        self.assertEqual([r["candidate_id"] for r in results], ["c2", "c1"])

    def test_respects_top_k(self):
        self.assertEqual(len(NoOpReranker().rerank("jd", self.rows(), top_k=1)), 1)

    def test_empty_input(self):
        self.assertEqual(NoOpReranker().rerank("jd", [], top_k=5), [])

    def test_reports_itself_as_not_reranked(self):
        """The API derives its `reranked` response field from this phrase, so a
        no-op backend must not claim the results were reranked."""
        results = NoOpReranker().rerank("jd", self.rows(), top_k=2)
        for row in results:
            self.assertIn("not scored by reranker", row["match_reasoning"])

    def test_is_configured_is_false(self):
        self.assertFalse(NoOpReranker().is_configured)

    def test_emits_every_field_the_response_model_needs(self):
        """A missing key here surfaces as a 500 at response-serialisation time,
        well away from the cause."""
        required = {"candidate_id", "name", "score", "match_reasoning", "cv_path"}
        for row in NoOpReranker().rerank("jd", self.rows(), top_k=2):
            self.assertEqual(required - set(row), set())


class ContractParityTest(unittest.TestCase):
    """All three backends are used interchangeably by the request path, so they
    have to agree on more than the method name."""

    def test_all_backends_expose_the_same_surface(self):
        for cls in (CVReranker, CrossEncoderReranker, NoOpReranker):
            self.assertTrue(hasattr(cls, "rerank"), cls.__name__)
            self.assertTrue(hasattr(cls, "is_configured"), cls.__name__)

    def test_all_backends_return_empty_for_empty_input(self):
        CrossEncoderReranker._model = None
        CrossEncoderReranker._load_failed = True  # avoid a real model load
        try:
            for backend in (CVReranker(), CrossEncoderReranker(), NoOpReranker()):
                self.assertEqual(
                    backend.rerank("jd", [], top_k=5), [], type(backend).__name__
                )
        finally:
            CrossEncoderReranker._load_failed = False

    def test_all_backends_produce_the_same_keys(self):
        candidates = [
            {
                "candidate_id": "c1",
                "name": "A",
                "score": 0.5,
                "cv_path": "/a.pdf",
                "resume_summary": "python backend engineer",
            }
        ]
        CrossEncoderReranker._model = None
        CrossEncoderReranker._load_failed = False

        with patch.object(
            CrossEncoderReranker, "_score_pairs", return_value=[1.0]
        ):
            cross_rows = CrossEncoderReranker().rerank("jd", candidates, top_k=1)
        noop_rows = NoOpReranker().rerank("jd", candidates, top_k=1)

        self.assertEqual(set(cross_rows[0]), set(noop_rows[0]))
        # And neither leaks the internal sort flag into the response.
        self.assertNotIn("_judged", cross_rows[0])
        self.assertNotIn("_judged", noop_rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
