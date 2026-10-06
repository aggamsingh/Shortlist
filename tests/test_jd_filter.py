"""The job-description boilerplate filter, the styles used to evaluate it, and its
wiring into POST /screen.

The behaviour tests use the real embedder, because the filter's whole job is a
semantic judgement that a mock cannot make. The wiring tests use mocks, because
what they assert is plumbing: which text reaches which stage.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api.main as main_module
from api.jd_filter import JDFilter, build_jd_filter, split_sentences
from evaluation.corpus import select_queries
from evaluation.jd_styles import STYLES, check_allowed, wrap
from indexer.embedder import CVEmbedder

REQUIREMENT = "You must have strong Python experience with FastAPI, Docker and PostgreSQL."
ABOUT_US = "About us. We are a fast growing company with a great culture and a clear mission."
BENEFITS = "Benefits include health insurance, generous paid leave and a home office budget."


class SplitSentencesTest(unittest.TestCase):
    def test_splits_on_sentence_punctuation(self):
        self.assertEqual(split_sentences("One. Two! Three?"), ["One.", "Two!", "Three?"])

    def test_splits_bullets_on_newlines(self):
        self.assertEqual(split_sentences("- a\n- b\n- c"), ["- a", "- b", "- c"])

    def test_splits_after_a_heading_colon(self):
        self.assertEqual(split_sentences("Perks: hybrid working"), ["Perks:", "hybrid working"])

    def test_drops_empty_pieces(self):
        self.assertEqual(split_sentences("a.\n\n  \nb."), ["a.", "b."])

    def test_empty_and_none(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences(None), [])


class JDFilterBehaviourTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.embedder = CVEmbedder()
        except Exception as e:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"embedding model unavailable: {e}")
        cls.filter = JDFilter(cls.embedder)

    def test_removes_company_and_benefits_text_and_keeps_the_requirement(self):
        text = " ".join([ABOUT_US, REQUIREMENT, BENEFITS])
        result = self.filter.apply(text)
        self.assertIn("Python", result)
        self.assertNotIn("great culture", result)
        self.assertNotIn("health insurance", result)

    def test_never_returns_an_empty_query(self):
        """A filter that can erase the query turns a bad guess into a failed search."""
        text = ABOUT_US + " " + BENEFITS
        self.assertEqual(self.filter.apply(text), text)

    def test_a_single_sentence_is_returned_unchanged(self):
        self.assertEqual(self.filter.apply(ABOUT_US), ABOUT_US)

    def test_empty_and_whitespace_are_returned_unchanged(self):
        self.assertEqual(self.filter.apply(""), "")
        self.assertEqual(self.filter.apply("   "), "   ")

    def test_preserves_sentence_order(self):
        first = "Experience with Kubernetes and Terraform is required."
        second = "You will own CI/CD pipelines and Docker deployments."
        result = self.filter.apply(" ".join([ABOUT_US, first, BENEFITS, second]))
        self.assertLess(result.index("Kubernetes"), result.index("CI/CD"))

    def test_is_deterministic(self):
        text = " ".join([ABOUT_US, REQUIREMENT, BENEFITS])
        self.assertEqual(self.filter.apply(text), self.filter.apply(text))

    def test_handles_unicode_without_crashing(self):
        text = "हमारी कंपनी बहुत अच्छी है। " + REQUIREMENT + " 日本語のテキスト。"
        self.assertTrue(self.filter.apply(text))

    def test_a_threshold_that_drops_everything_falls_back_to_the_original(self):
        strict = JDFilter(self.embedder, threshold=10.0)
        text = REQUIREMENT + " " + ABOUT_US
        self.assertEqual(strict.apply(text), text)

    def test_a_threshold_that_keeps_everything_changes_nothing(self):
        lenient = JDFilter(self.embedder, threshold=-10.0)
        text = ABOUT_US + " " + REQUIREMENT
        self.assertEqual(lenient.apply(text), text.strip())

    def test_clean_job_descriptions_are_mostly_untouched(self):
        """Measured: 96% of words survive on the labelled queries. If this drops
        sharply, the filter has started eating real requirements."""
        kept = []
        for query in select_queries("all"):
            original = query["job_description"]
            kept.append(len(self.filter.apply(original).split()) / len(original.split()))
        self.assertGreater(sum(kept) / len(kept), 0.85)

    def test_the_threshold_can_be_set_from_the_environment(self):
        with patch.dict(os.environ, {"JD_FILTER_THRESHOLD": "0.5"}):
            self.assertEqual(JDFilter(self.embedder).threshold, 0.5)


class BuildFilterTest(unittest.TestCase):
    def test_disabled_by_environment(self):
        for value in ("false", "0", "no", "off", "FALSE"):
            with patch.dict(os.environ, {"JD_FILTER": value}):
                self.assertIsNone(build_jd_filter(MagicMock()), value)

    def test_enabled_by_default(self):
        embedder = MagicMock()
        embedder.embed_text.return_value = [0.1] * 4
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JD_FILTER", None)
            self.assertIsInstance(build_jd_filter(embedder), JDFilter)


class JDStylesTest(unittest.TestCase):
    def test_plain_is_the_identity(self):
        self.assertEqual(wrap("requirements", "plain"), "requirements")

    def test_wrapping_adds_text_on_both_sides(self):
        for style in STYLES:
            wrapped = wrap("REQS", style)
            self.assertIn("REQS", wrapped)
            self.assertGreater(wrapped.index("REQS"), 50, style)
            self.assertGreater(len(wrapped) - wrapped.index("REQS") - 4, 50, style)

    def test_unknown_style_is_rejected(self):
        with self.assertRaises(ValueError):
            wrap("x", "Z")

    def test_style_c_is_held_out(self):
        self.assertEqual(STYLES["C"]["split"], "test")
        self.assertEqual(STYLES["A"]["split"], "dev")
        self.assertEqual(STYLES["B"]["split"], "dev")

    def test_the_held_out_style_is_refused_on_dev_and_all(self):
        """Using it to choose anything makes it a dev style and destroys the
        confirmation it exists to provide."""
        for split in ("dev", "all"):
            with self.assertRaises(ValueError):
                check_allowed("C", split)
        check_allowed("C", "test")  # allowed

    def test_dev_styles_are_allowed_everywhere(self):
        for style in ("plain", "A", "B"):
            for split in ("dev", "test", "all"):
                check_allowed(style, split)

    def test_styles_are_independent_texts(self):
        texts = [STYLES[s]["before"] + STYLES[s]["after"] for s in STYLES]
        self.assertEqual(len(set(texts)), len(texts))


class ServiceWiringTest(unittest.TestCase):
    """Which text reaches which stage of POST /screen."""

    KEY = "jd-filter-test-key"
    GLOBALS = ("embedder", "retriever", "reranker", "store", "jd_filter")
    RETRIEVED = [
        {"candidate_id": "c1", "name": "Ada", "cv_path": "/a.pdf", "score": 0.9,
         "years_of_experience": 5, "location": "Pune", "resume_summary": "python"},
    ]
    RERANKED = [{"candidate_id": "c1", "name": "Ada", "score": 0.9,
                 "match_reasoning": "good", "cv_path": "/a.pdf"}]

    def setUp(self):
        self._saved = {n: getattr(main_module, n) for n in self.GLOBALS}
        self._saved_key = os.environ.get("API_KEY")
        os.environ["API_KEY"] = self.KEY
        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.retriever = MagicMock()
        main_module.retriever.search_candidates.return_value = list(self.RETRIEVED)
        main_module.reranker = MagicMock()
        main_module.reranker.rerank.return_value = list(self.RERANKED)
        main_module.store = MagicMock()
        self.client = TestClient(main_module.app)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(main_module, name, value)
        if self._saved_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = self._saved_key

    def screen(self, jd="ORIGINAL JOB DESCRIPTION"):
        response = self.client.post(
            "/api/v1/screen",
            json={"job_description": jd, "top_k": 3},
            headers={"X-API-Key": self.KEY},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_retrieval_gets_the_filtered_text_and_the_reranker_the_original(self):
        main_module.jd_filter = MagicMock()
        main_module.jd_filter.apply.return_value = "FILTERED"
        self.screen()

        main_module.embedder.embed_text.assert_called_with("FILTERED")
        kwargs = main_module.retriever.search_candidates.call_args.kwargs
        self.assertEqual(kwargs["query_text"], "FILTERED")
        self.assertEqual(
            main_module.reranker.rerank.call_args.kwargs["jd"], "ORIGINAL JOB DESCRIPTION"
        )

    def test_a_filter_failure_never_fails_the_search(self):
        main_module.jd_filter = MagicMock()
        main_module.jd_filter.apply.side_effect = RuntimeError("model exploded")
        body = self.screen()

        self.assertEqual(len(body["candidates"]), 1)
        main_module.embedder.embed_text.assert_called_with("ORIGINAL JOB DESCRIPTION")

    def test_no_filter_means_the_original_text_is_used(self):
        main_module.jd_filter = None
        self.screen()
        main_module.embedder.embed_text.assert_called_with("ORIGINAL JOB DESCRIPTION")
        self.assertNotIn("filter_ms", self.screen()["timings"])

    def test_filter_cost_is_reported_when_it_runs(self):
        main_module.jd_filter = MagicMock()
        main_module.jd_filter.apply.return_value = "FILTERED"
        timings = self.screen()["timings"]
        self.assertIn("filter_ms", timings)
        parts = sum(timings[k] for k in ("filter_ms", "embed_ms", "retrieve_ms", "rerank_ms"))
        self.assertAlmostEqual(timings["total_ms"], parts, delta=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
