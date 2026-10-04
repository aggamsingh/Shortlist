"""Reranker robustness tests.

The reranker consumes free-form LLM output, which is the least trustworthy input
in the system: models omit candidates, repeat ids, invent ids, answer on the
wrong numeric scale, and occasionally return prose instead of JSON. None of that
may corrupt a response or crash a request.
"""

import os
import unittest
from unittest.mock import patch

from api.reranker import (  # noqa: E402
    CVReranker,
    apply_scale,
    detect_score_scale,
    parse_score,
    truncate_text,
)


def make_candidates():
    return [
        {"candidate_id": "c1", "name": "Alice", "cv_path": "/a.pdf",
         "score": 0.80, "resume_summary": "python fastapi"},
        {"candidate_id": "c2", "name": "Bob", "cv_path": "/b.pdf",
         "score": 0.60, "resume_summary": "java spring"},
        {"candidate_id": "c3", "name": "Carol", "cv_path": "/c.pdf",
         "score": 0.40, "resume_summary": "react"},
    ]


class ParseScoreTest(unittest.TestCase):
    def test_float_and_string_forms(self):
        self.assertEqual(parse_score(0.9), 0.9)
        self.assertEqual(parse_score("0.85"), 0.85)
        self.assertEqual(parse_score(95), 95.0)

    def test_unusable_values_return_none(self):
        for value in (None, "abc", {}, [], True, False, float("nan"), float("inf")):
            self.assertIsNone(parse_score(value), f"{value!r} should be unusable")


class ScoreScaleTest(unittest.TestCase):
    def test_zero_to_one_scale(self):
        self.assertEqual(detect_score_scale([0.9, 0.7, 0.3]), 1.0)

    def test_zero_to_hundred_scale(self):
        self.assertEqual(detect_score_scale([95, 70, 30]), 100.0)

    def test_zero_to_ten_scale(self):
        self.assertEqual(detect_score_scale([9, 7, 3]), 10.0)

    def test_slight_overshoot_is_clamped_not_rescaled(self):
        """A lone 1.5 among 0.x scores is over-confidence, not a 0-10 answer.

        Rescaling it in isolation would map 1.5 -> 0.015 and rank the model's
        best pick last, which is the opposite of what it said.
        """
        self.assertEqual(detect_score_scale([1.5, 0.9, 0.3]), 1.0)
        self.assertEqual(apply_scale(1.5, 1.0, 0.0), 1.0)

    def test_empty_and_unusable_batches(self):
        self.assertEqual(detect_score_scale([]), 1.0)
        self.assertEqual(detect_score_scale([None, None]), 1.0)

    def test_relative_ordering_always_survives_normalisation(self):
        """Normalisation may rescale, but must never reorder."""
        for batch in ([0.9, 0.7, 0.3], [95, 70, 30], [9, 7, 3], [1.5, 0.9, 0.3], [100, 50, 1]):
            scale = detect_score_scale(batch)
            normalised = [apply_scale(v, scale, 0.0) for v in batch]
            raw_order = sorted(range(len(batch)), key=lambda i: -batch[i])
            new_order = sorted(range(len(batch)), key=lambda i: -normalised[i])
            self.assertEqual(raw_order, new_order, f"reordered: {batch}")

    def test_values_are_clamped_into_unit_range(self):
        self.assertEqual(apply_scale(-3, 1.0, 0.0), 0.0)
        self.assertEqual(apply_scale(5.0, 1.0, 0.0), 1.0)

    def test_fallback_used_when_value_missing(self):
        self.assertEqual(apply_scale(None, 1.0, 0.42), 0.42)


class TruncateTest(unittest.TestCase):
    def test_long_text_is_cut_and_marked(self):
        text = " ".join(f"word{i}" for i in range(400))
        out = truncate_text(text, 100)
        self.assertLess(len(out), len(text))
        self.assertTrue(out.endswith("[truncated]"))

    def test_cut_falls_on_a_word_boundary(self):
        text = " ".join(f"word{i}" for i in range(400))
        body = truncate_text(text, 100).replace(" [truncated]", "")
        self.assertFalse(body.endswith("wor"))
        self.assertTrue(all(w.startswith("word") for w in body.split()))

    def test_short_text_passes_through(self):
        self.assertEqual(truncate_text("hello", 100), "hello")

    def test_none_becomes_empty_string(self):
        self.assertEqual(truncate_text(None, 100), "")


class ParseRankingsTest(unittest.TestCase):
    def test_documented_shape(self):
        out = CVReranker._parse_rankings('{"rankings":[{"candidate_id":"a"}]}', "T")
        self.assertEqual(out, [{"candidate_id": "a"}])

    def test_bare_list_accepted(self):
        self.assertEqual(CVReranker._parse_rankings('[{"candidate_id":"a"}]', "T"),
                         [{"candidate_id": "a"}])

    def test_alternative_key_name_accepted(self):
        self.assertEqual(CVReranker._parse_rankings('{"results":[{"candidate_id":"a"}]}', "T"),
                         [{"candidate_id": "a"}])

    def test_non_dict_entries_are_dropped(self):
        out = CVReranker._parse_rankings('{"rankings":[{"candidate_id":"a"},"junk",5,null]}', "T")
        self.assertEqual(out, [{"candidate_id": "a"}])

    def test_malformed_bodies_raise(self):
        for body in ("not json", '{"rankings":"nope"}', "42", None):
            with self.assertRaises(RuntimeError):
                CVReranker._parse_rankings(body, "T")


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.reranker = CVReranker()

    def test_hallucinated_candidate_never_reaches_output(self):
        rankings = [{"candidate_id": "GHOST", "score": 0.99, "match_reasoning": "x"}]
        out = self.reranker._merge(make_candidates(), rankings, top_k=10)
        self.assertNotIn("GHOST", [c["candidate_id"] for c in out])

    def test_duplicate_ids_take_first_judgement(self):
        rankings = [
            {"candidate_id": "c1", "score": 0.95, "match_reasoning": "first"},
            {"candidate_id": "c1", "score": 0.10, "match_reasoning": "second"},
        ]
        out = self.reranker._merge(make_candidates(), rankings, top_k=10)
        alice = next(c for c in out if c["candidate_id"] == "c1")
        self.assertEqual(alice["match_reasoning"], "first")

    def test_omitted_candidates_fall_back_to_vector_score(self):
        rankings = [{"candidate_id": "c1", "score": 0.95, "match_reasoning": "good"}]
        out = self.reranker._merge(make_candidates(), rankings, top_k=10)
        bob = next(c for c in out if c["candidate_id"] == "c2")
        self.assertEqual(bob["score"], 0.60)
        self.assertIn("not scored", bob["match_reasoning"])

    def test_judged_candidates_rank_above_fallbacks(self):
        """LLM scores and cosine similarities are different units.

        Interleaving them by raw value would let an unjudged 0.6 cosine outrank a
        judged 0.55, so judged candidates are ordered as a block.
        """
        rankings = [{"candidate_id": "c3", "score": 0.55, "match_reasoning": "ok"}]
        out = self.reranker._merge(make_candidates(), rankings, top_k=10)
        self.assertEqual(out[0]["candidate_id"], "c3")

    def test_overshooting_top_pick_stays_first(self):
        rankings = [
            {"candidate_id": "c1", "score": 1.5, "match_reasoning": "great"},
            {"candidate_id": "c2", "score": 0.4, "match_reasoning": "ok"},
        ]
        out = self.reranker._merge(make_candidates(), rankings, top_k=10)
        self.assertEqual(out[0]["candidate_id"], "c1")
        self.assertLessEqual(out[0]["score"], 1.0)

    def test_internal_fields_do_not_leak(self):
        out = self.reranker._merge(make_candidates(), [], top_k=10)
        for row in out:
            self.assertNotIn("_judged", row)
            self.assertEqual(
                set(row), {"candidate_id", "name", "score", "match_reasoning", "cv_path"}
            )

    def test_total_failure_preserves_vector_order(self):
        out = self.reranker._merge(make_candidates(), [], top_k=10)
        self.assertEqual([c["name"] for c in out], ["Alice", "Bob", "Carol"])

    def test_top_k_truncates(self):
        out = self.reranker._merge(make_candidates(), [], top_k=2)
        self.assertEqual(len(out), 2)

    def test_scores_always_within_unit_range(self):
        rankings = [
            {"candidate_id": "c1", "score": 500, "match_reasoning": "x"},
            {"candidate_id": "c2", "score": -5, "match_reasoning": "y"},
        ]
        for row in self.reranker._merge(make_candidates(), rankings, top_k=10):
            self.assertGreaterEqual(row["score"], 0.0)
            self.assertLessEqual(row["score"], 1.0)

    def test_non_string_reasoning_is_coerced(self):
        rankings = [{"candidate_id": "c1", "score": 0.9, "match_reasoning": 12345}]
        out = self.reranker._merge(make_candidates(), rankings, top_k=10)
        self.assertIsInstance(out[0]["match_reasoning"], str)


class RequestSizeTest(unittest.TestCase):
    """Oversized requests must be split, not dropped and not retried.

    Found running 32 real CVs through the service: 30 candidates at 1200 chars
    built a ~8300-token prompt against Groq's 8000 TPM cap. The provider rejects
    that outright (HTTP 413), and retrying cannot help because the request is
    deterministically too large.
    """

    def _candidates(self, n, chars=1200):
        return [{"candidate_id": f"c{i}", "name": f"P{i}", "cv_path": "",
                 "score": 0.5, "resume_summary": "x " * (chars // 2)}
                for i in range(n)]

    def test_large_shortlist_is_split_into_batches(self):
        reranker = CVReranker()
        reranker.token_budget = 6000
        batches = reranker._split_to_fit("jd", self._candidates(30))
        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(len(b) for b in batches), 30, "candidates were dropped")

    def test_every_batch_fits_the_budget(self):
        reranker = CVReranker()
        reranker.token_budget = 6000
        for batch in reranker._split_to_fit("jd", self._candidates(30)):
            estimated = len(reranker._build_prompt("jd", batch)) // 4
            self.assertLessEqual(estimated, reranker.token_budget)

    def test_small_shortlist_stays_one_batch(self):
        reranker = CVReranker()
        reranker.token_budget = 20000
        self.assertEqual(len(reranker._split_to_fit("jd", self._candidates(3))), 1)

    def test_edge_cases(self):
        reranker = CVReranker()
        self.assertEqual(reranker._split_to_fit("jd", []), [])
        self.assertEqual(len(reranker._split_to_fit("jd", self._candidates(1))), 1)

    def test_one_failing_batch_does_not_lose_the_others(self):
        """A batch that errors must not discard judgements already obtained."""
        reranker = CVReranker()
        reranker.gemini_key = "stub"
        reranker.token_budget = 2000
        calls = {"n": 0}

        def flaky(jd, batch):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("413 request too large")
            return [{"candidate_id": c["candidate_id"], "score": 0.9,
                     "match_reasoning": "judged"} for c in batch]

        reranker._rerank_with_gemini = flaky
        out = reranker.rerank("jd", self._candidates(10), top_k=10)
        judged = [c for c in out if c["match_reasoning"] == "judged"]
        self.assertGreater(len(judged), 0, "surviving batches lost their scores")
        self.assertEqual(len(out), 10, "candidates disappeared")

    def test_oversized_request_is_not_retried(self):
        """413 is deterministic; retrying burns backoff and still fails."""
        from indexer.llm import retry_delay

        for message in ("Error code: 413 - Request too large for model",
                        "please reduce your message size and try again"):
            self.assertEqual(retry_delay(Exception(message), 0), 0.0, message)

    def test_per_minute_rate_limit_is_still_retried(self):
        from indexer.llm import retry_delay

        self.assertGreater(
            retry_delay(Exception("429 rate limit reached, try again in 3.5s"), 0), 0
        )


class RerankWithoutProviderTest(unittest.TestCase):
    """Behaviour when no usable provider is configured.

    These pin the environment explicitly rather than relying on the ambient
    process env. A developer with a real key in .env would otherwise see these
    fail, because anything that calls load_dotenv() earlier in the run leaks
    that key into os.environ for every test that follows.
    """

    @staticmethod
    def _no_keys(**overrides):
        env = {"GEMINI_API_KEY": "", "GROQ_API_KEY": ""}
        env.update(overrides)
        return patch.dict(os.environ, env, clear=False)

    def test_unconfigured_reranker_degrades_instead_of_raising(self):
        with self._no_keys():
            reranker = CVReranker()
            self.assertFalse(reranker.is_configured)
            out = reranker.rerank("some jd", make_candidates(), top_k=2)
        self.assertEqual([c["name"] for c in out], ["Alice", "Bob"])

    def test_empty_candidate_list(self):
        with self._no_keys():
            self.assertEqual(CVReranker().rerank("jd", [], 5), [])

    def test_placeholder_keys_are_treated_as_unconfigured(self):
        with self._no_keys(GEMINI_API_KEY="your_gemini_api_key_here"):
            self.assertFalse(CVReranker().is_configured)

    def test_provider_failure_degrades_to_vector_order(self):
        """A provider outage must return ranked results, not a 500."""
        reranker = CVReranker()
        reranker.gemini_key = "real-looking-key"
        reranker._rerank_with_gemini = lambda jd, c: (_ for _ in ()).throw(
            RuntimeError("provider down")
        )
        out = reranker.rerank("jd", make_candidates(), top_k=3)
        self.assertEqual([c["name"] for c in out], ["Alice", "Bob", "Carol"])

    def test_prompt_respects_character_budget(self):
        reranker = CVReranker()
        reranker.max_chars = 200
        huge = [{"candidate_id": "c1", "name": "A", "cv_path": "",
                 "score": 0.5, "resume_summary": "word " * 5000}]
        prompt = reranker._build_prompt("jd", huge)
        self.assertIn("[truncated]", prompt)
        self.assertLess(len(prompt), 3000)

    def test_candidate_count_is_capped(self):
        reranker = CVReranker()
        reranker.max_candidates = 2
        many = [{"candidate_id": f"c{i}", "name": f"N{i}", "cv_path": "",
                 "score": 1 - i / 100, "resume_summary": "x"} for i in range(50)]
        out = reranker.rerank("jd", many, top_k=50)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
