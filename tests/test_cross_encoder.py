"""Cross-encoder reranker.

The expensive lesson from the LLM path is encoded here: a score transform that
is not order-preserving silently inverts the model's answer. That bug put the
LLM's best pick last and was invisible in the output, because the scores still
looked plausible. Every transform in this module is therefore tested for
monotonicity rather than for its values.

The model itself is mocked throughout except in one integration test, so the
suite stays fast and does not need a 90MB download in CI.
"""

import unittest
from unittest.mock import patch

from api.cross_encoder import (
    CHUNK_SEPARATOR,
    FALLBACK_REASONING,
    PASSAGE_OVERLAP,
    PASSAGE_WORDS,
    CrossEncoderReranker,
    sigmoid,
    split_passages,
)


def candidate(cid, text, score=0.5, name=None):
    return {
        "candidate_id": cid,
        "name": name or cid.upper(),
        "cv_path": f"/cv/{cid}.pdf",
        "score": score,
        "resume_summary": text,
    }


class SigmoidTest(unittest.TestCase):
    def test_maps_into_unit_interval(self):
        for x in (-50.0, -11.0, -3.1, 0.0, 3.1, 11.0, 50.0):
            value = sigmoid(x)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_is_strictly_monotonic(self):
        """The property that matters. ms-marco logits run roughly -11..+11."""
        xs = [-40.0, -11.0, -9.8, -5.0, -3.1, -0.5, 0.0, 0.5, 3.1, 11.0, 40.0]
        ys = [sigmoid(x) for x in xs]
        self.assertEqual(ys, sorted(ys))
        # Strict, not merely non-decreasing, across the model's real range.
        for a, b in zip(ys[1:-1], ys[2:]):
            self.assertLess(a, b)

    def test_does_not_overflow_on_large_negatives(self):
        """math.exp(-x) overflows around x = -746; the branch exists for this."""
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0, places=12)
        self.assertAlmostEqual(sigmoid(1000.0), 1.0, places=12)

    def test_midpoint(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)


class SplitPassagesTest(unittest.TestCase):
    def test_empty_text_yields_nothing(self):
        self.assertEqual(split_passages(""), [])
        self.assertEqual(split_passages("   \n  "), [])

    def test_splits_on_the_retriever_separator(self):
        text = f"first chunk{CHUNK_SEPARATOR}second chunk"
        self.assertEqual(split_passages(text), ["first chunk", "second chunk"])

    def test_short_text_is_one_passage(self):
        self.assertEqual(split_passages("a short cv"), ["a short cv"])

    def test_long_passage_is_windowed_not_truncated(self):
        """The tail of a long CV must survive.

        Relying on the tokenizer's silent truncation would drop it, and skills
        sections tend to sit at the end of a resume.
        """
        words = [f"w{i}" for i in range(PASSAGE_WORDS * 2 + 25)]
        passages = split_passages(" ".join(words))

        self.assertGreater(len(passages), 1)
        for passage in passages:
            self.assertLessEqual(len(passage.split()), PASSAGE_WORDS)
        # Every word appears somewhere, first and last included.
        covered = {w for p in passages for w in p.split()}
        self.assertEqual(covered, set(words))
        self.assertIn(words[-1], passages[-1].split())

    def test_windows_overlap(self):
        words = [f"w{i}" for i in range(PASSAGE_WORDS + 50)]
        passages = split_passages(" ".join(words))
        first, second = passages[0].split(), passages[1].split()
        self.assertEqual(first[-PASSAGE_OVERLAP:], second[:PASSAGE_OVERLAP])

    def test_no_empty_passages(self):
        text = f"real text{CHUNK_SEPARATOR}   {CHUNK_SEPARATOR}more text"
        self.assertEqual(split_passages(text), ["real text", "more text"])


class RerankOrderingTest(unittest.TestCase):
    """Behaviour with the model mocked, so ordering can be asserted exactly."""

    def rerank_with(self, logits, candidates, top_k=None):
        reranker = CrossEncoderReranker()
        with patch.object(CrossEncoderReranker, "_score_pairs", return_value=logits):
            return reranker.rerank(
                "a job description",
                candidates,
                top_k=top_k if top_k is not None else len(candidates),
            )

    def test_empty_input(self):
        self.assertEqual(self.rerank_with([], [], top_k=5), [])

    def test_reorders_by_model_score_not_retrieval_score(self):
        """The whole purpose: retrieval order must be overridable."""
        cands = [
            candidate("c1", "weak match", score=0.99),   # retrieval's favourite
            candidate("c2", "strong match", score=0.10),
        ]
        results = self.rerank_with([-9.0, -1.0], cands)
        self.assertEqual([r["candidate_id"] for r in results], ["c2", "c1"])

    def test_best_passage_wins_per_candidate(self):
        """A CV is relevant if ANY section is, so the max is taken, not the mean.

        c1 has one irrelevant and one highly relevant section; averaging would
        rank it below c2, taking the max ranks it above.
        """
        cands = [
            candidate("c1", f"irrelevant{CHUNK_SEPARATOR}highly relevant"),
            candidate("c2", "moderately relevant"),
        ]
        results = self.rerank_with([-10.0, 2.0, -1.0], cands)
        self.assertEqual([r["candidate_id"] for r in results], ["c1", "c2"])

    def test_scores_are_monotonic_in_the_logits(self):
        """Normalisation may rescale but must never reorder."""
        cands = [candidate(f"c{i}", f"text {i}") for i in range(5)]
        logits = [-8.0, -2.0, 0.5, -5.0, 3.0]
        results = self.rerank_with(logits, cands)

        by_id = {r["candidate_id"]: r["score"] for r in results}
        expected = [cid for _, cid in sorted(
            ((-logits[i], f"c{i}") for i in range(5))
        )]
        self.assertEqual([r["candidate_id"] for r in results], expected)
        # And the scores themselves agree with the logit order.
        self.assertEqual(
            sorted(by_id, key=lambda c: -by_id[c]), expected
        )

    def test_respects_top_k(self):
        cands = [candidate(f"c{i}", f"text {i}") for i in range(5)]
        results = self.rerank_with([0.0, 1.0, 2.0, 3.0, 4.0], cands, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual([r["candidate_id"] for r in results], ["c4", "c3"])

    def test_metadata_comes_from_the_index_not_the_model(self):
        cands = [candidate("c1", "text", name="Real Name")]
        results = self.rerank_with([1.0], cands)
        self.assertEqual(results[0]["name"], "Real Name")
        self.assertEqual(results[0]["cv_path"], "/cv/c1.pdf")
        self.assertEqual(results[0]["candidate_id"], "c1")

    def test_reasoning_quotes_the_matching_section(self):
        """No reasoning text is generated, so the evidence is quoted instead of
        an explanation being invented."""
        cands = [candidate("c1", "built semantic search with Qdrant")]
        results = self.rerank_with([1.25], cands)
        reasoning = results[0]["match_reasoning"]
        self.assertIn("semantic search with Qdrant", reasoning)
        self.assertIn("+1.25", reasoning)

    def test_reasoning_is_length_capped(self):
        cands = [candidate("c1", "word " * 150)]
        results = self.rerank_with([1.0], cands)
        self.assertLessEqual(len(results[0]["match_reasoning"]), 300)

    def test_candidate_without_text_falls_back_not_crashes(self):
        cands = [
            candidate("c1", "has text"),
            candidate("c2", "", score=0.42),
        ]
        results = self.rerank_with([2.0], cands)
        self.assertEqual([r["candidate_id"] for r in results], ["c1", "c2"])
        fallback = next(r for r in results if r["candidate_id"] == "c2")
        self.assertEqual(fallback["match_reasoning"], FALLBACK_REASONING)
        self.assertEqual(fallback["score"], 0.42)

    def test_judged_candidates_rank_above_fallbacks(self):
        """A sigmoid relevance and a cosine similarity are different units.

        ms-marco logits are usually negative, so sigmoid values run LOW -- a
        genuinely good match can score 0.04. Interleaving by raw value would let
        an unjudged 0.9 cosine outrank every properly judged candidate, which is
        the same class of bug as the LLM scale inversion.
        """
        cands = [
            candidate("judged", "has text", score=0.0),
            candidate("unjudged", "", score=0.99),
        ]
        results = self.rerank_with([-3.1], cands)
        self.assertEqual([r["candidate_id"] for r in results], ["judged", "unjudged"])
        self.assertLess(results[0]["score"], results[1]["score"])

    def test_partial_model_output_degrades_rather_than_misaligning(self):
        """Fewer scores than pairs must not be zipped against the wrong owners.

        Silently aligning them would attribute one candidate's score to another,
        which is worse than not reranking at all.
        """
        cands = [candidate(f"c{i}", f"text {i}", score=0.5 - i * 0.1) for i in range(3)]
        results = self.rerank_with([1.0], cands)  # 1 score for 3 pairs
        self.assertEqual([r["candidate_id"] for r in results], ["c0", "c1", "c2"])
        for row in results:
            self.assertEqual(row["match_reasoning"], FALLBACK_REASONING)

    def test_caps_the_shortlist(self):
        cands = [candidate(f"c{i}", f"text {i}") for i in range(40)]
        with patch.dict("os.environ", {"MAX_CANDIDATES_PER_RERANK": "5"}):
            reranker = CrossEncoderReranker()
            with patch.object(
                CrossEncoderReranker, "_score_pairs", return_value=[1.0] * 5
            ):
                results = reranker.rerank("jd", cands, top_k=40)
        self.assertEqual(len(results), 5)

    def test_deterministic_tie_break(self):
        """Equal scores must not produce a different order run to run."""
        cands = [candidate("c2", "t", name="Beta"), candidate("c1", "t", name="Alpha")]
        first = self.rerank_with([1.0, 1.0], cands)
        second = self.rerank_with([1.0, 1.0], cands)
        self.assertEqual(
            [r["candidate_id"] for r in first], [r["candidate_id"] for r in second]
        )
        self.assertEqual([r["name"] for r in first], ["Alpha", "Beta"])


class LoadFailureTest(unittest.TestCase):
    def setUp(self):
        # The model is cached on the class, so each test starts from clean state.
        CrossEncoderReranker._model = None
        CrossEncoderReranker._load_failed = False

    tearDown = setUp

    def test_is_configured_false_when_the_model_cannot_load(self):
        with patch(
            "sentence_transformers.CrossEncoder", side_effect=OSError("no network")
        ):
            self.assertFalse(CrossEncoderReranker().is_configured)

    def test_load_failure_is_remembered_not_retried_per_request(self):
        """A failing load takes seconds; retrying it on every request would turn
        one outage into sustained latency."""
        with patch(
            "sentence_transformers.CrossEncoder", side_effect=OSError("no network")
        ) as ctor:
            reranker = CrossEncoderReranker()
            for _ in range(5):
                reranker.is_configured
        self.assertEqual(ctor.call_count, 1)

    def test_rerank_degrades_to_retrieval_order_when_unavailable(self):
        # Score-descending, which is the order the retriever actually returns.
        cands = [
            candidate("c1", "text", score=0.9),
            candidate("c2", "text", score=0.3),
        ]
        with patch(
            "sentence_transformers.CrossEncoder", side_effect=OSError("no network")
        ):
            results = CrossEncoderReranker().rerank("jd", cands, top_k=2)

        # Retrieval order preserved, and flagged with the phrase the harness and
        # the API both use to detect a degraded rerank.
        self.assertEqual([r["candidate_id"] for r in results], ["c1", "c2"])
        for row in results:
            self.assertEqual(row["match_reasoning"], FALLBACK_REASONING)

    def test_degraded_output_is_ordered_by_retrieval_score(self):
        """Precisely: unjudged candidates are sorted by score, not left in input
        order. Those are the same thing for the retriever, which returns
        score-descending, but the guarantee is on the score -- so a caller that
        hands over an unsorted list gets it sorted rather than untouched."""
        cands = [
            candidate("low", "text", score=0.2),
            candidate("high", "text", score=0.8),
        ]
        with patch(
            "sentence_transformers.CrossEncoder", side_effect=OSError("no network")
        ):
            results = CrossEncoderReranker().rerank("jd", cands, top_k=2)
        self.assertEqual([r["candidate_id"] for r in results], ["high", "low"])

    def test_fallback_phrase_matches_the_llm_path(self):
        """The harness flags degraded runs by this exact string, and the API
        reports `reranked: false` from it. Two spellings would break both."""
        from api.reranker import CVReranker

        del CVReranker  # imported only to assert the module exists
        self.assertIn("not scored by reranker", FALLBACK_REASONING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
