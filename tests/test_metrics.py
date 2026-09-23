"""Unit tests for ranking metrics, checked against hand-computed values.

Every quality claim in the README rests on these functions, so they are verified
against numbers worked out by hand rather than against their own output.
"""

import math
import unittest

from evaluation.metrics import (
    aggregate,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# a and d are strong (grade 2), b is partial (1), c is irrelevant (0).
RELEVANCE = {"a": 2, "b": 1, "c": 0, "d": 2}


class RecallTest(unittest.TestCase):
    def test_all_relevant_in_top_k(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "d", "c"], RELEVANCE, 3), 1.0)

    def test_partial_recall(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c", "d"], RELEVANCE, 2), 2 / 3)

    def test_none_in_top_k(self):
        self.assertAlmostEqual(recall_at_k(["c", "a", "b", "d"], RELEVANCE, 1), 0.0)

    def test_no_relevant_documents_scores_zero(self):
        self.assertAlmostEqual(recall_at_k(["a"], {"a": 0}, 3), 0.0)

    def test_grade_threshold_excludes_partial_matches(self):
        # threshold=2 counts only strong matches: a and d.
        self.assertAlmostEqual(recall_at_k(["a", "b"], RELEVANCE, 2, threshold=2), 0.5)


class PrecisionTest(unittest.TestCase):
    def test_all_top_k_relevant(self):
        self.assertAlmostEqual(precision_at_k(["a", "b", "c"], RELEVANCE, 2), 1.0)

    def test_one_irrelevant_in_top_k(self):
        self.assertAlmostEqual(precision_at_k(["a", "c", "b"], RELEVANCE, 3), 2 / 3)

    def test_zero_k_does_not_divide_by_zero(self):
        self.assertAlmostEqual(precision_at_k(["a"], RELEVANCE, 0), 0.0)

    def test_k_larger_than_result_list(self):
        self.assertAlmostEqual(precision_at_k(["a"], RELEVANCE, 4), 0.25)


class ReciprocalRankTest(unittest.TestCase):
    def test_first_result_relevant(self):
        self.assertAlmostEqual(reciprocal_rank(["a", "c"], RELEVANCE), 1.0)

    def test_second_result_relevant(self):
        self.assertAlmostEqual(reciprocal_rank(["c", "a"], RELEVANCE), 0.5)

    def test_third_result_relevant(self):
        self.assertAlmostEqual(reciprocal_rank(["c", "z", "b"], RELEVANCE), 1 / 3)

    def test_nothing_relevant(self):
        self.assertAlmostEqual(reciprocal_rank(["c"], RELEVANCE), 0.0)


class NdcgTest(unittest.TestCase):
    def test_dcg_matches_hand_calculation(self):
        # [a(2), c(0), b(1)] -> (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4) = 3 + 0.5
        self.assertAlmostEqual(dcg_at_k(["a", "c", "b"], RELEVANCE, 3), 3.5)

    def test_ideal_dcg_matches_hand_calculation(self):
        # ideal [a(2), d(2), b(1)] -> 3/1 + 3/log2(3) + 1/2
        expected = 3 + 3 / math.log2(3) + 0.5
        self.assertAlmostEqual(dcg_at_k(["a", "d", "b", "c"], RELEVANCE, 3), expected)

    def test_ndcg_ratio(self):
        expected = 3.5 / (3 + 3 / math.log2(3) + 0.5)
        self.assertAlmostEqual(ndcg_at_k(["a", "c", "b"], RELEVANCE, 3), expected)

    def test_perfect_ranking_scores_one(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "d", "b", "c"], RELEVANCE, 3), 1.0)

    def test_all_irrelevant_scores_zero(self):
        self.assertAlmostEqual(ndcg_at_k(["c"], RELEVANCE, 3), 0.0)

    def test_empty_relevance_map_scores_zero(self):
        self.assertAlmostEqual(ndcg_at_k(["a"], {}, 3), 0.0)

    def test_better_ordering_scores_higher(self):
        """The metric has to actually reward a better ranking."""
        good = ndcg_at_k(["a", "d", "b"], RELEVANCE, 3)
        bad = ndcg_at_k(["b", "c", "a"], RELEVANCE, 3)
        self.assertGreater(good, bad)

    def test_grade_two_outranks_grade_one(self):
        strong_first = ndcg_at_k(["a", "b"], RELEVANCE, 2)
        partial_first = ndcg_at_k(["b", "a"], RELEVANCE, 2)
        self.assertGreater(strong_first, partial_first)


class AggregateTest(unittest.TestCase):
    def test_macro_average(self):
        self.assertAlmostEqual(aggregate([{"m": 1.0}, {"m": 0.0}])["m"], 0.5)

    def test_empty_input(self):
        self.assertEqual(aggregate([]), {})


class CorpusIntegrityTest(unittest.TestCase):
    """The benchmark is only meaningful if its fixtures are self-consistent."""

    def test_candidate_ids_unique(self):
        from evaluation.corpus import CANDIDATES

        ids = [c["id"] for c in CANDIDATES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_labels_reference_real_candidates(self):
        from evaluation.corpus import ALL_QUERIES, CANDIDATES

        known = {c["id"] for c in CANDIDATES}
        for query in ALL_QUERIES:
            for cid in query["relevance"]:
                self.assertIn(cid, known, f"{query['id']} labels unknown candidate {cid}")

    def test_query_ids_unique(self):
        from evaluation.corpus import ALL_QUERIES

        ids = [q["id"] for q in ALL_QUERIES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_rendered_cv_metadata_matches_fixture(self):
        """Parsed years/location must agree with the declared fixture values."""
        from evaluation.corpus import CANDIDATES, render_cv
        from indexer.parser import extract_years_of_experience
        from indexer.run import extract_location

        for candidate in CANDIDATES:
            text = render_cv(candidate)
            self.assertEqual(
                extract_years_of_experience(text), candidate["years"], candidate["id"]
            )
            self.assertEqual(extract_location(text), candidate["location"], candidate["id"])

    def test_every_query_faces_a_large_distractor_field(self):
        """Distractors are what stop keyword matching from scoring well.

        Measured per query: a query with only a handful of non-relevant
        candidates is trivially easy regardless of the retrieval strategy.
        """
        from evaluation.corpus import corpus_stats

        self.assertGreaterEqual(corpus_stats()["min_distractors_per_query"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
