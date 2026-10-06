"""The rank-correlation helper behind the rerank-gate analysis.

A conclusion of "no usable signal" is only as good as the statistic behind it,
and the first version of this helper was wrong: it broke ties by input position,
which silently distorts the result on data full of ties (several queries have a
delta of exactly zero, and one signal is binary). These values are hand-computed.
"""

import unittest

from evaluation.analyze_rerank_gate import average_ranks, spearman


class AverageRanksTest(unittest.TestCase):
    def test_distinct_values(self):
        self.assertEqual(average_ranks([30, 10, 20]), [3.0, 1.0, 2.0])

    def test_ties_share_the_mean_rank(self):
        # 10 -> 1, the two 20s share ranks 2 and 3, 30 -> 4.
        self.assertEqual(average_ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_all_tied(self):
        self.assertEqual(average_ranks([5, 5, 5]), [2.0, 2.0, 2.0])

    def test_order_of_ties_does_not_matter(self):
        """The property positional tie-breaking violated."""
        a = average_ranks([2, 1, 2, 3])
        b = average_ranks([2, 1, 2, 3][::-1])[::-1]
        self.assertEqual(a, b)


class SpearmanTest(unittest.TestCase):
    def test_perfect_positive(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_negative(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_monotone_nonlinear_is_still_one(self):
        """Rank correlation, not Pearson: any monotone relationship scores 1."""
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 4, 9, 100]), 1.0)

    def test_hand_computed_value_with_ties(self):
        # Ranks x = [1, 2.5, 2.5, 4], ranks y = [1, 2, 3, 4].
        # numerator 4.5, denominator sqrt(4.5 * 5) -> 0.9487.
        self.assertAlmostEqual(
            spearman([1, 2, 2, 3], [1, 2, 3, 4]), 4.5 / (22.5 ** 0.5), places=9
        )

    def test_is_symmetric(self):
        xs, ys = [3, 1, 4, 1, 5], [9, 2, 6, 5, 3]
        self.assertAlmostEqual(spearman(xs, ys), spearman(ys, xs))

    def test_constant_input_gives_zero_not_a_crash(self):
        """A signal that never varies carries no information; dividing by its
        zero variance must not raise."""
        self.assertEqual(spearman([1, 1, 1, 1], [1, 2, 3, 4]), 0.0)

    def test_rejects_mismatched_or_tiny_input(self):
        with self.assertRaises(ValueError):
            spearman([1, 2, 3], [1, 2])
        with self.assertRaises(ValueError):
            spearman([1], [1])

    def test_independent_of_input_order(self):
        xs, ys = [1, 2, 2, 3, 5], [2, 1, 4, 4, 3]
        pairs = list(zip(xs, ys))
        shuffled = [pairs[i] for i in (3, 0, 4, 1, 2)]
        self.assertAlmostEqual(
            spearman(xs, ys),
            spearman([p[0] for p in shuffled], [p[1] for p in shuffled]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
