"""Guards on the dev/test query split.

The split exists because every design decision in this project was originally
made by looking at all 15 queries and then reported on those same 15. A split
only fixes that if it stays airtight, and the ways it can leak are all quiet:
a query added to both lists, a default flipped to `test` for convenience, a
held-out query edited to match something retrieval already does well.

These tests make each of those loud.
"""

import unittest

from evaluation.corpus import (
    ALL_QUERIES,
    CANDIDATES,
    DEV_HARD_QUERIES,
    DEV_QUERIES,
    HARD_QUERIES,
    QUERIES,
    TEST_HARD_QUERIES,
    TEST_QUERIES,
    corpus_stats,
    select_queries,
)


class SplitIntegrityTest(unittest.TestCase):
    def test_dev_and_test_share_no_queries(self):
        """The whole point. An overlap silently restores the leakage."""
        dev = {q["id"] for q in select_queries("dev")}
        test = {q["id"] for q in select_queries("test")}
        self.assertEqual(dev & test, set())
        self.assertEqual(len(dev) + len(test), len(ALL_QUERIES))

    def test_no_query_object_appears_in_both_splits(self):
        """Identity, not just id: the same dict in both lists would be stamped
        with whichever split was applied last."""
        dev_ids = {id(q) for q in DEV_QUERIES + DEV_HARD_QUERIES}
        test_ids = {id(q) for q in TEST_QUERIES + TEST_HARD_QUERIES}
        self.assertEqual(dev_ids & test_ids, set())

    def test_split_label_matches_membership(self):
        for query in DEV_QUERIES + DEV_HARD_QUERIES:
            self.assertEqual(query["split"], "dev", query["id"])
        for query in TEST_QUERIES + TEST_HARD_QUERIES:
            self.assertEqual(query["split"], "test", query["id"])

    def test_legacy_names_still_mean_the_dev_set(self):
        """QUERIES / HARD_QUERIES are referenced by older tests and by the
        historical numbers in the README. If they silently started including
        held-out queries, every previously reported figure would change
        meaning without anything failing."""
        self.assertIs(QUERIES, DEV_QUERIES)
        self.assertIs(HARD_QUERIES, DEV_HARD_QUERIES)
        self.assertEqual(len(QUERIES), 8)
        self.assertEqual(len(HARD_QUERIES), 7)

    def test_default_split_is_dev(self):
        """A `test` default is how a held-out set gets quietly consumed."""
        self.assertEqual(
            [q["id"] for q in select_queries()],
            [q["id"] for q in select_queries("dev", "all")],
        )

    def test_run_eval_cli_defaults_to_dev(self):
        """Same guard, one layer out: the harness a human actually invokes."""
        import io

        from evaluation import run_eval

        # main() builds a real index, so read the flag definition rather than
        # running it. Crude, but it fails if someone changes the default.
        with io.open(run_eval.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('"--split"', source)
        self.assertIn('default="dev"', source)

    def test_kind_partitions_the_split(self):
        for split in ("dev", "test", "all"):
            cross = select_queries(split, "cross")
            within = select_queries(split, "within")
            everything = select_queries(split, "all")
            self.assertEqual(len(cross) + len(within), len(everything))
            self.assertEqual(
                {q["id"] for q in cross} | {q["id"] for q in within},
                {q["id"] for q in everything},
            )

    def test_rejects_unknown_split_and_kind(self):
        with self.assertRaises(ValueError):
            select_queries("holdout")
        with self.assertRaises(ValueError):
            select_queries("dev", "hard")


class TestSetQualityTest(unittest.TestCase):
    """The held-out set has to be big enough and hard enough to be worth
    reading, or it provides false reassurance rather than a check."""

    def test_held_out_set_is_not_token(self):
        self.assertGreaterEqual(len(select_queries("test")), 15)
        self.assertGreaterEqual(len(select_queries("test", "cross")), 6)
        self.assertGreaterEqual(len(select_queries("test", "within")), 6)

    def test_labels_reference_real_candidates_with_valid_grades(self):
        known = {c["id"] for c in CANDIDATES}
        for query in select_queries("test"):
            self.assertTrue(query["relevance"], f"{query['id']} has no labels")
            for cid, grade in query["relevance"].items():
                self.assertIn(cid, known, f"{query['id']} labels unknown {cid}")
                # 0 is expressed by omission; an explicit 0 would be counted as
                # a label by corpus_stats and inflate the distractor count.
                self.assertIn(grade, (1, 2), f"{query['id']}:{cid} grade {grade}")

    def test_every_held_out_query_has_a_strong_answer(self):
        """A query with only partial matches cannot distinguish a good ranking
        from a mediocre one, because there is no correct top result."""
        for query in select_queries("test"):
            self.assertIn(2, query["relevance"].values(), query["id"])

    def test_held_out_queries_face_a_large_distractor_field(self):
        self.assertGreaterEqual(
            corpus_stats("test")["min_distractors_per_query"], 20
        )

    def test_within_role_test_queries_stay_inside_the_python_cluster(self):
        """Their discriminating power depends on the strong answers all being
        plausible Python backend hires. A within-role query whose best answer
        is the React developer is really a cross-role query, and would be
        easy for the wrong reason."""
        cluster = {
            "c01", "c02", "c03", "c04", "c21", "c22", "c23", "c24", "c25",
            "c26", "c27", "c28", "c29", "c30", "c31", "c32",
        }
        for query in select_queries("test", "within"):
            strong = {cid for cid, g in query["relevance"].items() if g == 2}
            self.assertTrue(
                strong <= cluster,
                f"{query['id']} has strong answers outside the cluster: "
                f"{sorted(strong - cluster)}",
            )

    def test_every_candidate_is_relevant_to_something(self):
        """Adding the held-out queries closed the last gaps. A candidate that
        is never the answer to anything is pure corpus weight: it costs index
        size and tells you nothing about ranking."""
        covered = {cid for q in ALL_QUERIES for cid in q["relevance"]}
        self.assertEqual(
            sorted({c["id"] for c in CANDIDATES} - covered),
            [],
        )

    def test_stats_report_both_splits_regardless_of_the_one_requested(self):
        """So a printed report always shows how much was held back."""
        for split in ("dev", "test", "all"):
            stats = corpus_stats(split)
            self.assertEqual(stats["dev_queries"], len(select_queries("dev")))
            self.assertEqual(stats["test_queries"], len(select_queries("test")))
            self.assertEqual(stats["queries"], len(select_queries(split)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
