"""The short-query forms must stay in step with the labelled corpus.

A missing or stale entry would not fail loudly: the harness would silently evaluate
fewer queries, and a headline number would quietly rest on a different set.
"""

import unittest

from evaluation.corpus import ALL_QUERIES, select_queries
from evaluation.short_queries import KEYWORDS, STYLES, TITLES, short_text


class ShortQueriesTest(unittest.TestCase):
    def test_every_query_has_a_keyword_form_and_nothing_else_does(self):
        self.assertEqual(set(KEYWORDS), {q["id"] for q in ALL_QUERIES})

    def test_titles_only_name_real_queries(self):
        self.assertLessEqual(set(TITLES), {q["id"] for q in ALL_QUERIES})

    def test_titles_cover_cross_role_queries_only(self):
        """Within-role labels are narrow on purpose; a bare title would mark the
        other right answers wrong. See the module docstring."""
        within = {q["id"] for q in select_queries("all", "within")}
        self.assertEqual(set(TITLES) & within, set())

    def test_the_python_backend_query_has_no_title_form(self):
        """'python backend engineer' fits sixteen CVs and two are labelled strong."""
        self.assertNotIn("q1_python_backend", TITLES)

    def test_they_really_are_short(self):
        for qid, text in KEYWORDS.items():
            self.assertLessEqual(len(text.split()), 11, qid)
            self.assertGreaterEqual(len(text.split()), 3, qid)
        for qid, text in TITLES.items():
            self.assertLessEqual(len(text.split()), 5, qid)
            self.assertGreaterEqual(len(text.split()), 2, qid)

    def test_no_entry_is_just_the_job_description(self):
        by_id = {q["id"]: q["job_description"] for q in ALL_QUERIES}
        for qid, text in KEYWORDS.items():
            self.assertNotEqual(text, by_id[qid], qid)
            self.assertLess(len(text), len(by_id[qid]) / 2, qid)

    def test_no_blank_or_duplicate_phrases(self):
        values = list(KEYWORDS.values())
        self.assertTrue(all(v.strip() for v in values))
        self.assertEqual(len(values), len(set(values)))

    def test_lookup(self):
        self.assertEqual(short_text("q9_go_backend", "keywords"), "go grpc services")
        self.assertEqual(short_text("q5_data_engineer", "title"), "data engineer")
        self.assertIsNone(short_text("h1_vector_search", "title"))
        self.assertIsNone(short_text("no_such_query", "keywords"))

    def test_jd_style_is_not_a_lookup(self):
        with self.assertRaises(ValueError):
            short_text("q1_python_backend", "jd")
        with self.assertRaises(ValueError):
            short_text("q1_python_backend", "bogus")

    def test_the_styles_the_harness_offers_match(self):
        self.assertEqual(STYLES, ("jd", "title", "keywords"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
