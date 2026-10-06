"""Keyword-stuffing defence: unit behaviour, retrieval integration, and the response.

The attack is simple and was devastating before the defence: an applicant pastes
the job description into their CV, and one irrelevant candidate per query ranked
first in 37 of 37 dev queries, through retrieval and through the cross-encoder
alike. Every integration test here has a control that runs the same attack with the
defence switched off and asserts that it WORKS, so a passing test cannot just mean
the attack was never effective in the fixture.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

import api.main as main_module
from api import stuffing
from api.retriever import CVRetriever

JD = (
    "We are hiring a senior Python backend engineer with strong experience building "
    "FastAPI microservices, containerised with Docker, backed by PostgreSQL and a "
    "vector database for semantic search over large document collections."
)
LEGIT_CV = (
    "Ananya Rao Senior Backend Engineer. Seven years designing Python services. "
    "Led a team of five building FastAPI APIs that handle twelve thousand requests "
    "per second and migrated a monolith to Kubernetes."
)


class ShinglesAndOverlapTest(unittest.TestCase):
    def test_shingles_are_case_and_punctuation_insensitive(self):
        self.assertEqual(
            stuffing.shingles("Python, FastAPI; Docker! and Redis"),
            stuffing.shingles("python fastapi docker and redis"),
        )

    def test_a_text_shorter_than_four_words_has_no_shingles(self):
        self.assertEqual(stuffing.shingles("one two three"), frozenset())

    def test_a_verbatim_copy_overlaps_completely(self):
        q = stuffing.shingles(JD)
        fraction, shared = stuffing.overlap(q, JD)
        self.assertEqual(fraction, 1.0)
        self.assertEqual(shared, len(q))

    def test_an_honest_cv_barely_overlaps(self):
        fraction, shared = stuffing.overlap(stuffing.shingles(JD), LEGIT_CV)
        self.assertLess(fraction, 0.12)
        self.assertLessEqual(shared, 2)

    def test_empty_query_overlaps_nothing(self):
        self.assertEqual(stuffing.overlap(frozenset(), LEGIT_CV), (0.0, 0))


class IsCopyTest(unittest.TestCase):
    def test_verbatim_copy_is_detected(self):
        self.assertTrue(stuffing.is_copy(stuffing.shingles(JD), JD))

    def test_copy_with_surrounding_text_is_detected(self):
        self.assertTrue(stuffing.is_copy(stuffing.shingles(JD), LEGIT_CV + " " + JD + " Thanks."))

    def test_copy_with_changed_case_and_punctuation_is_detected(self):
        self.assertTrue(
            stuffing.is_copy(stuffing.shingles(JD), JD.upper().replace(",", " ;"))
        )

    def test_honest_cv_is_not_a_copy(self):
        self.assertFalse(stuffing.is_copy(stuffing.shingles(JD), LEGIT_CV))

    def test_a_partial_copy_over_the_threshold_is_detected(self):
        half = " ".join(JD.split()[: len(JD.split()) * 6 // 10])
        self.assertTrue(stuffing.is_copy(stuffing.shingles(JD), half))

    def test_a_small_overlap_is_not_a_copy(self):
        quarter = " ".join(JD.split()[: len(JD.split()) // 5])
        self.assertFalse(stuffing.is_copy(stuffing.shingles(JD), quarter))

    def test_a_very_short_job_description_cannot_be_judged(self):
        """A legitimate CV can share half of a five-word query by accident, so
        detection is off rather than guessing."""
        short = "Python backend engineer with FastAPI"
        self.assertFalse(stuffing.is_copy(stuffing.shingles(short), short))

    def test_the_threshold_is_configurable(self):
        half = " ".join(JD.split()[: len(JD.split()) * 6 // 10])
        with patch.dict(os.environ, {"STUFFING_OVERLAP_THRESHOLD": "0.95"}):
            self.assertFalse(stuffing.is_copy(stuffing.shingles(JD), half))

    def test_can_be_disabled(self):
        for value in ("false", "0", "no", "off"):
            with patch.dict(os.environ, {"STUFFING_DEFENSE": value}):
                self.assertFalse(stuffing.enabled(), value)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STUFFING_DEFENSE", None)
            self.assertTrue(stuffing.enabled())


class CosineTest(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(stuffing.cosine([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(stuffing.cosine([1, 0], [0, 1]), 0.0)

    def test_zero_vector_is_safe(self):
        self.assertEqual(stuffing.cosine([0, 0], [1, 1]), 0.0)

    def test_suspicious_threshold_separates_the_measured_populations(self):
        """Legitimate chunks topped out at 0.685 (dev) and 0.620 (test); keyword soup
        started at 0.758 and 0.738. The default has to sit between them."""
        self.assertGreater(stuffing.suspicious_cosine(), 0.685)
        self.assertLess(stuffing.suspicious_cosine(), 0.738)

    def test_missing_vectors_never_flag(self):
        self.assertFalse(stuffing.is_suspiciously_close(None, [1, 0]))
        self.assertFalse(stuffing.is_suspiciously_close([1, 0], None))
        self.assertFalse(stuffing.is_suspiciously_close([], [1, 0]))


class RetrievalDefenceTest(unittest.TestCase):
    """Real embedded Qdrant. Vectors are hand-placed so the attack is deterministic."""

    QUERY = [1.0, 0.0, 0.0, 0.0]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="stuffing_"))
        cls.client = QdrantClient(path=str(cls.tmp / "q"))
        cls.client.create_collection(
            "s", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )

        def point(pid, cid, name, vec, text):
            return models.PointStruct(
                id=pid, vector=vec,
                payload={"candidate_id": cid, "name": name, "cv_path": f"/{cid}.pdf",
                         "chunk_text": text, "years_of_experience": 5, "location": "Pune"},
            )

        cls.client.upsert("s", points=[
            # An honest, genuinely good candidate, moderately close to the query.
            point(1, "honest", "Honest Strong", [0.8, 0.6, 0.0, 0.0], LEGIT_CV),
            point(2, "honest", "Honest Strong", [0.5, 0.0, 0.8, 0.0], "More honest experience text here."),
            # Irrelevant, but one chunk is the job description pasted in, with a
            # vector identical to the query (what embedding that text produces).
            point(3, "stuffer", "Keyword Stuffer", list(cls.QUERY), JD),
            point(4, "stuffer", "Keyword Stuffer", [0.0, 0.0, 0.0, 1.0], "Knits sweaters and sells them."),
            # Irrelevant and honest.
            point(5, "other", "Unrelated Person", [0.0, 0.2, 0.9, 0.3], "Pastry chef for ten years."),
        ])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def search(self, defence=True, jd=JD, **env):
        values = {"STUFFING_DEFENSE": "true" if defence else "false", **env}
        with patch.dict(os.environ, values):
            r = CVRetriever(client=self.client)
            r.collection_name = "s"
            r.hybrid_enabled = False
            return r.search_candidates(self.QUERY, filters=None, top_n=5, query_text=jd)

    def by_id(self, results):
        return {c["candidate_id"]: c for c in results}

    def test_control_the_attack_works_with_the_defence_off(self):
        """If this fails the fixture is wrong and every test below proves nothing."""
        results = self.search(defence=False)
        self.assertEqual(results[0]["candidate_id"], "stuffer")

    def test_the_stuffer_no_longer_ranks_first(self):
        results = self.search()
        self.assertNotEqual(results[0]["candidate_id"], "stuffer")
        self.assertEqual(results[0]["candidate_id"], "honest")

    def test_the_stuffer_is_flagged_and_the_honest_candidate_is_not(self):
        flags = {c["candidate_id"]: c["possible_stuffing"] for c in self.search()}
        self.assertTrue(flags["stuffer"])
        self.assertFalse(flags["honest"])
        self.assertFalse(flags["other"])

    def test_the_copied_text_never_reaches_the_reranker(self):
        """resume_summary is what the cross-encoder and the LLM read. If the stuffed
        chunk is in it, the reranker is fooled even though retrieval was not."""
        stuffer = self.by_id(self.search())["stuffer"]
        self.assertNotIn("senior Python backend engineer", stuffer["resume_summary"])
        self.assertIn("sweaters", stuffer["resume_summary"])

    def test_the_stuffer_is_scored_on_their_real_chunks(self):
        stuffer = self.by_id(self.search())["stuffer"]
        honest = self.by_id(self.search())["honest"]
        self.assertLess(stuffer["score"], honest["score"])

    def test_disabling_the_defence_flags_nothing(self):
        for c in self.search(defence=False):
            self.assertFalse(c["possible_stuffing"])

    def test_a_clean_query_flags_nobody(self):
        """No false positives on honest CVs: none of them copy this query."""
        for c in self.search(jd="Experienced pastry chef for a busy hotel kitchen with banquet duties"):
            self.assertFalse(c["possible_stuffing"], c["candidate_id"])

    def test_a_candidate_whose_every_chunk_is_a_copy_scores_zero_and_sorts_last(self):
        client = QdrantClient(path=str(self.tmp / "all_copy"))
        try:
            client.create_collection(
                "c", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
            )
            payload = lambda cid: {"candidate_id": cid, "name": cid, "cv_path": "", "chunk_text": JD,
                                   "years_of_experience": 1, "location": "Pune"}
            client.upsert("c", points=[
                models.PointStruct(id=1, vector=list(self.QUERY), payload=payload("pure")),
                models.PointStruct(id=2, vector=[0.3, 0.9, 0.0, 0.0],
                                   payload={**payload("fine"), "chunk_text": LEGIT_CV}),
            ])
            with patch.dict(os.environ, {"STUFFING_DEFENSE": "true"}):
                r = CVRetriever(client=client)
                r.collection_name, r.hybrid_enabled = "c", False
                results = r.search_candidates(self.QUERY, filters=None, top_n=5, query_text=JD)
        finally:
            client.close()
        self.assertEqual([c["candidate_id"] for c in results], ["fine", "pure"])
        self.assertEqual(results[-1]["score"], 0.0)
        self.assertTrue(results[-1]["possible_stuffing"])
        self.assertEqual(results[-1]["resume_summary"], "")


class SuspiciousCloseMatchTest(unittest.TestCase):
    """The flag-only signal for text that is NOT a verbatim copy."""

    QUERY = [1.0, 0.0, 0.0, 0.0]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="suspicious_"))
        cls.client = QdrantClient(path=str(cls.tmp / "q"))
        cls.client.create_collection(
            "s", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )

        def point(pid, cid, vec, text):
            return models.PointStruct(
                id=pid, vector=vec,
                payload={"candidate_id": cid, "name": cid, "cv_path": "", "chunk_text": text,
                         "years_of_experience": 1, "location": "Pune"})

        cls.client.upsert("s", points=[
            # Shuffled keywords: no phrase in common with the JD, but very close.
            point(1, "soup", [0.95, 0.31, 0.0, 0.0], "docker fastapi postgresql python backend engineer vector database"),
            point(2, "honest", [0.6, 0.8, 0.0, 0.0], LEGIT_CV),
        ])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def search(self):
        r = CVRetriever(client=self.client)
        r.collection_name, r.hybrid_enabled = "s", False
        return {c["candidate_id"]: c for c in r.search_candidates(
            self.QUERY, filters=None, top_n=5, query_text=JD)}

    def test_soup_is_flagged_as_suspiciously_close(self):
        results = self.search()
        self.assertTrue(results["soup"]["suspicious_match"])
        self.assertFalse(results["honest"]["suspicious_match"])

    def test_it_is_not_flagged_as_a_verbatim_copy(self):
        self.assertFalse(self.search()["soup"]["possible_stuffing"])

    def test_flagging_does_not_change_the_score_or_the_text(self):
        """Flag only: an honest tailored CV can score high too, and silently
        removing its best chunk would penalise exactly that effort."""
        soup = self.search()["soup"]
        self.assertGreater(soup["score"], 0.0)
        self.assertIn("docker", soup["resume_summary"])


class FlagsInTheResponseTest(unittest.TestCase):
    KEY = "stuffing-flags-key"
    GLOBALS = ("embedder", "retriever", "reranker", "store", "jd_filter")

    def setUp(self):
        self._saved = {n: getattr(main_module, n) for n in self.GLOBALS}
        self._saved_key = os.environ.get("API_KEY")
        os.environ["API_KEY"] = self.KEY
        main_module.jd_filter = None
        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.store = MagicMock()
        main_module.reranker = MagicMock()
        main_module.reranker.rerank.return_value = [
            {"candidate_id": "a", "name": "A", "score": 0.9, "match_reasoning": "m", "cv_path": "/a"},
            {"candidate_id": "b", "name": "B", "score": 0.8, "match_reasoning": "m", "cv_path": "/b"},
            {"candidate_id": "c", "name": "C", "score": 0.7, "match_reasoning": "m", "cv_path": "/c"},
        ]
        main_module.retriever = MagicMock()
        main_module.retriever.search_candidates.return_value = [
            {"candidate_id": "a", "name": "A", "cv_path": "/a", "score": 0.9, "resume_summary": "x",
             "possible_stuffing": True, "suspicious_match": False},
            {"candidate_id": "b", "name": "B", "cv_path": "/b", "score": 0.8, "resume_summary": "x",
             "possible_stuffing": False, "suspicious_match": True},
            {"candidate_id": "c", "name": "C", "cv_path": "/c", "score": 0.7, "resume_summary": "x"},
        ]
        self.client = TestClient(main_module.app)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(main_module, name, value)
        if self._saved_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = self._saved_key

    def screen(self):
        response = self.client.post(
            "/api/v1/screen", json={"job_description": "python backend", "top_k": 3},
            headers={"X-API-Key": self.KEY})
        self.assertEqual(response.status_code, 200, response.text)
        return {c["candidate_id"]: c for c in response.json()["candidates"]}

    def test_each_flag_reaches_the_right_candidate(self):
        got = self.screen()
        self.assertEqual(got["a"]["flags"], ["possible_keyword_stuffing"])
        self.assertEqual(got["b"]["flags"], ["suspiciously_close_match"])

    def test_an_unflagged_candidate_has_an_empty_list_not_a_missing_field(self):
        self.assertEqual(self.screen()["c"]["flags"], [])

    def test_a_candidate_can_carry_both_flags(self):
        main_module.retriever.search_candidates.return_value[0]["suspicious_match"] = True
        self.assertEqual(
            self.screen()["a"]["flags"],
            ["possible_keyword_stuffing", "suspiciously_close_match"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
