"""The recruiter-facing API: browse the pool, read a CV, decide, export.

Before these endpoints existed the service could only answer "rank these
candidates" and then forgot it had done so. A recruiter could not see who was in
the pool, open a resume (`cv_path` is a server-side path a client cannot use),
revisit a shortlist, or record a single decision.

These run against a real embedded Qdrant and a real SQLite store, because the
value of this layer is in the persistence and the file serving -- the parts a
mock would assume away.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from docx import Document
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

import api.main as main_module
from api.catalogue import CandidateCatalogue
from api.store import ScreeningStore
from indexer.embedder import CVEmbedder

HEADERS = {"X-API-Key": "recruiter-test-key"}
COLLECTION = "recruiter_test"

PEOPLE = [
    ("Ananya Rao", "Bangalore", 7, "Python, FastAPI, Qdrant vector database, Docker"),
    ("Aditya Verma", "Mumbai", 4, "React, TypeScript, Next.js, CSS"),
    ("Sandeep Reddy", "Hyderabad", 8, "Kubernetes, AWS, Terraform, Docker"),
]


def write_cv(path: Path, name: str, location: str, years: int, skills: str) -> None:
    doc = Document()
    doc.add_paragraph(name)
    doc.add_paragraph(f"{location}, India")
    doc.add_paragraph("Summary")
    doc.add_paragraph(f"Engineer with {years} years of experience.")
    doc.add_paragraph("Skills")
    doc.add_paragraph(skills)
    doc.save(str(path))


class RecruiterApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.embedder = CVEmbedder()
        except Exception as e:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"embedding model unavailable: {e}")

        cls.tmp = Path(tempfile.mkdtemp(prefix="recruiter_api_"))
        cls.cv_dir = cls.tmp / "cvs"
        cls.cv_dir.mkdir()
        cls.client_q = QdrantClient(path=str(cls.tmp / "qdrant"))
        cls.client_q.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=cls.embedder.dimension, distance=models.Distance.COSINE
            ),
        )

        points = []
        cls.ids = {}
        for i, (name, location, years, skills) in enumerate(PEOPLE, start=1):
            filename = f"{name.replace(' ', '_')}_Resume.docx"
            write_cv(cls.cv_dir / filename, name, location, years, skills)
            cid = f"cand-{i}"
            cls.ids[name] = cid
            text = f"{name} {location} {years} years {skills}"
            points.append(models.PointStruct(
                id=i,
                vector=cls.embedder.embed_text(text),
                payload={
                    "candidate_id": cid,
                    "name": name,
                    "location": location,
                    "years_of_experience": years,
                    # Absolute, inside the configured CV root, as the indexer writes it.
                    "cv_path": str(cls.cv_dir / filename).replace("\\", "/"),
                    "chunk_text": text,
                },
            ))
        cls.client_q.upsert(COLLECTION, points=points)

        import os

        os.environ["API_KEY"] = "recruiter-test-key"
        os.environ["CV_FOLDER_PATH"] = str(cls.cv_dir)

        main_module.store = ScreeningStore(path=str(cls.tmp / "store.db"))
        main_module.catalogue = CandidateCatalogue(cls.client_q, COLLECTION)
        cls.client = TestClient(main_module.app)

    @classmethod
    def tearDownClass(cls):
        try:
            main_module.store.close()
            cls.client_q.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, path):
        return self.client.get(path, headers=HEADERS)

    # ---- browsing the pool ----

    def test_list_candidates(self):
        body = self.get("/api/v1/candidates").json()
        self.assertEqual(body["total"], 3)
        self.assertEqual([c["name"] for c in body["candidates"]],
                         ["Aditya Verma", "Ananya Rao", "Sandeep Reddy"])

    def test_list_is_paginated(self):
        body = self.get("/api/v1/candidates?limit=2&offset=1").json()
        self.assertEqual(body["total"], 3, "total must count matches, not the page")
        self.assertEqual(len(body["candidates"]), 2)

    def test_filter_by_location_accepts_an_alias(self):
        """A recruiter typing Bengaluru must find a CV indexed as Bangalore."""
        body = self.get("/api/v1/candidates?location=Bengaluru").json()
        self.assertEqual([c["name"] for c in body["candidates"]], ["Ananya Rao"])

    def test_filter_by_experience_and_name(self):
        self.assertEqual(self.get("/api/v1/candidates?min_experience=7").json()["total"], 2)
        self.assertEqual(
            self.get("/api/v1/candidates?name_contains=sandeep").json()["total"], 1
        )

    def test_candidate_detail_includes_indexed_text(self):
        body = self.get(f"/api/v1/candidates/{self.ids['Ananya Rao']}").json()
        self.assertEqual(body["name"], "Ananya Rao")
        self.assertIn("Qdrant", body["indexed_text"])

    def test_unknown_candidate_is_404(self):
        self.assertEqual(self.get("/api/v1/candidates/nope").status_code, 404)

    def test_pool_stats(self):
        body = self.get("/api/v1/candidates/stats").json()
        self.assertEqual(body["candidates"], 3)
        self.assertEqual(body["experience_years"]["max"], 8)
        self.assertIn("Bangalore", body["locations"])

    # ---- reading the actual CV ----

    def test_cv_download_returns_the_real_file(self):
        """The whole point: cv_path is a server path a client cannot open."""
        response = self.get(f"/api/v1/candidates/{self.ids['Ananya Rao']}/cv")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 1000)
        self.assertTrue(response.content.startswith(b"PK"), "not a .docx payload")

    def test_cv_download_for_unknown_candidate_is_404(self):
        self.assertEqual(self.get("/api/v1/candidates/nope/cv").status_code, 404)

    def test_cv_outside_the_cv_root_is_refused(self):
        """Serving a payload-supplied path unchecked is a traversal hole."""
        outside = self.tmp / "escaped.docx"
        write_cv(outside, "Escaped", "Delhi", 1, "none")
        self.client_q.upsert(COLLECTION, points=[models.PointStruct(
            id=99,
            vector=self.embedder.embed_text("escaped"),
            payload={"candidate_id": "cand-escape", "name": "Escaped",
                     "location": "Delhi", "years_of_experience": 1,
                     "cv_path": str(outside).replace("\\", "/"),
                     "chunk_text": "escaped"},
        )])
        try:
            # 410, not 200: the candidate exists but the file will not be served.
            self.assertEqual(self.get("/api/v1/candidates/cand-escape/cv").status_code, 410)
        finally:
            self.client_q.delete(COLLECTION, points_selector=models.PointIdsList(points=[99]))

    # ---- screening history and decisions ----

    def _screen(self, top_k=3):
        """Persist a screening run under a job id unique to this test.

        Sharing one id across tests let decisions written by an earlier test
        leak into a later one asserting everything was undecided.
        """
        job_id = f"job-{self.id().rsplit('.', 1)[-1]}"
        main_module.store.save_screening(
            job_id=job_id,
            job_description="Backend engineer with Qdrant vector database experience",
            filters=None,
            candidates=[
                {"candidate_id": self.ids[name], "name": name, "score": 0.9 - i / 10,
                 "match_reasoning": "matched", "cv_path": ""}
                for i, (name, *_rest) in enumerate(PEOPLE[:top_k])
            ],
            timings={"total_ms": 12.3},
            reranked=True,
        )
        return job_id

    def test_reopening_a_screening_returns_stored_results(self):
        job = self._screen()
        body = self.get(f"/api/v1/screenings/{job}").json()
        self.assertTrue(body["reranked"])
        self.assertEqual(len(body["candidates"]), 3)
        self.assertTrue(all(c["decision"] == "undecided" for c in body["candidates"]))

    def test_unknown_screening_is_404(self):
        self.assertEqual(self.get("/api/v1/screenings/missing").status_code, 404)

    def test_decision_round_trip(self):
        job = self._screen()
        cid = self.ids["Ananya Rao"]
        response = self.client.put(
            f"/api/v1/screenings/{job}/candidates/{cid}/decision",
            json={"decision": "shortlisted", "note": "Qdrant in production"},
            headers=HEADERS,
        )
        self.assertEqual(response.status_code, 200)

        reopened = self.get(f"/api/v1/screenings/{job}").json()
        decided = next(c for c in reopened["candidates"] if c["candidate_id"] == cid)
        self.assertEqual(decided["decision"], "shortlisted")
        self.assertEqual(decided["note"], "Qdrant in production")

    def test_decision_can_be_revised(self):
        """Recruiters change their minds; the last write wins."""
        job = self._screen()
        cid = self.ids["Aditya Verma"]
        for value in ("maybe", "rejected"):
            self.client.put(
                f"/api/v1/screenings/{job}/candidates/{cid}/decision",
                json={"decision": value}, headers=HEADERS,
            )
        reopened = self.get(f"/api/v1/screenings/{job}").json()
        decided = next(c for c in reopened["candidates"] if c["candidate_id"] == cid)
        self.assertEqual(decided["decision"], "rejected")

    def test_invalid_decision_rejected(self):
        job = self._screen()
        cid = self.ids["Ananya Rao"]
        response = self.client.put(
            f"/api/v1/screenings/{job}/candidates/{cid}/decision",
            json={"decision": "hire_immediately"}, headers=HEADERS,
        )
        self.assertEqual(response.status_code, 422)

    def test_decision_on_candidate_outside_the_run_is_400(self):
        """A decision belongs to a run, so this is a client error not a 404."""
        job = self._screen(top_k=1)
        response = self.client.put(
            f"/api/v1/screenings/{job}/candidates/{self.ids['Sandeep Reddy']}/decision",
            json={"decision": "shortlisted"}, headers=HEADERS,
        )
        self.assertEqual(response.status_code, 400)

    # ---- export ----

    def test_csv_export_contains_only_the_shortlist(self):
        job = self._screen()
        self.client.put(
            f"/api/v1/screenings/{job}/candidates/{self.ids['Ananya Rao']}/decision",
            json={"decision": "shortlisted", "note": "yes"}, headers=HEADERS,
        )
        self.client.put(
            f"/api/v1/screenings/{job}/candidates/{self.ids['Aditya Verma']}/decision",
            json={"decision": "rejected"}, headers=HEADERS,
        )
        csv_text = self.get(f"/api/v1/screenings/{job}/shortlist.csv").text
        self.assertIn("Ananya Rao", csv_text)
        self.assertNotIn("Aditya Verma", csv_text)
        self.assertIn("name,candidate_id,score,decision,note,match_reasoning",
                      csv_text.splitlines()[0])

    def test_csv_export_all_includes_every_candidate(self):
        job = self._screen()
        csv_text = self.get(f"/api/v1/screenings/{job}/shortlist.csv?decision=all").text
        for name, *_rest in PEOPLE:
            self.assertIn(name, csv_text)

    def test_history_counts_shortlisted(self):
        job = self._screen()
        self.client.put(
            f"/api/v1/screenings/{job}/candidates/{self.ids['Ananya Rao']}/decision",
            json={"decision": "shortlisted"}, headers=HEADERS,
        )
        body = self.get("/api/v1/screenings").json()
        entry = next(s for s in body["screenings"] if s["job_id"] == job)
        self.assertEqual(entry["shortlisted_count"], 1)
        self.assertEqual(entry["candidate_count"], 3)

    # ---- auth ----

    def test_every_recruiter_endpoint_requires_the_api_key(self):
        for path in ("/api/v1/candidates", "/api/v1/candidates/stats",
                     f"/api/v1/candidates/{self.ids['Ananya Rao']}",
                     f"/api/v1/candidates/{self.ids['Ananya Rao']}/cv",
                     "/api/v1/screenings", "/api/v1/screenings/any-id"):
            self.assertEqual(self.client.get(path).status_code, 401, path)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = ScreeningStore(path=str(self.tmp / "s.db"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_survives_reopening(self):
        """A restart must not lose shortlists."""
        self.store.save_screening("j1", "jd", None, [{"candidate_id": "c1"}], {}, True)
        self.store.set_decision("j1", "c1", "shortlisted", "note")
        self.store.close()

        reopened = ScreeningStore(path=str(self.tmp / "s.db"))
        try:
            self.assertIsNotNone(reopened.get_screening("j1"))
            self.assertEqual(reopened.get_decisions("j1")["c1"]["decision"], "shortlisted")
        finally:
            reopened.close()
            self.store = reopened  # so tearDown does not double-close

    def test_rejects_unknown_decision_value(self):
        self.store.save_screening("j1", "jd", None, [], {}, False)
        with self.assertRaises(ValueError):
            self.store.set_decision("j1", "c1", "definitely_hire")

    def test_history_is_newest_first(self):
        for i in range(3):
            self.store.save_screening(f"j{i}", f"jd {i}", None, [], {}, False)
        ids = [s["job_id"] for s in self.store.list_screenings()]
        self.assertEqual(ids[0], "j2")

    def test_long_job_description_is_previewed(self):
        self.store.save_screening("j1", "x" * 500, None, [], {}, False)
        preview = self.store.list_screenings()[0]["job_description_preview"]
        self.assertLess(len(preview), 200)
        self.assertTrue(preview.endswith("..."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
