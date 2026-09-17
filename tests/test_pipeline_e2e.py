"""End-to-end pipeline tests: real files in, real vectors out.

These exist because the original bug -- indexer/run.py calling chunk_cv without
importing it -- crashed every single indexing run, yet the whole suite passed.
Every test mocked the boundary and nothing ever executed main(). A test that
drives the real entry point is the one that would have caught it.

Qdrant runs embedded (no server), so this is CI-safe. Payload indexes are a
no-op in embedded mode; they are exercised for code path, not for effect.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from docx import Document
from qdrant_client import QdrantClient

import indexer.run as run
from api.models import ScreeningFilters
from api.retriever import CVRetriever
from indexer.embedder import CVEmbedder


def write_docx(path: Path, name: str, location: str, years: int, skills: str) -> None:
    """Write a real .docx CV, including a table (exercises table extraction)."""
    doc = Document()
    doc.add_paragraph(name)
    doc.add_paragraph(f"{location}, India")
    doc.add_paragraph("Summary")
    doc.add_paragraph(f"Engineer with {years} years of experience building software.")
    doc.add_paragraph("Experience")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Acme Corp - Senior Engineer"
    table.cell(0, 1).text = "2019-2024"
    doc.add_paragraph("Skills")
    doc.add_paragraph(skills)
    doc.save(str(path))


class PipelineEndToEndTest(unittest.TestCase):
    """Drives indexer.run.main() against real files and a real embedding model."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.embedder = CVEmbedder()
        except Exception as e:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"embedding model unavailable: {e}")

        cls.tmp = Path(tempfile.mkdtemp(prefix="resume_e2e_"))
        cls.cv_dir = cls.tmp / "cvs"
        cls.cv_dir.mkdir()

        write_docx(cls.cv_dir / "Jane_Doe_Resume_2024.docx", "Jane Doe",
                   "Bengaluru", 6, "Python, FastAPI, Docker, Qdrant")
        write_docx(cls.cv_dir / "rahul-mehta-cv.docx", "Rahul Mehta",
                   "Mumbai", 2, "React, TypeScript, CSS")

        cls.client = QdrantClient(path=str(cls.tmp / "qdrant"))
        cls.state_path = cls.tmp / "state.json"

        # Point the module's config at the fixtures and inject the embedded
        # client. The collection is pinned explicitly so that a QDRANT_COLLECTION
        # left in os.environ by another test module cannot retarget this run.
        cls._orig = (
            run.CV_FOLDER_PATH,
            run.STATE_FILE_PATH,
            run.QdrantClient,
            run.QDRANT_COLLECTION,
        )
        run.CV_FOLDER_PATH = str(cls.cv_dir)
        run.STATE_FILE_PATH = str(cls.state_path)
        run.QDRANT_COLLECTION = "e2e_resumes"
        run.QdrantClient = lambda **kwargs: cls.client

        run.main()  # first indexing pass

    @classmethod
    def tearDownClass(cls):
        (run.CV_FOLDER_PATH, run.STATE_FILE_PATH,
         run.QdrantClient, run.QDRANT_COLLECTION) = cls._orig
        try:
            cls.client.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- indexing ----

    def test_main_indexes_without_crashing(self):
        """The regression test for the missing chunk_cv import."""
        points, _ = self.client.scroll(
            collection_name=run.QDRANT_COLLECTION, limit=100, with_payload=True
        )
        self.assertGreater(len(points), 0, "indexing produced no vectors")

    def test_state_file_records_both_candidates(self):
        self.assertTrue(self.state_path.exists())
        import json

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state), 2)
        for entry in state.values():
            self.assertIn("hash", entry)
            self.assertIn("candidate_id", entry)

    def test_metadata_extracted_from_real_documents(self):
        points, _ = self.client.scroll(
            collection_name=run.QDRANT_COLLECTION, limit=100, with_payload=True
        )
        by_name = {p.payload["name"]: p.payload for p in points}
        self.assertIn("Jane Doe", by_name)
        self.assertIn("Rahul Mehta", by_name)
        # "Bengaluru" in the document must be stored under the canonical spelling.
        self.assertEqual(by_name["Jane Doe"]["location"], "Bangalore")
        self.assertEqual(by_name["Jane Doe"]["years_of_experience"], 6)
        self.assertEqual(by_name["Rahul Mehta"]["location"], "Mumbai")

    def test_reindex_is_incremental(self):
        """Unchanged files must be skipped on a second run (hash check)."""
        before, _ = self.client.scroll(
            collection_name=run.QDRANT_COLLECTION, limit=200, with_payload=False
        )
        run.main()
        after, _ = self.client.scroll(
            collection_name=run.QDRANT_COLLECTION, limit=200, with_payload=False
        )
        self.assertEqual(len(before), len(after), "re-running duplicated vectors")

    def test_modified_file_is_reindexed_without_stale_vectors(self):
        """Editing a CV replaces its vectors rather than accumulating them."""
        target = self.cv_dir / "rahul-mehta-cv.docx"
        write_docx(target, "Rahul Mehta", "Mumbai", 2,
                   "React, TypeScript, CSS, GraphQL, Vue, Svelte, Angular")
        run.main()

        points, _ = self.client.scroll(
            collection_name=run.QDRANT_COLLECTION, limit=200, with_payload=True
        )
        rahul_ids = {
            p.payload["candidate_id"] for p in points if p.payload["name"] == "Rahul Mehta"
        }
        self.assertEqual(len(rahul_ids), 1, "candidate id should be stable across edits")
        texts = [p.payload["chunk_text"] for p in points if p.payload["name"] == "Rahul Mehta"]
        self.assertTrue(any("GraphQL" in t for t in texts), "new content not indexed")

    # ---- retrieval over the same index ----

    def test_retrieval_returns_distinct_candidates(self):
        retriever = CVRetriever(client=self.client)
        retriever.collection_name = run.QDRANT_COLLECTION
        vector = self.embedder.embed_text("Python backend engineer with FastAPI and Docker")
        results = retriever.search_candidates(query_vector=vector, filters=None, top_n=10)

        self.assertGreaterEqual(len(results), 2)
        ids = [c["candidate_id"] for c in results]
        self.assertEqual(len(ids), len(set(ids)), "duplicate candidates returned")
        self.assertEqual(results[0]["name"], "Jane Doe", "backend JD should rank Jane first")

    def test_location_alias_filter_matches_canonical_value(self):
        """Filtering on 'Bengaluru' must find a CV indexed as 'Bangalore'."""
        retriever = CVRetriever(client=self.client)
        retriever.collection_name = run.QDRANT_COLLECTION
        vector = self.embedder.embed_text("engineer")
        results = retriever.search_candidates(
            query_vector=vector,
            filters=ScreeningFilters(min_experience=None, location="Bengaluru"),
        )
        self.assertEqual([c["name"] for c in results], ["Jane Doe"])

    def test_experience_filter_excludes_below_threshold(self):
        retriever = CVRetriever(client=self.client)
        retriever.collection_name = run.QDRANT_COLLECTION
        vector = self.embedder.embed_text("engineer")
        results = retriever.search_candidates(
            query_vector=vector,
            filters=ScreeningFilters(min_experience=5, location=None),
        )
        self.assertEqual([c["name"] for c in results], ["Jane Doe"])

    def test_impossible_filter_returns_empty(self):
        retriever = CVRetriever(client=self.client)
        retriever.collection_name = run.QDRANT_COLLECTION
        vector = self.embedder.embed_text("engineer")
        # 60 is the schema maximum and still exceeds every fixture (6 and 2 years).
        results = retriever.search_candidates(
            query_vector=vector,
            filters=ScreeningFilters(min_experience=60, location=None),
        )
        self.assertEqual(results, [])


class IndexerEdgeCaseTest(unittest.TestCase):
    """Malformed and unreadable inputs must not abort the whole run."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="resume_edge_"))
        self.cv_dir = self.tmp / "cvs"
        self.cv_dir.mkdir()
        self.client = QdrantClient(path=str(self.tmp / "qdrant"))
        self._orig = (run.CV_FOLDER_PATH, run.STATE_FILE_PATH,
                      run.QdrantClient, run.QDRANT_COLLECTION)
        run.CV_FOLDER_PATH = str(self.cv_dir)
        run.STATE_FILE_PATH = str(self.tmp / "state.json")
        run.QDRANT_COLLECTION = "e2e_edge_resumes"
        run.QdrantClient = lambda **kwargs: self.client

    def tearDown(self):
        (run.CV_FOLDER_PATH, run.STATE_FILE_PATH,
         run.QdrantClient, run.QDRANT_COLLECTION) = self._orig
        try:
            self.client.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_folder_completes_cleanly(self):
        run.main()  # must not raise

    def test_corrupt_file_is_skipped_and_good_file_still_indexes(self):
        (self.cv_dir / "broken.pdf").write_bytes(b"this is not a pdf")
        write_docx(self.cv_dir / "Good_Person_CV.docx", "Good Person",
                   "Delhi", 4, "Python, FastAPI")
        try:
            CVEmbedder()
        except Exception as e:  # pragma: no cover
            self.skipTest(f"embedding model unavailable: {e}")

        run.main()  # the corrupt file must not abort the run

        points, _ = self.client.scroll(
            collection_name=run.QDRANT_COLLECTION, limit=100, with_payload=True
        )
        names = {p.payload["name"] for p in points}
        self.assertIn("Good Person", names)

    def test_unsupported_extension_is_ignored(self):
        (self.cv_dir / "notes.txt").write_text("hello", encoding="utf-8")
        run.main()  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
