"""Awkward files through the real indexer.

Found by throwing corrupt, empty, oversized and duplicate files at it. Most of it
held up: every bad file was skipped with a logged reason, nothing crashed, and a
re-run created no duplicates. Two gaps did not:

  * the same CV saved under two names was indexed as two candidates, so a
    recruiter would see one person twice with identical scores
  * there was no size cap: one 120,000-word file became 700 chunks, and a
    600,000-word file stalled a run for four minutes
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from docx import Document
from qdrant_client import QdrantClient

import indexer.run as run
from indexer.embedder import CVEmbedder
from indexer.parser import content_fingerprint, truncate_words


def write_docx(path: Path, *lines: str) -> None:
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(str(path))


CV = (
    "Jane Doe", "Bengaluru, India", "Summary",
    "Backend engineer with 6 years of experience building Python services.",
    "Skills", "Python, FastAPI, Docker, Qdrant",
)


class TruncateWordsTest(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(truncate_words("one two three", 10), ("one two three", False))

    def test_cuts_after_the_nth_word(self):
        self.assertEqual(truncate_words("one two three four", 2), ("one two", True))

    def test_text_of_exactly_the_limit_is_not_flagged(self):
        self.assertEqual(truncate_words("one two three", 3), ("one two three", False))

    def test_trailing_whitespace_after_the_limit_is_not_truncation(self):
        self.assertEqual(truncate_words("one two  \n\n", 2), ("one two  \n\n", False))

    def test_keeps_line_breaks_inside_the_kept_part(self):
        text, cut = truncate_words("one\ntwo\nthree\nfour", 3)
        self.assertEqual(text, "one\ntwo\nthree")
        self.assertTrue(cut)

    def test_zero_or_negative_disables_the_cap(self):
        text = "word " * 1000
        self.assertEqual(truncate_words(text, 0), (text, False))
        self.assertEqual(truncate_words(text, -5), (text, False))

    def test_empty_text(self):
        self.assertEqual(truncate_words("", 5), ("", False))


class ContentFingerprintTest(unittest.TestCase):
    def test_identical_text_matches(self):
        self.assertEqual(content_fingerprint("Jane Doe\nPython"), content_fingerprint("Jane Doe\nPython"))

    def test_case_and_whitespace_do_not_matter(self):
        self.assertEqual(
            content_fingerprint("Jane   Doe\n\nPYTHON developer"),
            content_fingerprint("jane doe python developer"),
        )

    def test_different_words_differ(self):
        self.assertNotEqual(content_fingerprint("python developer"), content_fingerprint("java developer"))

    def test_a_single_changed_word_differs(self):
        self.assertNotEqual(
            content_fingerprint("six years of python"), content_fingerprint("seven years of python")
        )


class IndexerRobustnessTest(unittest.TestCase):
    """Real indexer, real embedding model, real embedded Qdrant."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.embedder = CVEmbedder()
        except Exception as e:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"embedding model unavailable: {e}")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="indexer_robust_"))
        self.cv_dir = self.tmp / "cvs"
        self.cv_dir.mkdir()
        self.client = QdrantClient(path=str(self.tmp / "qdrant"))
        self.state_path = self.tmp / "state.json"
        self._orig = (run.CV_FOLDER_PATH, run.STATE_FILE_PATH, run.QdrantClient,
                      run.QDRANT_COLLECTION, run.MAX_CV_WORDS, run.BM25_STATE_PATH,
                      run.METADATA_CACHE_PATH)
        run.CV_FOLDER_PATH = str(self.cv_dir)
        run.STATE_FILE_PATH = str(self.state_path)
        run.QDRANT_COLLECTION = "robust"
        run.BM25_STATE_PATH = str(self.tmp / "bm25.json")
        run.METADATA_CACHE_PATH = str(self.tmp / "meta.json")
        run.QdrantClient = lambda **kwargs: self.client

    def tearDown(self):
        (run.CV_FOLDER_PATH, run.STATE_FILE_PATH, run.QdrantClient, run.QDRANT_COLLECTION,
         run.MAX_CV_WORDS, run.BM25_STATE_PATH, run.METADATA_CACHE_PATH) = self._orig
        try:
            self.client.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def points(self):
        points, _ = self.client.scroll("robust", limit=10000, with_payload=True)
        return points

    def indexed_files(self):
        return sorted({Path(p.payload["cv_path"]).name for p in self.points()})

    def state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    # ---- duplicates ----

    def test_the_same_cv_under_two_names_is_indexed_once(self):
        write_docx(self.cv_dir / "jane.docx", *CV)
        write_docx(self.cv_dir / "jane_copy.docx", *CV)
        run.main()
        self.assertEqual(len({p.payload["candidate_id"] for p in self.points()}), 1)

    def test_the_duplicate_is_skipped_and_the_distinct_cv_kept(self):
        write_docx(self.cv_dir / "jane.docx", *CV)
        write_docx(self.cv_dir / "jane_copy.docx", *CV)
        write_docx(self.cv_dir / "other.docx", "Rahul Mehta", "Mumbai", "React, TypeScript")
        run.main()
        self.assertEqual(len({p.payload["candidate_id"] for p in self.points()}), 2)

    def test_a_duplicate_added_later_does_not_displace_the_original(self):
        """The ordering trap. 'a_copy' sorts before 'z_original', so without
        already-indexed-first ordering the new copy would be kept and the original,
        skipped as the 'duplicate', would leave its old vectors stale in the index."""
        write_docx(self.cv_dir / "z_original.docx", *CV)
        run.main()
        first = self.indexed_files()
        self.assertEqual(first, ["z_original.docx"])

        write_docx(self.cv_dir / "a_copy.docx", *CV)
        run.main()
        self.assertEqual(self.indexed_files(), ["z_original.docx"])
        self.assertNotIn("a_copy.docx", self.indexed_files())

    def test_rerunning_with_a_duplicate_present_creates_no_new_points(self):
        write_docx(self.cv_dir / "jane.docx", *CV)
        write_docx(self.cv_dir / "jane_copy.docx", *CV)
        run.main()
        before = len(self.points())
        run.main()
        self.assertEqual(len(self.points()), before)

    def test_cvs_that_differ_by_one_word_are_both_kept(self):
        """A fingerprint that matched near-duplicates would silently drop real
        candidates; only identical text counts."""
        write_docx(self.cv_dir / "a.docx", *CV)
        write_docx(self.cv_dir / "b.docx", *[line.replace("6 years", "7 years") for line in CV])
        run.main()
        self.assertEqual(len({p.payload["candidate_id"] for p in self.points()}), 2)

    # ---- size cap ----

    def test_an_oversized_cv_is_cut_to_the_cap(self):
        run.MAX_CV_WORDS = 300
        doc = Document()
        doc.add_paragraph("Jane Doe")
        doc.add_paragraph("Bengaluru, India")
        for _ in range(200):
            doc.add_paragraph("Built Python services with FastAPI and Docker. " * 5)
        doc.save(str(self.cv_dir / "huge.docx"))
        run.main()

        words = sum(len(p.payload["chunk_text"].split()) for p in self.points())
        # Chunks overlap, so the total can exceed the cap by the overlap, but it
        # must be nowhere near the ~10,000 words in the file.
        self.assertLess(words, 600)
        self.assertGreater(len(self.points()), 0)

    def test_a_cv_under_the_cap_is_indexed_in_full(self):
        run.MAX_CV_WORDS = 20000
        write_docx(self.cv_dir / "jane.docx", *CV)
        run.main()
        text = " ".join(p.payload["chunk_text"] for p in self.points())
        self.assertIn("Qdrant", text)

    def test_the_cap_can_be_disabled(self):
        run.MAX_CV_WORDS = 0
        doc = Document()
        doc.add_paragraph("Jane Doe")
        for _ in range(60):
            doc.add_paragraph("Built Python services with FastAPI and Docker. " * 5)
        doc.save(str(self.cv_dir / "long.docx"))
        run.main()
        self.assertGreater(sum(len(p.payload["chunk_text"].split()) for p in self.points()), 1000)

    # ---- hostile files ----

    def test_bad_files_are_skipped_and_good_ones_still_indexed(self):
        write_docx(self.cv_dir / "good.docx", *CV)
        (self.cv_dir / "corrupt.docx").write_bytes(b"\x00\x01\x02 not a zip")
        (self.cv_dir / "zero.pdf").write_bytes(b"")
        (self.cv_dir / "truncated.pdf").write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\n")
        write_docx(self.cv_dir / "empty.docx")
        write_docx(self.cv_dir / "blank.docx", "   ", "")
        (self.cv_dir / "notes.txt").write_text("not a cv")
        (self.cv_dir / "legacy.doc").write_bytes(b"\xd0\xcf\x11\xe0 old word")
        run.main()
        self.assertEqual(self.indexed_files(), ["good.docx"])
        self.assertEqual(list(self.state()), [str(self.cv_dir / "good.docx")])

    def test_uppercase_unicode_and_long_names_are_handled(self):
        write_docx(self.cv_dir / "UPPER.DOCX", *CV)
        write_docx(self.cv_dir / "résumé_日本語_🚀 final (v2).docx", "Rahul Mehta", "Mumbai", "React")
        write_docx(self.cv_dir / (("x" * 150) + ".docx"), "Priya Nair", "Chennai", "Go")
        run.main()
        self.assertEqual(len({p.payload["candidate_id"] for p in self.points()}), 3)

    def test_same_filename_in_different_folders_is_two_candidates(self):
        (self.cv_dir / "team_a").mkdir()
        (self.cv_dir / "team_b").mkdir()
        write_docx(self.cv_dir / "team_a" / "cv.docx", *CV)
        write_docx(self.cv_dir / "team_b" / "cv.docx", "Rahul Mehta", "Mumbai", "React, TypeScript")
        run.main()
        self.assertEqual(len({p.payload["candidate_id"] for p in self.points()}), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
