"""
Pure unit tests for indexer parser and utility functions.
No ML dependencies required — runs instantly with zero downloads.
"""
import unittest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer.parser import (
    clean_text,
    extract_years_of_experience,
    chunk_cv,
    chunk_by_words
)
from indexer.run import clean_candidate_name, extract_location
from indexer.utils import calculate_file_hash, load_index_state, save_index_state
import tempfile
import json


class TestTextCleaning(unittest.TestCase):
    def test_clean_text_strips_whitespace(self):
        text = "Hello    World!   \n New Line  "
        self.assertEqual(clean_text(text), "Hello World! New Line")

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_clean_text_single_word(self):
        self.assertEqual(clean_text("   Python   "), "Python")


class TestExperienceExtraction(unittest.TestCase):
    def test_years_pattern(self):
        self.assertEqual(extract_years_of_experience("I have 5 years of experience."), 5)

    def test_yrs_abbreviation(self):
        self.assertEqual(extract_years_of_experience("Total Experience: 12 yrs"), 12)

    def test_no_experience(self):
        self.assertEqual(extract_years_of_experience("No experience mentioned here."), 0)

    def test_plus_notation(self):
        self.assertEqual(extract_years_of_experience("8+ years of experience in Python"), 8)

    def test_max_capped(self):
        # Anything > 40 is treated as invalid
        self.assertEqual(extract_years_of_experience("50 years of experience"), 0)


class TestLocationExtraction(unittest.TestCase):
    def test_delhi(self):
        self.assertEqual(extract_location("I live in Delhi, India."), "Delhi")

    def test_bengaluru_normalized(self):
        self.assertEqual(extract_location("Worked at a startup in Bengaluru."), "Bangalore")

    def test_bangalore_normalized(self):
        self.assertEqual(extract_location("Based in Bangalore."), "Bangalore")

    def test_gurgaon_normalized(self):
        self.assertEqual(extract_location("Office in Gurugram."), "Gurgaon")

    def test_unknown(self):
        self.assertEqual(extract_location("Remote worker."), "Unknown")

    def test_mumbai(self):
        self.assertEqual(extract_location("I am based in Mumbai."), "Mumbai")


class TestCandidateNameCleaning(unittest.TestCase):
    def test_resume_suffix_removed(self):
        self.assertEqual(clean_candidate_name("Jane_Doe_Resume_2026.pdf"), "Jane Doe")

    def test_cv_suffix_removed(self):
        self.assertEqual(clean_candidate_name("john-smith-cv.docx"), "John Smith")

    def test_standalone_cv(self):
        self.assertEqual(clean_candidate_name("cv_developer.pdf"), "Developer")

    def test_empty_fallback(self):
        self.assertEqual(clean_candidate_name("cv.pdf"), "Unknown Candidate")


class TestChunking(unittest.TestCase):
    def test_section_chunking_detects_headers(self):
        text = """
Jane Doe
Jane@example.com

Summary
Passionate engineer with 5 years experience building backend systems.

Experience
Worked at Google for 3 years as a backend developer.
Built REST APIs using FastAPI and Python.

Skills
Python, FastAPI, Qdrant, Docker, PostgreSQL
"""
        chunks = chunk_cv(text)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(any("SUMMARY" in c for c in chunks))
        self.assertTrue(any("EXPERIENCE" in c for c in chunks))
        self.assertTrue(any("SKILLS" in c for c in chunks))

    def test_word_chunking_splits_long_text(self):
        long_text = " ".join(["word"] * 500)
        chunks = chunk_by_words(long_text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be at most chunk_size words
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 200)

    def test_short_text_single_chunk(self):
        short_text = "This is a short resume text."
        chunks = chunk_by_words(short_text)
        self.assertEqual(len(chunks), 1)


class TestStateUtils(unittest.TestCase):
    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "index_state.json")
            test_state = {"file.pdf": {"hash": "abc123", "name": "Test User"}}
            save_index_state(state_path, test_state)
            loaded = load_index_state(state_path)
            self.assertEqual(loaded, test_state)

    def test_load_missing_state_returns_empty(self):
        result = load_index_state("/nonexistent/path/state.json")
        self.assertEqual(result, {})

    def test_file_hash(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w') as f:
            f.write("Test content for hashing")
            tmp_path = f.name
        try:
            hash1 = calculate_file_hash(tmp_path)
            hash2 = calculate_file_hash(tmp_path)
            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 32)  # MD5 hex digest length
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
