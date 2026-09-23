"""Metadata extraction: regex fast path, LLM fallback, and cost discipline.

The feature only earns its place if the expensive path is rare and the cheap one
is never abandoned when the expensive one fails. Both properties are asserted
here, alongside rejection of implausible model output.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from indexer.metadata import (
    MetadataCache,
    MetadataExtractor,
    _clean_name,
    _clean_years,
    looks_like_person_name,
)

# Defeats every regex path: date ranges instead of "N years", a city outside the
# hardcoded list, and nothing usable in the filename.
HARD_CV = """Priyanka Deshmukh
Senior Backend Engineer
Nagpur, Maharashtra, India | priyanka.d@example.com

Experience
Acme Corp, Senior Backend Engineer, 2018 - 2024
Built FastAPI services.
Beta Ltd, Backend Engineer, 2015 - 2018
"""

EASY_CV = """Jane Doe
Backend Engineer
Delhi, India

Summary
Engineer with 6 years of experience in Python.
"""


class GoodClient:
    is_configured = True
    last_prompt = None

    def complete_json(self, prompt):
        GoodClient.last_prompt = prompt
        return {
            "name": "Priyanka Deshmukh",
            "years_of_experience": 9,
            "location": "Nagpur",
        }


class BrokenClient:
    is_configured = True
    calls = 0

    def complete_json(self, prompt):
        BrokenClient.calls += 1
        raise RuntimeError("quota exhausted")


class NameHeuristicTest(unittest.TestCase):
    def test_real_names_are_trusted(self):
        for name in ("Jane Doe", "Priyanka Deshmukh", "John Smith Jr"):
            self.assertTrue(looks_like_person_name(name), name)

    def test_filename_artefacts_are_not_trusted(self):
        """The case the fallback exists for.

        clean_candidate_name turns cv_final_v2.pdf into "Cv Final V", which is
        junk but is not the literal "Unknown Candidate" sentinel. An equality
        check missed it entirely.
        """
        for name in ("Cv Final V", "Resume Copy", "Document Updated",
                     "Unknown Candidate", "Developer", ""):
            self.assertFalse(looks_like_person_name(name), name)


class ValidationTest(unittest.TestCase):
    def test_job_titles_and_junk_are_rejected_as_names(self):
        self.assertEqual(_clean_name("Senior Backend Engineer @ Acme!!"), "")
        self.assertEqual(_clean_name("a"), "")
        self.assertEqual(_clean_name(None), "")
        self.assertEqual(_clean_name(12345), "")
        self.assertEqual(_clean_name("x" * 80), "")

    def test_plausible_names_are_accepted(self):
        self.assertEqual(_clean_name("  priyanka   deshmukh "), "Priyanka Deshmukh")

    def test_implausible_year_counts_are_rejected(self):
        for value in (999, -3, "abc", None, {}):
            self.assertEqual(_clean_years(value), 0, repr(value))

    def test_plausible_year_counts_are_accepted(self):
        self.assertEqual(_clean_years(9), 9)
        self.assertEqual(_clean_years("7"), 7)
        self.assertEqual(_clean_years(6.0), 6)


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        GoodClient.last_prompt = None
        BrokenClient.calls = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _extractor(self, client=None, name="c.json"):
        return MetadataExtractor(cache_path=str(self.tmp / name), client=client)

    def test_regex_success_never_calls_the_llm(self):
        """Cost discipline: the expensive path must stay proportional to failures."""
        extractor = self._extractor(client=GoodClient())
        result = extractor.extract(EASY_CV, "Jane Doe", file_hash="h")
        self.assertEqual(result["source"], "regex")
        self.assertEqual(result["years_of_experience"], 6)
        self.assertEqual(result["location"], "Delhi")
        self.assertEqual(extractor.stats["llm_called"], 0)
        self.assertIsNone(GoodClient.last_prompt)

    def test_llm_resolves_what_regex_could_not(self):
        extractor = self._extractor(client=GoodClient())
        result = extractor.extract(HARD_CV, "Cv Final V", file_hash="h")
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["name"], "Priyanka Deshmukh")
        self.assertEqual(result["years_of_experience"], 9)
        self.assertEqual(result["location"], "Nagpur")

    def test_prompt_requests_only_the_missing_fields(self):
        """Asking for fields regex already resolved wastes tokens."""
        extractor = self._extractor(client=GoodClient())
        extractor.extract(HARD_CV, "Priyanka Deshmukh", file_hash="h")
        prompt = GoodClient.last_prompt
        self.assertIn("years_of_experience", prompt)
        self.assertIn("location", prompt)
        self.assertNotIn('"name":', prompt)

    def test_prompt_truncates_the_cv(self):
        extractor = self._extractor(client=GoodClient())
        extractor.extract(HARD_CV + ("filler text " * 5000), "Cv Final V", file_hash="h")
        self.assertLess(len(GoodClient.last_prompt), 3000)

    def test_cache_prevents_a_second_call_for_identical_content(self):
        first = self._extractor(client=GoodClient(), name="shared.json")
        first.extract(HARD_CV, "Cv Final V", file_hash="same")
        first.save_cache()

        second = self._extractor(client=BrokenClient(), name="shared.json")
        result = second.extract(HARD_CV, "Cv Final V", file_hash="same")
        self.assertEqual(result["source"], "cache")
        self.assertEqual(result["years_of_experience"], 9)
        self.assertEqual(BrokenClient.calls, 0, "cache did not prevent the call")

    def test_provider_failure_falls_back_to_regex(self):
        extractor = self._extractor(client=BrokenClient())
        result = extractor.extract(HARD_CV, "Cv Final V", file_hash="h")
        self.assertEqual(result["source"], "regex")
        self.assertEqual(result["name"], "Cv Final V")
        self.assertEqual(result["years_of_experience"], 0)

    def test_one_failure_disables_the_fallback_for_the_run(self):
        """An exhausted quota must not cost a failed call per remaining CV."""
        extractor = self._extractor(client=BrokenClient())
        for i in range(5):
            extractor.extract(HARD_CV, "Cv Final V", file_hash=f"h{i}")
        self.assertEqual(BrokenClient.calls, 1)
        self.assertEqual(extractor.stats["llm_failed"], 1)

    def test_implausible_model_output_is_discarded(self):
        class JunkClient:
            is_configured = True

            def complete_json(self, prompt):
                return {"name": "Senior Engineer @ Acme!!",
                        "years_of_experience": 999, "location": ""}

        extractor = self._extractor(client=JunkClient())
        result = extractor.extract(HARD_CV, "Cv Final V", file_hash="h")
        self.assertEqual(result["name"], "Cv Final V")
        self.assertEqual(result["years_of_experience"], 0)
        self.assertEqual(result["location"], "Unknown")

    def test_disabled_fallback_uses_regex_only(self):
        extractor = self._extractor(client=GoodClient())
        extractor.enabled = False
        result = extractor.extract(HARD_CV, "Cv Final V", file_hash="h")
        self.assertEqual(result["source"], "regex")

    def test_unconfigured_client_uses_regex_only(self):
        class Unconfigured:
            is_configured = False

            def complete_json(self, prompt):
                raise AssertionError("must not be called")

        extractor = self._extractor(client=Unconfigured())
        self.assertEqual(extractor.extract(HARD_CV, "Cv Final V")["source"], "regex")

    def test_llm_location_is_normalised(self):
        class AliasClient:
            is_configured = True

            def complete_json(self, prompt):
                return {"location": "Bengaluru", "years_of_experience": 4, "name": None}

        extractor = self._extractor(client=AliasClient())
        result = extractor.extract(HARD_CV, "Cv Final V", file_hash="h")
        self.assertEqual(result["location"], "Bangalore")


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        path = self.tmp / "c.json"
        cache = MetadataCache(str(path))
        cache.put("h", {"location": "Pune"})
        cache.save()
        self.assertEqual(MetadataCache(str(path)).get("h"), {"location": "Pune"})

    def test_missing_file_is_empty_not_an_error(self):
        self.assertIsNone(MetadataCache(str(self.tmp / "nope.json")).get("h"))

    def test_corrupt_file_degrades_to_empty(self):
        path = self.tmp / "bad.json"
        path.write_text("not json", encoding="utf-8")
        self.assertIsNone(MetadataCache(str(path)).get("h"))

    def test_save_is_a_noop_when_nothing_changed(self):
        path = self.tmp / "c.json"
        MetadataCache(str(path)).save()
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
