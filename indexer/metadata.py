"""Candidate metadata extraction: cheap regex first, LLM only when it fails.

The problem
-----------
Regex extraction is fast and free but brittle. Years of experience only matches
phrases like "5 years of experience", so a CV listing date ranges
("2019 - 2024") yields 0. Location matches a hardcoded city list, so anyone
outside it is "Unknown". The candidate's name comes from the *filename*, which
is wrong whenever the file is called "cv_final_v2.pdf".

Why not just use an LLM for everything
--------------------------------------
Cost. One call per CV across a large corpus is real money and real latency, for
a job that regex already does correctly most of the time. So the LLM runs only
on the fields regex could not resolve, and the result is cached by content hash
so re-indexing never pays twice for the same document.

That makes the expensive path proportional to the *failure rate* of the cheap
one, rather than to corpus size.

Degradation
-----------
No key, a provider outage, or malformed output all fall back to whatever regex
produced. Metadata extraction never fails an indexing run.
"""

import json
import os
from pathlib import Path

from indexer.llm import LLMClient
from indexer.parser import extract_years_of_experience, normalize_location
from indexer.utils import get_logger

logger = get_logger("indexer.metadata")

# How much of the CV to show the model. The identifying details -- name,
# contact line, opening summary -- are at the top, and sending a whole resume
# to resolve three fields is waste.
LLM_CONTEXT_CHARS = 1500

UNKNOWN_NAME = "Unknown Candidate"
UNKNOWN_LOCATION = "Unknown"


class MetadataCache:
    """Content-hash keyed cache of LLM extraction results."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._data = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not read metadata cache at {path}: {e}. Starting empty.")

    def get(self, file_hash: str):
        return self._data.get(file_hash)

    def put(self, file_hash: str, value: dict) -> None:
        self._data[file_hash] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"Saved metadata cache ({len(self._data)} entries) to {self.path}")
        except Exception as e:
            logger.warning(f"Could not save metadata cache: {e}")


# Words that betray a filename-derived "name" as an artefact rather than a
# person: "Cv Final V", "Resume Copy", "Document Updated".
FILENAME_NOISE = frozenset(
    """
    cv resume curriculum vitae profile doc document file copy final draft new old
    latest updated version v ver rev sample template untitled
    """.split()
)


def looks_like_person_name(value: str) -> bool:
    """Whether a filename-derived name is plausibly a real person's name.

    clean_candidate_name only returns "Unknown Candidate" when the filename
    reduces to nothing. A file called cv_final_v2.pdf survives that as
    "Cv Final V", which is junk but passes an equality check -- so the LLM was
    never asked to fix precisely the case it exists for.
    """
    if not value or value == UNKNOWN_NAME:
        return False
    tokens = [t for t in value.split() if t]
    if not 2 <= len(tokens) <= 4:
        return False  # real names here are "First Last", occasionally three parts
    if any(t.lower() in FILENAME_NOISE for t in tokens):
        return False
    if any(len(t) < 2 or not t.isalpha() for t in tokens):
        return False
    return True


# Words that mark a string as a job title rather than a person's name. Models
# asked for "the candidate's name" sometimes answer with their role.
TITLE_WORDS = frozenset(
    """
    engineer developer architect manager lead senior junior principal staff
    consultant analyst scientist designer specialist director head intern
    programmer administrator officer executive associate
    """.split()
)

# Characters a real name may contain. Anything else -- @, digits, punctuation --
# means the model returned an email, a title with decoration, or a sentence.
_NAME_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'.")


def _clean_name(value) -> str:
    """Accept a model-supplied name only if it looks like one.

    A character-ratio test is not enough: "Senior Backend Engineer @ Acme!!" is
    90% letters and spaces and sails through one. This uses a strict character
    allowlist plus a job-title check instead.
    """
    if not isinstance(value, str):
        return ""
    name = " ".join(value.split()).strip(" .,-")
    if not (2 <= len(name) <= 60):
        return ""
    if any(c not in _NAME_ALLOWED for c in name):
        return ""
    tokens = name.split()
    if not 1 <= len(tokens) <= 4:
        return ""
    if any(t.lower().strip(".") in TITLE_WORDS for t in tokens):
        return ""
    return name.title()


def _clean_years(value) -> int:
    """Accept a plausible number of years, else 0."""
    try:
        years = int(float(value))
    except (TypeError, ValueError):
        return 0
    return years if 0 <= years <= 60 else 0


class MetadataExtractor:
    """Regex-first metadata extraction with an optional LLM fallback."""

    def __init__(self, cache_path: str = None, client: LLMClient = None):
        self.enabled = os.getenv("METADATA_LLM_FALLBACK", "true").strip().lower() not in (
            "0", "false", "no",
        )
        self.cache = MetadataCache(
            cache_path or os.getenv("METADATA_CACHE_PATH", "./data/metadata_cache.json")
        )
        self._client = client
        self._client_failed = False
        self.stats = {"regex_only": 0, "llm_called": 0, "cache_hits": 0, "llm_failed": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = LLMClient()
        return self._client

    def _llm_available(self) -> bool:
        # One hard failure disables the fallback for the rest of the run. Without
        # this, an exhausted quota costs a failed call and a backoff for every
        # remaining CV in the corpus.
        if not self.enabled or self._client_failed:
            return False
        try:
            return self.client.is_configured
        except Exception:
            return False

    def _build_prompt(self, text: str, missing: list) -> str:
        wanted = {
            "name": '"name": the candidate\'s full personal name as written on the CV',
            "years_of_experience": '"years_of_experience": total professional experience in whole years, as an integer. Infer it from employment date ranges if it is not stated outright.',
            "location": '"location": the city the candidate is based in. Use the contact details, not an employer or university address.',
        }
        fields = "\n".join(f"{i + 1}. {wanted[f]}" for i, f in enumerate(missing))
        schema = ", ".join(f'"{f}": ...' for f in missing)
        return (
            "Extract structured details from this CV extract.\n\n"
            f"CV:\n{text[:LLM_CONTEXT_CHARS]}\n\n"
            f"Return JSON with exactly these fields:\n{fields}\n\n"
            f"Format: {{{schema}}}\n"
            "If a field genuinely cannot be determined from the text, use null for "
            "it. Do not guess at a value that is not supported by the CV."
        )

    def extract(self, text: str, filename_name: str, file_hash: str = None) -> dict:
        """Return {name, years_of_experience, location, source}.

        ``filename_name`` is the name derived from the filename, used as the
        regex-path answer for the name field.
        """
        from indexer.run import extract_location  # local import: avoids a cycle

        years = extract_years_of_experience(text)
        location = extract_location(text)
        name = filename_name

        missing = []
        if years == 0:
            missing.append("years_of_experience")
        if location == UNKNOWN_LOCATION:
            missing.append("location")
        if not looks_like_person_name(name):
            missing.append("name")

        result = {
            "name": name,
            "years_of_experience": years,
            "location": location,
            "source": "regex",
        }

        if not missing:
            self.stats["regex_only"] += 1
            return result

        if file_hash:
            cached = self.cache.get(file_hash)
            if cached:
                self.stats["cache_hits"] += 1
                result.update({k: v for k, v in cached.items() if k in result})
                result["source"] = "cache"
                return result

        if not self._llm_available():
            self.stats["regex_only"] += 1
            return result

        logger.info(f"Regex could not resolve {missing}; asking the LLM.")
        try:
            answer = self.client.complete_json(self._build_prompt(text, missing))
            self.stats["llm_called"] += 1
        except Exception as e:
            self.stats["llm_failed"] += 1
            self._client_failed = True
            logger.warning(
                f"Metadata LLM fallback failed ({e}); keeping regex values and "
                "disabling the fallback for the rest of this run."
            )
            return result

        resolved = {}
        if "years_of_experience" in missing:
            value = _clean_years(answer.get("years_of_experience"))
            if value:
                resolved["years_of_experience"] = value
        if "location" in missing:
            raw = answer.get("location")
            if isinstance(raw, str) and raw.strip():
                resolved["location"] = normalize_location(raw)
        if "name" in missing:
            value = _clean_name(answer.get("name"))
            if value:
                resolved["name"] = value

        if resolved:
            result.update(resolved)
            result["source"] = "llm"
            if file_hash:
                self.cache.put(file_hash, resolved)
        return result

    def save_cache(self) -> None:
        self.cache.save()

    def log_summary(self) -> None:
        s = self.stats
        total = s["regex_only"] + s["llm_called"] + s["cache_hits"] + s["llm_failed"]
        if not total:
            return
        logger.info(
            f"Metadata: {s['regex_only']} regex-only, {s['cache_hits']} cached, "
            f"{s['llm_called']} LLM calls, {s['llm_failed']} LLM failures "
            f"({s['llm_called']}/{total} CVs needed the expensive path)."
        )
