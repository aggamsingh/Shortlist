"""Detect CV chunks that copy the job description (keyword stuffing).

The attack: an applicant pastes the job description into their resume, often in
white text. Measured on the dev set, one irrelevant candidate doing this ranked
FIRST in 37 of 37 queries under hybrid retrieval, and 28 of 37 landed in the top 5
even after cross-encoder reranking. A screener that this trick defeats is not
trustworthy.

What this catches: near-verbatim copies. A legitimate chunk shares at most 2
four-word phrases with a job description (max fraction 0.12); a copy shares all
of them. The thresholds below were fixed before looking at any attack result and
sit well clear of that legitimate maximum.

What this does NOT stop, measured rather than assumed:

  * Shuffled keyword soup, where the job description's words are reordered so no
    four-word phrase survives, evades the copy detector completely and still
    ranks first. The obvious second signal (how many of the job description's
    words appear) cannot separate it: a legitimate strong CV chunk covers up to
    93% of them. What does separate it is embedding closeness (see below), but
    only as a FLAG: the candidate keeps their rank and the recruiter is warned.
  * Soup diluted with filler, which can sit under the closeness threshold.
  * Paraphrase, which is just a CV.

Stopping those needs signals this module does not have: parse-time detection of
hidden text (white or tiny font), or a coherence check on the chunk. Neither is
built. Treat this as removing the laziest, most common version of the attack and
warning about the next, not as solving it.
"""

import os
import re

NGRAM = 4
# A job description shorter than this many 4-grams (about 11 words) is too short
# to judge a copy of: a legitimate CV can share half of a five-word query by
# accident, so detection is switched off rather than guessing.
MIN_SHARED = 8
DEFAULT_OVERLAP = 0.30


def _tokens(text: str) -> list:
    return re.findall(r"\w+", (text or "").lower())


def shingles(text: str, n: int = NGRAM) -> frozenset:
    """Word n-grams of a text, lowercased and punctuation-insensitive."""
    words = _tokens(text)
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def overlap(query_shingles: frozenset, chunk_text: str) -> tuple:
    """(fraction of the query's n-grams present in the chunk, how many)."""
    if not query_shingles:
        return 0.0, 0
    shared = len(query_shingles & shingles(chunk_text))
    return shared / len(query_shingles), shared


def enabled() -> bool:
    return os.getenv("STUFFING_DEFENSE", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def threshold() -> float:
    return float(os.getenv("STUFFING_OVERLAP_THRESHOLD", DEFAULT_OVERLAP))


def is_copy(query_shingles: frozenset, chunk_text: str) -> bool:
    """True if the chunk reproduces a substantial part of the job description."""
    if len(query_shingles) < MIN_SHARED:
        return False
    fraction, shared = overlap(query_shingles, chunk_text)
    return shared >= MIN_SHARED and fraction >= threshold()


# ---------------------------------------------------------------------------
# "Too good to be true" similarity.
#
# Keyword soup (a shuffled bag of the job description's words) has no phrase in
# common with it, so the copy detector above misses it. It is, however, far closer
# to the job description in embedding space than any real CV chunk ever is:
#
#     dense cosine to the job description   legitimate chunks   keyword soup
#     dev   (2368 chunk-query pairs)        max 0.685           min 0.758
#     test  (1088 chunk-query pairs)        max 0.620           min 0.738
#
# This is a FLAG, not an exclusion. A candidate who tailored an honest CV to a
# posting can also score high, and silently removing their best chunk would
# penalise exactly the effort a recruiter wants. The margin on the attack side is
# thin (0.018 on test), and an attacker who dilutes the soup with filler can walk
# under the threshold; what they cannot do is get far above legitimate chunks
# without being flagged.
# ---------------------------------------------------------------------------

DEFAULT_SUSPICIOUS_COSINE = 0.72


def suspicious_cosine() -> float:
    return float(os.getenv("STUFFING_COSINE_THRESHOLD", DEFAULT_SUSPICIOUS_COSINE))


def cosine(a, b) -> float:
    """Cosine similarity of two vectors; 0.0 if either has no length."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def is_suspiciously_close(query_vector, chunk_vector) -> bool:
    if not query_vector or not chunk_vector:
        return False
    return cosine(query_vector, chunk_vector) >= suspicious_cosine()
