"""Strip company boilerplate from a job description before it is embedded.

Why this exists. The evaluation's job descriptions are 30-60 tokens of pure
requirements. Real postings are 300-800 tokens and open with "About us" and end
with benefits and an equal-opportunity statement. Measured on the dev set, wrapping
the same requirements in that text cost hybrid retrieval 0.08-0.15 nDCG@5 -- and
it was not truncation (MiniLM reads 256 tokens and most variants stayed under it).
Generic company prose simply dilutes the vector and the BM25 query.

What did NOT work: retrieving per chunk of the posting and fusing the lists.
Boilerplate is most of the chunks, so each noise chunk retrieves its own noise
list and those outvote the real requirements (nDCG 0.43 against 0.77 unfiltered).

What does: score each sentence by whether it reads more like a role requirement
or like company/benefits text, and keep the former. The two prototype phrases
below were written before any result was seen and are not tuned. One threshold
was chosen on dev boilerplate styles A and B, then confirmed once on a third,
independently written style on held-out queries: +0.080 nDCG@5, with no change
on clean job descriptions (-0.001).

Limits, stated plainly: it recovers about 60% of the loss, not all of it; the
threshold is sensitive (0.0 was worse than no filter on one style); and the three
boilerplate styles are my own writing, not a sample of real postings.
"""

import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

# Fixed before any evaluation. Changing them is a new experiment, not a tweak.
REQUIREMENT_PROTOTYPE = (
    "Required skills, technologies, tools and years of experience for this role. "
    "Responsibilities and qualifications for the position."
)
BOILERPLATE_PROTOTYPE = (
    "About our company, mission, culture and values. Benefits, salary, perks, "
    "insurance, leave and how to apply. Equal opportunity employer statement."
)

DEFAULT_THRESHOLD = -0.05

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")


def split_sentences(text: str) -> list:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    return array / norm if norm else array


class JDFilter:
    """Keeps the sentences of a job description that read like requirements."""

    def __init__(self, embedder, threshold: float = None):
        self.embedder = embedder
        self.threshold = (
            threshold
            if threshold is not None
            else float(os.getenv("JD_FILTER_THRESHOLD", DEFAULT_THRESHOLD))
        )
        # Embedded once, at construction, rather than per request.
        self._requirement = _unit(embedder.embed_text(REQUIREMENT_PROTOTYPE))
        self._boilerplate = _unit(embedder.embed_text(BOILERPLATE_PROTOTYPE))

    def margins(self, sentences: list) -> list:
        """Requirement-likeness minus boilerplate-likeness, per sentence."""
        if not sentences:
            return []
        vectors = self.embedder.embed_texts(sentences)  # one batched call
        return [
            float(_unit(v) @ self._requirement - _unit(v) @ self._boilerplate)
            for v in vectors
        ]

    def apply(self, text: str) -> str:
        """Return the job description with boilerplate sentences removed.

        Never returns an empty string: if nothing survives, or there is nothing to
        split, the original text is returned unchanged. A filter that can erase a
        query would turn a bad guess into a failed search.
        """
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return text
        kept = [
            sentence
            for sentence, margin in zip(sentences, self.margins(sentences))
            if margin > self.threshold
        ]
        if not kept:
            return text
        if len(kept) < len(sentences):
            logger.info(f"JD filter kept {len(kept)} of {len(sentences)} sentences.")
        return " ".join(kept)


def build_jd_filter(embedder):
    """The configured filter, or None when disabled (JD_FILTER=false)."""
    if os.getenv("JD_FILTER", "true").strip().lower() in ("0", "false", "no", "off"):
        return None
    return JDFilter(embedder)
