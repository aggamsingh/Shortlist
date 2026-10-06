"""Local cross-encoder reranking, as an alternative to the LLM.

Why a second reranker at all: the LLM reranker costs ~1.5 s and an API call on
every request, which is ~97% of total latency, and it measurably *hurts* on easy
queries. A cross-encoder is the standard middle ground between bi-encoder
retrieval and an LLM judge -- it reads the job description and one CV passage
*together* (so it can condition on the query, which a bi-encoder cannot) but it
is a 22M-parameter classifier rather than a generative model, so it runs locally
in tens of milliseconds with no API call, no key and no rate limit.

What it gives up: no reasoning text. The LLM explains *why* a candidate matched;
a cross-encoder only produces a number. This module quotes the highest-scoring
passage instead, which is honest about what the evidence actually was without
inventing an explanation the model never produced.

Interface-compatible with CVReranker.rerank(), deliberately, so the two can be
swapped and compared on the same harness.
"""

import logging
import math
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# A CV is relevant if *any* part of it is, so passages are scored separately and
# the best one wins. Splitting also keeps each pair inside the model's 512-token
# window: relying on the tokenizer's silent truncation would quietly discard the
# tail of every long CV, and the tail is where skills sections tend to live.
PASSAGE_WORDS = 180
PASSAGE_OVERLAP = 40

# The retriever joins a candidate's matched chunks with this separator.
CHUNK_SEPARATOR = "\n---\n"

FALLBACK_REASONING = "Vector-similarity match (not scored by reranker)."


def sigmoid(x: float) -> float:
    """Squash a relevance logit into 0-1.

    Monotonic, and that is the point rather than a detail: the LLM path once
    rescaled scores per-value and inverted the model's own ranking. Any
    transform applied here must be order-preserving, which is asserted in the
    tests.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # exp(-x) overflows for very negative x; this form does not.
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def split_passages(text: str) -> list[str]:
    """Split a candidate's text into model-sized passages."""
    if not text or not text.strip():
        return []

    pieces = [p.strip() for p in text.split(CHUNK_SEPARATOR) if p.strip()]
    passages = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= PASSAGE_WORDS:
            passages.append(piece)
            continue
        step = PASSAGE_WORDS - PASSAGE_OVERLAP
        for start in range(0, len(words), step):
            window = words[start : start + PASSAGE_WORDS]
            if window:
                passages.append(" ".join(window))
            if start + PASSAGE_WORDS >= len(words):
                break
    return passages


class CrossEncoderReranker:
    """Reranks with a local cross-encoder. Degrades to retrieval order."""

    # One model shared across requests. Loading costs ~1s and ~90MB, and a
    # per-request load would dominate the latency this exists to avoid.
    _model = None
    _load_failed = False
    _lock = threading.Lock()

    def __init__(self):
        self.model_name = os.getenv("CROSS_ENCODER_MODEL", DEFAULT_MODEL)
        self.max_candidates = max(
            1, int(os.getenv("MAX_CANDIDATES_PER_RERANK", "30"))
        )
        self.batch_size = max(1, int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32")))
        self.max_length = max(128, int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512")))

    # ---------------------------------------------------------------- loading

    def _load(self):
        """Load once, and remember failure so every later request is not slow."""
        cls = type(self)
        if cls._model is not None or cls._load_failed:
            return cls._model
        with cls._lock:
            if cls._model is not None or cls._load_failed:
                return cls._model
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading cross-encoder '{self.model_name}' on CPU...")
                cls._model = CrossEncoder(
                    self.model_name, device="cpu", max_length=self.max_length
                )
                logger.info("Cross-encoder loaded.")
            except Exception as e:
                # Treated exactly like an LLM outage: the ordering degrades, the
                # request does not fail.
                cls._load_failed = True
                logger.error(f"Could not load cross-encoder '{self.model_name}': {e}")
        return cls._model

    @property
    def is_configured(self) -> bool:
        """True if the model can be loaded. No key required, unlike the LLM."""
        return self._load() is not None

    # ---------------------------------------------------------------- scoring

    def _score_pairs(self, pairs: list[tuple]) -> list[float]:
        model = self._load()
        if model is None or not pairs:
            return []
        # ms-marco models are trained with num_labels=1 and no output
        # activation, so predict() returns unbounded logits (roughly -11..+11).
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(v) for v in raw]

    def rerank(self, jd: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score each candidate against the job description and reorder.

        Same contract as CVReranker.rerank: identity and metadata come from the
        index, never from the model, and a failure degrades to retrieval order.
        """
        if not candidates:
            return []

        shortlist = candidates[: self.max_candidates]
        if len(candidates) > len(shortlist):
            logger.info(
                f"Capping rerank input from {len(candidates)} to {len(shortlist)}."
            )

        # Build every (jd, passage) pair up front so the model runs in one
        # batched call rather than once per candidate.
        pairs, owners = [], []
        for index, cand in enumerate(shortlist):
            passages = split_passages(cand.get("resume_summary", ""))
            for passage in passages:
                pairs.append((jd, passage))
                owners.append(index)

        scores = self._score_pairs(pairs) if pairs else []
        if len(scores) != len(pairs):
            # Partial output is not something to paper over by guessing.
            if scores:
                logger.error(
                    f"Cross-encoder returned {len(scores)} scores for "
                    f"{len(pairs)} pairs; falling back to retrieval order."
                )
            scores = []

        # Best passage per candidate, and which passage it was.
        best: dict[int, tuple] = {}
        for pair_index, score in enumerate(scores):
            owner = owners[pair_index]
            if owner not in best or score > best[owner][0]:
                best[owner] = (score, pairs[pair_index][1])

        results = []
        for index, cand in enumerate(shortlist):
            judged = index in best
            if judged:
                logit, passage = best[index]
                score = sigmoid(logit)
                excerpt = " ".join(passage.split())[:200]
                reasoning = f"Cross-encoder relevance {logit:+.2f}. Best-matching section: {excerpt}"
            else:
                score = float(cand.get("score", 0.0))
                reasoning = FALLBACK_REASONING

            results.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "name": cand.get("name", "Unknown"),
                    "score": score,
                    "match_reasoning": reasoning.strip()[:300],
                    "cv_path": cand.get("cv_path", ""),
                    "_judged": judged,
                }
            )

        unjudged = len(shortlist) - len(best)
        if unjudged and best:
            logger.warning(
                f"{unjudged} candidate(s) had no usable text; used vector scores."
            )

        # Same ordering rule as the LLM path, for the same reason: a sigmoid
        # relevance and a cosine similarity are different units, so judged
        # candidates form a block ahead of fallbacks rather than being
        # interleaved by raw value.
        results.sort(key=lambda x: (not x["_judged"], -x["score"], x["name"]))
        for row in results:
            del row["_judged"]
        return results[:top_k]
