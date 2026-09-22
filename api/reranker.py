import json
import os

from indexer.utils import get_logger

logger = get_logger("api.reranker")

# Upper bound on resume text sent to the LLM per candidate. Retrieved chunks are
# concatenated, so without a cap a handful of verbose CVs can blow the context
# window (and the bill) on a single request. ~1200 chars is roughly 300 tokens,
# enough for the reranker to judge relevance from the best-matching sections.
DEFAULT_MAX_CHARS_PER_CANDIDATE = 1200

# Hard ceiling on candidates per LLM call, independent of RETRIEVAL_CANDIDATES.
DEFAULT_MAX_CANDIDATES = 30


def truncate_text(text: str, max_chars: int) -> str:
    """Trim to max_chars on a word boundary, marking that content was cut."""
    if text is None:
        return ""
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer breaking at whitespace so we do not sever a word mid-token.
    pivot = cut.rfind(" ")
    if pivot > max_chars * 0.6:
        cut = cut[:pivot]
    return cut.rstrip() + " [truncated]"


def parse_score(raw):
    """Parse an LLM-supplied score to a finite float, or None if unusable.

    LLMs return scores as floats, ints, strings, percentages or null. Scale is
    deliberately NOT applied here -- see detect_score_scale.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(f"Non-numeric score from LLM ({raw!r}); using retrieval score.")
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return value


def detect_score_scale(values) -> float:
    """Infer the scale of a batch of scores from their maximum.

    Models asked for 0.00-1.00 sometimes answer on a 0-10 or 0-100 scale. The
    scale must be inferred across the whole response rather than per value: a
    lone 1.5 among 0.x scores means "slightly over-confident", but rescaling it
    in isolation (1.5 -> 0.015) would rank the model's best pick last. Judging
    the batch together keeps the model's relative ordering intact, which is the
    part of its answer that actually drives results.
    """
    finite = [v for v in values if v is not None]
    if not finite:
        return 1.0
    peak = max(finite)
    if peak <= 1.0:
        return 1.0
    if peak <= 2.0:
        # A batch topping out just above 1.0 is an over-confident 0-1 answer, not
        # a 0-10 one -- a genuine 0-10 batch puts its best candidate well above 2.
        # Clamping keeps the top pick at the top; rescaling would bury it at 0.15.
        return 1.0
    if peak <= 10.0:
        return 10.0
    if peak <= 100.0:
        return 100.0
    return peak  # pathological range; normalise to the observed maximum


def apply_scale(value, scale: float, fallback: float) -> float:
    """Rescale a parsed score into [0.0, 1.0], falling back when unusable."""
    if value is None:
        return fallback
    if scale and scale != 1.0:
        value = value / scale
    return max(0.0, min(1.0, value))


class CVReranker:
    """LLM reranking of retrieved candidates, with a strict JSON contract."""

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        # Treat unset placeholders from .env.example as absent.
        self.gemini_key = None if self._is_placeholder(self.gemini_key) else self.gemini_key
        self.groq_key = None if self._is_placeholder(self.groq_key) else self.groq_key

        # Model ids move fast and get retired; keep them configurable.
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

        # Floors stop a stray 0 in .env from silently disabling the caps, which
        # would put the unbounded prompt back in play.
        self.max_chars = max(
            200, int(os.getenv("MAX_CHARS_PER_CANDIDATE", DEFAULT_MAX_CHARS_PER_CANDIDATE))
        )
        self.max_candidates = max(
            1, int(os.getenv("MAX_CANDIDATES_PER_RERANK", DEFAULT_MAX_CANDIDATES))
        )

        if not self.is_configured:
            logger.warning(
                "No LLM key configured (GEMINI_API_KEY / GROQ_API_KEY). "
                "Screening will fall back to vector-similarity ranking."
            )

        self._groq_client = None
        self._genai = None

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        """.env.example ships dummy values; treat them as unconfigured."""
        if not value or not value.strip():
            return True
        return value.strip().lower().startswith("your_")

    @property
    def is_configured(self) -> bool:
        return bool(self.gemini_key or self.groq_key)

    # ---- provider clients (imported lazily so the SDKs stay optional) ----

    def _get_groq(self):
        if self._groq_client is None:
            from groq import Groq

            self._groq_client = Groq(api_key=self.groq_key)
        return self._groq_client

    def _get_genai(self):
        if self._genai is None:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_key)
            self._genai = genai
        return self._genai

    # ---- prompt ----

    def _build_prompt(self, jd: str, candidates: list[dict]) -> str:
        blocks = []
        for i, c in enumerate(candidates):
            summary = truncate_text(c.get("resume_summary", ""), self.max_chars)
            blocks.append(
                f"Candidate Index: {i}\n"
                f"Candidate ID: {c['candidate_id']}\n"
                f"Candidate Name: {c.get('name', 'Unknown')}\n"
                f"Resume Extract:\n{summary}\n"
                "----------------------"
            )
        candidates_str = "\n".join(blocks)

        schema = (
            '{\n'
            '  "rankings": [\n'
            '    {"candidate_id": "<id>", "score": 0.95, '
            '"match_reasoning": "<one sentence>"}\n'
            '  ]\n'
            '}'
        )

        return (
            "You are an expert technical recruiter. Score each candidate against "
            "the job description.\n\n"
            f"Job Description:\n{jd}\n\n"
            f"Candidates:\n{candidates_str}\n\n"
            "For every candidate provide:\n"
            '1. "candidate_id": the exact ID string given above.\n'
            '2. "score": a number between 0.00 and 1.00 for how well they fit.\n'
            '3. "match_reasoning": one concise sentence (max 120 characters).\n\n'
            "Base your judgement only on the resume extracts above. Extracts may be "
            "truncated; do not penalise a candidate for content that appears cut off.\n\n"
            f"Return JSON exactly matching this schema:\n{schema}\n"
            "Include every candidate from the input exactly once."
        )

    # ---- providers ----

    def _rerank_with_gemini(self, jd: str, candidates: list[dict]) -> list[dict]:
        genai = self._get_genai()
        model = genai.GenerativeModel(self.gemini_model)
        logger.info(f"Reranking {len(candidates)} candidates via Gemini ({self.gemini_model}).")
        response = model.generate_content(
            self._build_prompt(jd, candidates),
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
        return self._parse_rankings(response.text, "Gemini")

    def _rerank_with_groq(self, jd: str, candidates: list[dict]) -> list[dict]:
        client = self._get_groq()
        logger.info(f"Reranking {len(candidates)} candidates via Groq ({self.groq_model}).")
        completion = client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": self._build_prompt(jd, candidates)}],
            response_format={"type": "json_object"},
            # Ranking should be reproducible: the same shortlist and JD must give
            # the same order twice, or the evaluation measures sampling noise.
            temperature=0,
        )
        return self._parse_rankings(completion.choices[0].message.content, "Groq")

    @staticmethod
    def _parse_rankings(raw_text: str, provider: str) -> list[dict]:
        """Parse the provider's JSON body into a list of ranking dicts."""
        try:
            payload = json.loads(raw_text)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"{provider} returned invalid JSON: {e}") from e

        # Accept a bare list as well as the documented {"rankings": [...]}.
        if isinstance(payload, list):
            rankings = payload
        elif isinstance(payload, dict):
            rankings = payload.get("rankings")
            if rankings is None:
                for value in payload.values():
                    if isinstance(value, list):
                        rankings = value
                        break
        else:
            rankings = None

        if not isinstance(rankings, list):
            raise RuntimeError(f"{provider} JSON contained no rankings array.")
        return [r for r in rankings if isinstance(r, dict)]

    # ---- public API ----

    def rerank(self, jd: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Rerank retrieved candidates; degrade to retrieval order on failure."""
        if not candidates:
            return []

        shortlist = candidates[: self.max_candidates]
        if len(candidates) > len(shortlist):
            logger.info(
                f"Capping rerank input from {len(candidates)} to {len(shortlist)} candidates."
            )

        rankings = []
        if self.is_configured:
            try:
                rankings = self._call_provider(jd, shortlist)
            except Exception as e:
                # A reranker outage should degrade the ordering, not 500 the request.
                logger.error(f"LLM reranking failed ({e}); falling back to vector scores.")
                rankings = []
        else:
            logger.info("No LLM configured; returning vector-similarity ranking.")

        return self._merge(shortlist, rankings, top_k)

    def _call_provider(self, jd: str, candidates: list[dict]) -> list[dict]:
        """Try the preferred provider, then the other one if it is configured."""
        errors = []
        if self.gemini_key:
            try:
                return self._rerank_with_gemini(jd, candidates)
            except Exception as e:
                errors.append(f"Gemini: {e}")
                logger.warning(f"Gemini reranking failed: {e}")
        if self.groq_key:
            try:
                return self._rerank_with_groq(jd, candidates)
            except Exception as e:
                errors.append(f"Groq: {e}")
                logger.warning(f"Groq reranking failed: {e}")
        raise RuntimeError("; ".join(errors) or "no provider configured")

    def _merge(self, candidates: list[dict], rankings: list[dict], top_k: int) -> list[dict]:
        """Join LLM judgements onto retrieval metadata, keeping metadata authoritative.

        Scores and reasoning come from the LLM; identity, name and cv_path always
        come from the index, so a hallucinated field cannot reach the response.
        """
        by_id = {}
        for r in rankings:
            cid = r.get("candidate_id")
            if isinstance(cid, str) and cid not in by_id:
                by_id[cid] = r  # first judgement wins if the model repeats an id

        # First pass: parse every score so the scale can be judged across the
        # whole batch (a per-value guess would reorder the model's own ranking).
        parsed = {cid: parse_score(r.get("score")) for cid, r in by_id.items()}
        scale = detect_score_scale(parsed.values())
        if scale != 1.0:
            logger.info(f"LLM answered on a 0-{scale:g} scale; normalising to 0-1.")

        results, rescored = [], 0
        for cand in candidates:
            cid = cand["candidate_id"]
            retrieval_score = float(cand.get("score", 0.0))
            judgement = by_id.get(cid)

            if judgement and parsed.get(cid) is not None:
                rescored += 1
                judged = True
                score = apply_scale(parsed.get(cid), scale, retrieval_score)
                reasoning = judgement.get("match_reasoning") or "Matched on resume content."
                if not isinstance(reasoning, str):
                    reasoning = str(reasoning)
            else:
                judged = False
                score = retrieval_score
                reasoning = "Vector-similarity match (not scored by reranker)."

            results.append(
                {
                    "candidate_id": cid,
                    "name": cand.get("name", "Unknown"),
                    "score": score,
                    "match_reasoning": reasoning.strip()[:300],
                    "cv_path": cand.get("cv_path", ""),
                    "_judged": judged,
                }
            )

        unscored = len(candidates) - rescored
        if rankings and unscored:
            logger.warning(f"{unscored} candidate(s) missing from LLM output; used vector scores.")

        # Reranked candidates are ordered ahead of fallbacks rather than merged by
        # raw value: an LLM score and a cosine similarity are different units, and
        # interleaving them would let an unjudged 0.6 cosine outrank a judged 0.55.
        # Within each group the ordering is meaningful. Name breaks ties so the
        # response is deterministic.
        results.sort(key=lambda x: (not x["_judged"], -x["score"], x["name"]))
        for row in results:
            del row["_judged"]
        return results[:top_k]
