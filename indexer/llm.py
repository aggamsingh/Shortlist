"""Provider-agnostic LLM client returning parsed JSON.

Lives in `indexer` rather than `api` because both layers need it and `api`
already depends on `indexer`; putting it the other way round would invert the
dependency. It carries transport concerns only -- provider selection, retry
policy, JSON parsing -- and knows nothing about resumes, reranking or metadata.

Two callers share it: the reranker (api/reranker.py) and metadata extraction
(indexer/metadata.py). They had no business holding two copies of the same
retry and provider-fallback logic.
"""

import json
import os
import re
import time

from indexer.utils import get_logger

logger = get_logger("indexer.llm")

# Backoff ceiling. Provider rate limits are usually enforced over a 60-second
# tokens-per-minute window, so a cap below that can never clear one -- the
# retries burn out inside the same window and the call degrades anyway.
MAX_RETRY_DELAY = 65.0


def is_placeholder(value: str) -> bool:
    """.env.example ships dummy values; treat them as unconfigured."""
    if not value or not value.strip():
        return True
    return value.strip().lower().startswith("your_")


def retry_delay(error: Exception, attempt: int) -> float:
    """Seconds to wait before retrying, or 0 if retrying is pointless.

    Rate limiting is the expected steady state on a metered API, not an outage.
    But a provider asking for longer than the ceiling is reporting an exhausted
    daily quota rather than a per-minute cap, and sleeping through a short
    backoff for that only delays the inevitable fallback.
    """
    text = str(error).lower()
    transient = any(
        marker in text
        for marker in ("rate limit", "rate_limit", "429", "timeout", "503", "502", "overloaded")
    )
    if not transient:
        return 0.0

    hint = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", text)
    if hint:
        wait = float(hint.group(1) or 0) * 60 + float(hint.group(2))
        if wait > MAX_RETRY_DELAY:
            logger.info(
                f"Provider asked for a {wait:.0f}s wait (likely a daily quota), "
                f"beyond the {MAX_RETRY_DELAY:.0f}s ceiling; not retrying."
            )
            return 0.0
        return min(wait + 0.5, MAX_RETRY_DELAY)
    return min(2.0 * (2**attempt), MAX_RETRY_DELAY)


def parse_json_object(raw_text: str, provider: str) -> dict:
    """Parse a provider response body into a dict."""
    try:
        payload = json.loads(raw_text)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"{provider} returned invalid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"{provider} returned JSON that is not an object.")
    return payload


class LLMClient:
    """Calls whichever provider is configured and returns parsed JSON."""

    def __init__(self):
        gemini = os.getenv("GEMINI_API_KEY")
        groq = os.getenv("GROQ_API_KEY")
        self.gemini_key = None if is_placeholder(gemini) else gemini
        self.groq_key = None if is_placeholder(groq) else groq

        # Model ids move fast and get retired; keep them configurable.
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.max_retries = max(0, int(os.getenv("LLM_MAX_RETRIES", "4")))

        self._groq_client = None
        self._genai = None

    @property
    def is_configured(self) -> bool:
        return bool(self.gemini_key or self.groq_key)

    # ---- providers (SDKs imported lazily so both stay optional) ----

    def _call_groq(self, prompt: str) -> dict:
        if self._groq_client is None:
            from groq import Groq

            self._groq_client = Groq(api_key=self.groq_key)
        completion = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            # Deterministic: the same input should give the same answer twice,
            # or evaluation measures sampling noise.
            temperature=0,
        )
        return parse_json_object(completion.choices[0].message.content, "Groq")

    def _call_gemini(self, prompt: str) -> dict:
        if self._genai is None:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_key)
            self._genai = genai
        model = self._genai.GenerativeModel(self.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
        return parse_json_object(response.text, "Gemini")

    def _with_retry(self, call, prompt: str, label: str) -> dict:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return call(prompt)
            except Exception as e:
                last_error = e
                delay = retry_delay(e, attempt) if attempt < self.max_retries else 0.0
                if delay <= 0:
                    raise
                logger.warning(
                    f"{label} unavailable (attempt {attempt + 1}/{self.max_retries + 1}); "
                    f"retrying in {delay:.1f}s."
                )
                time.sleep(delay)
        raise last_error

    def complete_json(self, prompt: str) -> dict:
        """Send a prompt, return the parsed JSON object.

        Gemini is tried first when both are configured, Groq second. Raises if
        no provider is configured or every provider fails -- callers decide
        whether that is fatal or merely a degraded path.
        """
        errors = []
        if self.gemini_key:
            try:
                return self._with_retry(self._call_gemini, prompt, "Gemini")
            except Exception as e:
                errors.append(f"Gemini: {e}")
                logger.warning(f"Gemini call failed: {e}")
        if self.groq_key:
            try:
                return self._with_retry(self._call_groq, prompt, "Groq")
            except Exception as e:
                errors.append(f"Groq: {e}")
                logger.warning(f"Groq call failed: {e}")
        raise RuntimeError("; ".join(errors) or "no LLM provider configured")
