import re
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ScreeningFilters(BaseModel):
    """Hard filters applied in Qdrant before semantic ranking."""

    min_experience: Optional[int] = Field(
        default=None, ge=0, le=60,
        description="Minimum years of experience required",
    )
    location: Optional[str] = Field(
        default=None, max_length=100,
        description="Candidate location, e.g. Delhi. Known aliases are normalised "
                    "(Bengaluru matches Bangalore).",
    )


class ScreenRequest(BaseModel):
    job_description: str = Field(
        ..., min_length=1, max_length=20000,
        description="The complete text of the job description",
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Number of top matches to return",
    )
    filters: Optional[ScreeningFilters] = Field(
        default=None, description="Filter conditions"
    )

    @field_validator("job_description")
    @classmethod
    def job_description_not_blank(cls, value: str) -> str:
        """Reject whitespace-only descriptions.

        min_length alone accepts "   ", which embeds to a meaningless vector and
        returns confident nonsense. Failing at the edge is cheaper than serving
        it and beats burning an LLM call on an empty query.
        """
        if not value.strip():
            raise ValueError("job_description must not be blank")
        # "!!! ???" and an emoji string are not blank, but they carry no content to
        # match on. Without this they returned ten confidently ranked candidates.
        # \w is Unicode-aware, so Hindi, Arabic and CJK text still passes.
        if not re.search(r"\w", value):
            raise ValueError("job_description must contain at least one letter or digit")
        return value.strip()


class CandidateMatch(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    name: str = Field(..., description="Name of the candidate")
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Relevance score, 0.0-1.0. From the LLM reranker when "
                    "available, otherwise the vector similarity score.",
    )
    match_reasoning: str = Field(
        ..., description="A concise explanation of why the candidate matches"
    )
    cv_path: str = Field(..., description="Path to the candidate's original resume file")
    flags: List[str] = Field(
        default_factory=list,
        description=(
            "Warnings a recruiter should see before trusting this ranking. "
            "'possible_keyword_stuffing': part of this CV copies the job "
            "description verbatim; that part was excluded from scoring. "
            "'suspiciously_close_match': a chunk is far closer to the job "
            "description than any real CV chunk measured, which is what shuffled "
            "keyword lists look like; it is flagged but NOT excluded, because an "
            "honest tailored CV can score high too. Neither is a guarantee: an "
            "empty list does not mean the CV is clean."
        ),
    )


class ScreenResponse(BaseModel):
    job_id: UUID = Field(..., description="Generated unique job transaction identifier")
    candidates: List[CandidateMatch] = Field(
        ..., description="Matched candidates, best first"
    )
    screened_at: str = Field(..., description="ISO 8601 UTC timestamp of screening")
    reranked: bool = Field(
        False,
        description=(
            "True if at least one candidate was scored by the reranker. False means "
            "the order is plain retrieval order: the reranker was unavailable, "
            "failed, or the 'none' backend is configured. Without this, a degraded "
            "ranking is indistinguishable from a good one."
        ),
    )
    timings: dict = Field(
        default_factory=dict,
        description="Per-stage wall-clock cost in milliseconds (embed, retrieve, rerank).",
    )


# ---------------------------------------------------------------------------
# Browsing the candidate pool
# ---------------------------------------------------------------------------

class CandidateSummary(BaseModel):
    """One candidate as they appear in a roster listing."""

    candidate_id: str
    name: str
    location: str
    years_of_experience: int
    cv_path: str = Field(..., description="Server-side path; fetch the file via /cv")
    chunk_count: int = Field(..., description="Indexed chunks for this candidate")


class CandidateList(BaseModel):
    total: int = Field(..., description="Candidates matching the filters, before paging")
    limit: int
    offset: int
    candidates: List[CandidateSummary]


class CandidateDetail(CandidateSummary):
    indexed_text: str = Field(
        ..., description="The text actually indexed for this candidate, chunks rejoined"
    )


class PoolStats(BaseModel):
    candidates: int
    chunks: int
    locations: dict
    experience_years: dict


# ---------------------------------------------------------------------------
# Screening history and recruiter decisions
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    decision: str = Field(
        ...,
        description="One of: shortlisted, rejected, maybe, undecided",
    )
    note: Optional[str] = Field(
        default=None, max_length=2000,
        description="Free-text note, e.g. why this candidate was rejected",
    )

    @field_validator("decision")
    @classmethod
    def known_decision(cls, value: str) -> str:
        allowed = ("shortlisted", "rejected", "maybe", "undecided")
        if value not in allowed:
            raise ValueError(f"decision must be one of {allowed}")
        return value


class DecisionRecord(BaseModel):
    candidate_id: str
    decision: str
    note: Optional[str] = None
    updated_at: str


class ScreeningSummary(BaseModel):
    job_id: str
    job_description_preview: str
    candidate_count: int
    shortlisted_count: int
    reranked: bool = Field(..., description="False means no candidate was scored by the reranker")
    created_at: str


class ScreeningHistory(BaseModel):
    total: int
    screenings: List[ScreeningSummary]


class ScreenedCandidate(CandidateMatch):
    """A search result, plus whatever the recruiter decided about them."""

    decision: str = Field(default="undecided")
    note: Optional[str] = None


class ScreeningDetail(BaseModel):
    job_id: str
    job_description: str
    filters: Optional[dict] = None
    candidates: List[ScreenedCandidate]
    timings: dict = Field(default_factory=dict)
    reranked: bool
    created_at: str
