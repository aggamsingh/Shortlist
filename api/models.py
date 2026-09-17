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


class ScreenResponse(BaseModel):
    job_id: UUID = Field(..., description="Generated unique job transaction identifier")
    candidates: List[CandidateMatch] = Field(
        ..., description="Matched candidates, best first"
    )
    screened_at: str = Field(..., description="ISO 8601 UTC timestamp of screening")
