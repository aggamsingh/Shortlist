from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID

class ScreeningFilters(BaseModel):
    min_experience: Optional[int] = Field(default=None, description="Minimum years of experience required")
    location: Optional[str] = Field(default=None, description="Location of candidate (e.g. Delhi)")

class ScreenRequest(BaseModel):
    job_description: str = Field(..., description="The complete text of the job description")
    top_k: Optional[int] = Field(default=None, description="Number of top matches to return")
    filters: Optional[ScreeningFilters] = Field(default=None, description="Filter conditions")

class CandidateMatch(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    name: str = Field(..., description="Name of the candidate")
    score: float = Field(..., description="Relevance score assigned by reranker, between 0.0 and 1.0")
    match_reasoning: str = Field(..., description="A concise explanation detailing why the candidate matches")
    cv_path: str = Field(..., description="Host path to the candidate's original resume file")

class ScreenResponse(BaseModel):
    job_id: UUID = Field(..., description="Generated unique job transaction identifier")
    candidates: List[CandidateMatch] = Field(..., description="List of matched candidates ordered by score")
    screened_at: str = Field(..., description="ISO 8601 UTC timestamp of screening process")
