import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load environment variables BEFORE importing project modules: api.retriever reads
# QDRANT_HOST / QDRANT_COLLECTION / RETRIEVAL_TOP_N at module import time.
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

# Import our custom modules
from indexer.utils import get_logger
from indexer.embedder import CVEmbedder
from api.models import ScreenRequest, ScreenResponse, CandidateMatch
from api.retriever import CVRetriever
from api.reranker import CVReranker

logger = get_logger("api.main")

# Configuration is read per request rather than captured at import. Binding it
# at import time means the value depends on whether .env loaded before this
# module was first imported -- the same failure mode that made api.retriever
# silently ignore .env, and it makes the auth key untestable in-process.


def get_api_key_setting() -> str:
    return os.getenv("API_KEY")


def get_default_top_k() -> int:
    try:
        return int(os.getenv("DEFAULT_TOP_K", "10"))
    except ValueError:
        logger.warning("DEFAULT_TOP_K is not an integer; falling back to 10.")
        return 10


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp in ISO 8601 with a trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextmanager
def stage_timer(sink: dict, key: str):
    """Record the wall-clock duration of a pipeline stage into ``sink`` (ms)."""
    started = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = round((time.perf_counter() - started) * 1000, 1)

# Global singletons pre-loaded on startup
embedder = None
retriever = None
reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to pre-load heavy embedding models and database clients."""
    global embedder, retriever, reranker
    logger.info("Initializing Shortlist microservice...")
    try:
        # Pre-load embedding model on CPU
        embedder = CVEmbedder()
        # Initialize database query client
        retriever = CVRetriever()
        # Initialize reranking clients
        reranker = CVReranker()
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize core services on startup: {e}")
        raise e
    yield
    logger.info("Shutting down Shortlist microservice...")

app = FastAPI(
    title="Shortlist API",
    description="Microservice for screening resumes against Job Descriptions using RAG & Reranking.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup API Key authentication header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Dependency that checks the shared API key on every request."""
    expected = get_api_key_setting()
    if not expected:
        logger.warning("API_KEY is not defined in environment variables. Access is unauthenticated.")
        return None
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide it in the X-API-Key header."
        )
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key."
        )
    return api_key

@app.post(
    "/api/v1/screen",
    response_model=ScreenResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Screen and Rerank Resume Candidates",
    description="Accepts a Job Description, retrieves candidates from Qdrant, applies filters, reranks them using an LLM, and returns top matches."
)
async def screen_resumes(request: ScreenRequest):
    global embedder, retriever, reranker
    
    if not embedder or not retriever or not reranker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services are still initializing or failed to load. Check server logs."
        )
        
    job_id = uuid.uuid4()
    screened_at = utc_now_iso()

    top_k = request.top_k if request.top_k is not None else get_default_top_k()
    logger.info(f"Received screen request (Job ID: {job_id}) | top_k: {top_k} | filters: {request.filters}")

    # Per-stage timings: embedding and retrieval are CPU/IO bound and stable,
    # while the LLM rerank dominates and varies by provider. Logging them apart
    # is what makes a latency regression attributable to a stage.
    timings = {}

    # 1. Embed job description text
    try:
        with stage_timer(timings, "embed_ms"):
            jd_vector = embedder.embed_text(request.job_description)
    except Exception as e:
        logger.error(f"Failed to embed Job Description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating Job Description vector embedding."
        )

    # 2. Retrieve candidates from Qdrant (applies min_experience & location filters)
    try:
        with stage_timer(timings, "retrieve_ms"):
            retrieved_candidates = retriever.search_candidates(
                query_vector=jd_vector,
                filters=request.filters
            )
    except Exception as e:
        logger.error(f"Error during Qdrant candidate retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during candidate retrieval."
        )

    if not retrieved_candidates:
        logger.info(f"No matching candidates found. timings={timings}")
        return ScreenResponse(
            job_id=job_id,
            candidates=[],
            screened_at=screened_at
        )

    # 3. LLM reranking. CVReranker degrades to vector ordering internally rather
    # than raising, so a provider outage returns ranked results instead of a 500.
    try:
        with stage_timer(timings, "rerank_ms"):
            reranked_candidates = reranker.rerank(
                jd=request.job_description,
                candidates=retrieved_candidates,
                top_k=top_k
            )
    except Exception as e:
        logger.error(f"Unexpected reranking failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing LLM reranking response."
        )

    # 4. Formulate response
    candidate_matches = [
        CandidateMatch(
            candidate_id=item["candidate_id"],
            name=item["name"],
            score=item["score"],
            match_reasoning=item["match_reasoning"],
            cv_path=item["cv_path"]
        ) for item in reranked_candidates
    ]

    timings["total_ms"] = round(sum(timings.values()), 1)
    logger.info(
        f"Screening complete (Job ID: {job_id}). "
        f"retrieved={len(retrieved_candidates)} returned={len(candidate_matches)} timings={timings}"
    )
    return ScreenResponse(
        job_id=job_id,
        candidates=candidate_matches,
        screened_at=screened_at
    )

@app.get(
    "/health",
    summary="Health check endpoint",
    description="Ensures database connectivity, LLM API credentials presence, and local model load status."
)
async def health_check():
    global embedder, retriever, reranker
    
    details = {
        "status": "healthy",
        "qdrant_connected": False,
        "model_loaded": False,
        "llm_configured": False
    }
    
    # Check model loading
    if embedder and embedder.model:
        details["model_loaded"] = True
        
    # Check database connectivity
    if retriever and retriever.client:
        try:
            # Ping Qdrant by querying collections
            retriever.client.get_collections()
            details["qdrant_connected"] = True
        except Exception as e:
            logger.error(f"Health check failed to contact Qdrant: {e}")
            details["status"] = "unhealthy"
            
    # Ask the reranker itself, so placeholder keys from .env.example are not
    # reported as a working LLM configuration.
    if reranker and reranker.is_configured:
        details["llm_configured"] = True
        
    # If core systems are broken, flag response as HTTP 503
    if details["status"] == "unhealthy" or not details["model_loaded"]:
        details["status"] = "unhealthy"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=details
        )
        
    return details
