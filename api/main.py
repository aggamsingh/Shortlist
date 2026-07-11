import os
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

# Import our custom modules
from indexer.utils import get_logger
from indexer.embedder import CVEmbedder
from api.models import ScreenRequest, ScreenResponse, CandidateMatch
from api.retriever import CVRetriever
from api.reranker import CVReranker

# Load environment variables
load_dotenv()

logger = get_logger("api.main")

# Load configuration parameters
API_KEY_VAL = os.getenv("API_KEY")
DEFAULT_TOP_K_VAL = int(os.getenv("DEFAULT_TOP_K", "10"))

# Global singletons pre-loaded on startup
embedder = None
retriever = None
reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to pre-load heavy embedding models and database clients."""
    global embedder, retriever, reranker
    logger.info("Initializing Resume Screening microservice...")
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
    logger.info("Shutting down Resume Screening microservice...")

app = FastAPI(
    title="Resume Screener API",
    description="Microservice for screening resumes against Job Descriptions using RAG & Reranking.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup API Key authentication header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Middleware dependency to check for valid API authorization key."""
    if not API_KEY_VAL:
        logger.warning("API_KEY is not defined in environment variables. Access is unauthenticated.")
        return None
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide it in the X-API-Key header."
        )
    if api_key != API_KEY_VAL:
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
    screened_at = datetime.utcnow().isoformat() + "Z"
    
    top_k = request.top_k if request.top_k is not None else DEFAULT_TOP_K_VAL
    logger.info(f"Received screen request (Job ID: {job_id}) | top_k: {top_k} | filters: {request.filters}")
    
    # 1. Embed job description text
    try:
        jd_vector = embedder.embed_text(request.job_description)
    except Exception as e:
        logger.error(f"Failed to embed Job Description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating Job Description vector embedding."
        )
        
    # 2. Retrieve candidates from Qdrant database (applies min_experience & location filters)
    try:
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
        logger.info("No matching candidates found in database.")
        return ScreenResponse(
            job_id=job_id,
            candidates=[],
            screened_at=screened_at
        )

    # 3. LLM Reranking (Groq or Gemini)
    try:
        reranked_candidates = reranker.rerank(
            jd=request.job_description,
            candidates=retrieved_candidates,
            top_k=top_k
        )
    except Exception as e:
        logger.error(f"Error during LLM reranking: {e}")
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
    
    logger.info(f"Screening complete for Job ID: {job_id}. Found {len(candidate_matches)} matches.")
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
            
    # Check LLM key configurations
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if gemini_key or groq_key:
        details["llm_configured"] = True
        
    # If core systems are broken, flag response as HTTP 503
    if details["status"] == "unhealthy" or not details["model_loaded"]:
        details["status"] = "unhealthy"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=details
        )
        
    return details
