import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load environment variables BEFORE importing project modules: api.retriever reads
# QDRANT_HOST / QDRANT_COLLECTION / RETRIEVAL_TOP_N at module import time.
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Query, Security, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader

# Import our custom modules
from indexer.utils import get_logger
from indexer.embedder import CVEmbedder
from api.catalogue import CandidateCatalogue
from api.models import (
    CandidateDetail,
    CandidateList,
    CandidateMatch,
    DecisionRecord,
    DecisionRequest,
    PoolStats,
    ScreenRequest,
    ScreenResponse,
    ScreeningDetail,
    ScreeningHistory,
)
from api.retriever import CVRetriever
from api.cross_encoder import CrossEncoderReranker
from api.reranker import CVReranker
from api.store import ScreeningStore

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
store = None
catalogue = None

def build_reranker():
    """Pick the reranking backend from RERANKER_BACKEND.

    Defaults to `llm`, which is what the service already did -- a new default
    would silently change every existing deployment's results.

    `none` is a supported choice rather than an error case: reranking
    measurably *hurts* on queries that retrieval already gets right, so running
    hybrid retrieval alone is a legitimate configuration, not a broken one.
    """
    backend = os.getenv("RERANKER_BACKEND", "llm").strip().lower()

    if backend in ("cross", "cross-encoder", "cross_encoder"):
        reranker = CrossEncoderReranker()
        if reranker.is_configured:
            logger.info("Reranking backend: local cross-encoder.")
            return reranker
        # Falling through to the LLM would make the service quietly ignore an
        # explicit configuration choice, so this is reported loudly instead.
        logger.error(
            "RERANKER_BACKEND=cross-encoder but the model could not be loaded; "
            "ranking will degrade to retrieval order."
        )
        return reranker

    if backend in ("none", "off", "retrieval"):
        logger.info("Reranking backend: none (hybrid retrieval order is final).")
        return NoOpReranker()

    if backend != "llm":
        logger.warning(f"Unknown RERANKER_BACKEND '{backend}'; falling back to llm.")
    logger.info("Reranking backend: LLM.")
    return CVReranker()


class NoOpReranker:
    """Keeps retrieval order, in the shape the rest of the service expects.

    A null object rather than `None` checks scattered through the request path:
    the endpoint already handles a reranker that declines to score, so reusing
    that path is less code and less risk than a second branch.
    """

    is_configured = False

    def rerank(self, jd: str, candidates: list, top_k: int) -> list:
        return [
            {
                "candidate_id": c["candidate_id"],
                "name": c.get("name", "Unknown"),
                "score": float(c.get("score", 0.0)),
                "match_reasoning": "Vector-similarity match (not scored by reranker).",
                "cv_path": c.get("cv_path", ""),
            }
            for c in sorted(candidates, key=lambda c: -float(c.get("score", 0.0)))
        ][:top_k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to pre-load heavy embedding models and database clients."""
    global embedder, retriever, reranker, store, catalogue
    logger.info("Initializing Shortlist microservice...")
    try:
        # Pre-load embedding model on CPU
        embedder = CVEmbedder()
        # Initialize database query client
        retriever = CVRetriever()
        # Reranking stage. Which backend is a deployment choice, not a code
        # change: the LLM reads the job description properly but costs an API
        # call and ~1.5s, while the local cross-encoder is free and ~0.6s.
        # Measured trade-offs are in the README.
        reranker = build_reranker()
        # Screening history and recruiter decisions
        store = ScreeningStore()
        # Read-only views over the indexed pool, sharing the retriever's client
        catalogue = CandidateCatalogue(retriever.client, retriever.collection_name)
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
                filters=request.filters,
                # Raw JD text drives the BM25 branch of hybrid retrieval.
                query_text=request.job_description,
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

    # Persist the run so the shortlist can be reopened or shared without
    # re-running the search, which would cost another LLM call and could return
    # a different order.
    reranked = any(
        "not scored by reranker" not in c["match_reasoning"] for c in reranked_candidates
    )
    try:
        store.save_screening(
            job_id=str(job_id),
            job_description=request.job_description,
            filters=request.filters.model_dump(exclude_none=True) if request.filters else None,
            candidates=[c.model_dump() for c in candidate_matches],
            timings=timings,
            reranked=reranked,
        )
    except Exception as e:
        # A failed write must not lose the results the caller is waiting for.
        logger.error(f"Could not persist screening {job_id}: {e}")

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
    if reranker is not None:
        # Which backend is running is operationally important: an identical
        # request returns different orderings under each one, so a bug report
        # without this field is not reproducible.
        details["reranker_backend"] = type(reranker).__name__
        
    # If core systems are broken, flag response as HTTP 503
    if details["status"] == "unhealthy" or not details["model_loaded"]:
        details["status"] = "unhealthy"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=details
        )
        
    return details


# ---------------------------------------------------------------------------
# Browsing the candidate pool
#
# A recruiter's first question is "who is in here?", which is not a similarity
# search. These read Qdrant's stored payloads directly.
# ---------------------------------------------------------------------------

def _require_ready():
    if not catalogue or not store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services are still initializing or failed to load.",
        )


@app.get(
    "/api/v1/candidates",
    response_model=CandidateList,
    dependencies=[Depends(verify_api_key)],
    tags=["Candidates"],
    summary="Browse the indexed candidate pool",
)
async def list_candidates(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    location: str = Query(None, description="Exact city match, alias-normalised"),
    min_experience: int = Query(None, ge=0, le=60),
    name_contains: str = Query(None, max_length=100),
):
    _require_ready()
    from indexer.parser import normalize_location

    try:
        return catalogue.list_candidates(
            limit=limit,
            offset=offset,
            location=normalize_location(location) if location else None,
            min_experience=min_experience,
            name_contains=name_contains,
        )
    except Exception as e:
        logger.error(f"Could not list candidates: {e}")
        raise HTTPException(status_code=500, detail="Error reading the candidate pool.")


@app.get(
    "/api/v1/candidates/stats",
    response_model=PoolStats,
    dependencies=[Depends(verify_api_key)],
    tags=["Candidates"],
    summary="Summary of what is indexed",
)
async def pool_stats():
    _require_ready()
    try:
        return catalogue.stats()
    except Exception as e:
        logger.error(f"Could not compute pool stats: {e}")
        raise HTTPException(status_code=500, detail="Error reading the candidate pool.")


@app.get(
    "/api/v1/candidates/{candidate_id}",
    response_model=CandidateDetail,
    dependencies=[Depends(verify_api_key)],
    tags=["Candidates"],
    summary="One candidate, including the text that was indexed",
)
async def get_candidate(candidate_id: str):
    _require_ready()
    record = catalogue.get_candidate(candidate_id)
    if not record:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return record


@app.get(
    "/api/v1/candidates/{candidate_id}/cv",
    dependencies=[Depends(verify_api_key)],
    tags=["Candidates"],
    summary="Download the candidate's original CV file",
)
async def get_candidate_cv(candidate_id: str):
    """Serve the original PDF/DOCX.

    The screening response carries a server-side `cv_path`, which a client
    cannot open. This is how a recruiter actually reads the resume.
    """
    _require_ready()
    if not catalogue.get_candidate(candidate_id):
        raise HTTPException(status_code=404, detail="Candidate not found.")

    path = catalogue.resolve_cv_file(candidate_id)
    if not path:
        # Indexed, but the file is gone or sits outside CV_FOLDER_PATH.
        # Distinguished from 404 so the caller knows the candidate exists.
        raise HTTPException(
            status_code=410,
            detail="The CV file is no longer available on this server.",
        )
    return FileResponse(path=str(path), filename=path.name)


# ---------------------------------------------------------------------------
# Screening history and recruiter decisions
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/screenings",
    response_model=ScreeningHistory,
    dependencies=[Depends(verify_api_key)],
    tags=["Screenings"],
    summary="Previous screening runs, newest first",
)
async def list_screenings(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    _require_ready()
    return {
        "total": store.count_screenings(),
        "screenings": store.list_screenings(limit=limit, offset=offset),
    }


@app.get(
    "/api/v1/screenings/{job_id}",
    response_model=ScreeningDetail,
    dependencies=[Depends(verify_api_key)],
    tags=["Screenings"],
    summary="Reopen a screening run with its decisions",
)
async def get_screening(job_id: str):
    """Returns the stored results, not a fresh search.

    Re-running would cost another LLM call and could return a different order,
    so a reopened shortlist shows exactly what the recruiter saw.
    """
    _require_ready()
    run = store.get_screening(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Screening not found.")

    decisions = store.get_decisions(job_id)
    for candidate in run["candidates"]:
        record = decisions.get(candidate["candidate_id"])
        candidate["decision"] = record["decision"] if record else "undecided"
        candidate["note"] = record["note"] if record else None
    return run


@app.put(
    "/api/v1/screenings/{job_id}/candidates/{candidate_id}/decision",
    response_model=DecisionRecord,
    dependencies=[Depends(verify_api_key)],
    tags=["Screenings"],
    summary="Shortlist, reject or flag a candidate",
)
async def set_decision(job_id: str, candidate_id: str, request: DecisionRequest):
    _require_ready()
    run = store.get_screening(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Screening not found.")
    if not any(c["candidate_id"] == candidate_id for c in run["candidates"]):
        # Decisions belong to a run, so a candidate outside it is a client error
        # rather than a missing record.
        raise HTTPException(
            status_code=400,
            detail="That candidate is not part of this screening.",
        )

    store.set_decision(job_id, candidate_id, request.decision, request.note)
    saved = store.get_decisions(job_id)[candidate_id]
    logger.info(f"Decision on {candidate_id} in {job_id}: {request.decision}")
    return {"candidate_id": candidate_id, **saved}


@app.get(
    "/api/v1/screenings/{job_id}/shortlist.csv",
    dependencies=[Depends(verify_api_key)],
    tags=["Screenings"],
    summary="Export the shortlist as CSV",
)
async def export_shortlist(
    job_id: str,
    decision: str = Query("shortlisted", description="Which decision to export, or 'all'"),
):
    """CSV because recruiters work in spreadsheets, and a shortlist has to leave
    the tool to be useful to anyone else."""
    _require_ready()
    run = store.get_screening(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Screening not found.")

    decisions = store.get_decisions(job_id)
    rows = [
        (c, decisions.get(c["candidate_id"], {}))
        for c in run["candidates"]
        if decision == "all"
        or decisions.get(c["candidate_id"], {}).get("decision") == decision
    ]

    def generate():
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["name", "candidate_id", "score", "decision", "note", "match_reasoning"]
        )
        yield buffer.getvalue()
        for candidate, record in rows:
            buffer.seek(0)
            buffer.truncate(0)
            writer.writerow([
                candidate["name"],
                candidate["candidate_id"],
                f"{candidate['score']:.3f}",
                record.get("decision", "undecided"),
                record.get("note") or "",
                candidate["match_reasoning"],
            ])
            yield buffer.getvalue()

    filename = "shortlist-" + str(job_id)[:8] + ".csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=" + filename},
    )
