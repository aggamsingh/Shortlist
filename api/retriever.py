import os

from qdrant_client import QdrantClient
from qdrant_client.http import models

from api.models import ScreeningFilters
from indexer.utils import connect_qdrant, get_logger
from indexer.parser import normalize_location

logger = get_logger("api.retriever")


class CVRetriever:
    """Vector search over indexed CV chunks, grouped back into candidates.

    Retrieval is grouped by ``candidate_id`` rather than returning a flat list of
    chunks. A flat top-N chunk search lets one verbose CV occupy many of the N
    slots, which starves the reranker of candidates to compare; grouping
    guarantees N *distinct* candidates while still surfacing each one's best
    matching chunks.
    """

    def __init__(self, client: QdrantClient = None):
        # Config is read here rather than at module import so that callers (and
        # tests) can set the environment before constructing the retriever.
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "resumes")

        # How many distinct candidates to pull back for reranking. RETRIEVAL_TOP_N
        # is the historical name for this knob and is still honoured.
        self.retrieval_candidates = int(
            os.getenv("RETRIEVAL_CANDIDATES", os.getenv("RETRIEVAL_TOP_N", "30"))
        )
        # Best-matching chunks kept per candidate to build its summary.
        self.chunks_per_candidate = int(os.getenv("CHUNKS_PER_CANDIDATE", "3"))

        # QDRANT_PATH runs Qdrant embedded from a local directory, so the whole
        # service works with no server and no Docker. Embedded mode takes an
        # exclusive file lock, so the indexer and the API cannot hold it at the
        # same time -- index first, then start the API. Use host/port for
        # anything beyond a local demo.
        self.path = os.getenv("QDRANT_PATH")

        if client is not None:
            self.client = client
        elif self.path:
            logger.info(f"Opening embedded Qdrant at {self.path}")
            self.client = QdrantClient(path=self.path)
        else:
            logger.info(f"Connecting CVRetriever to Qdrant at {self.host}:{self.port}")
            self.client = connect_qdrant(
                self.host, self.port,
                client_factory=lambda: QdrantClient(host=self.host, port=self.port),
            )

    def _build_filter(self, filters: ScreeningFilters):
        """Translate API filters into a Qdrant filter, or None if unfiltered."""
        conditions = []
        if filters:
            if filters.min_experience is not None:
                conditions.append(
                    models.FieldCondition(
                        key="years_of_experience",
                        range=models.Range(gte=filters.min_experience),
                    )
                )
            if filters.location:
                # Normalized through the same helper the indexer used when writing
                # the payload, so alias spellings ("Bengaluru") match "Bangalore".
                conditions.append(
                    models.FieldCondition(
                        key="location",
                        match=models.MatchValue(value=normalize_location(filters.location)),
                    )
                )
        return models.Filter(must=conditions) if conditions else None

    def search_candidates(
        self,
        query_vector: list[float],
        filters: ScreeningFilters,
        top_n: int = None,
    ) -> list[dict]:
        """Return up to ``top_n`` distinct candidates ranked by best chunk score."""
        if top_n is None:
            top_n = self.retrieval_candidates

        query_filter = self._build_filter(filters)

        try:
            logger.info(
                f"Searching '{self.collection_name}' for {top_n} distinct candidates "
                f"({self.chunks_per_candidate} chunks each)..."
            )
            response = self.client.query_points_groups(
                collection_name=self.collection_name,
                query=query_vector,
                group_by="candidate_id",
                limit=top_n,
                group_size=self.chunks_per_candidate,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}")
            raise

        candidates = []
        for group in response.groups:
            hits = [h for h in group.hits if h.payload]
            if not hits:
                continue

            # query_points_groups returns hits ordered best-first within a group.
            best = max(hits, key=lambda h: h.score)
            payload = best.payload

            # Preserve chunk order by relevance, dropping empties and duplicates.
            seen, chunk_texts = set(), []
            for h in sorted(hits, key=lambda h: h.score, reverse=True):
                text = (h.payload or {}).get("chunk_text", "").strip()
                if text and text not in seen:
                    seen.add(text)
                    chunk_texts.append(text)

            candidates.append(
                {
                    "candidate_id": payload.get("candidate_id") or str(group.id),
                    "name": payload.get("name", "Unknown"),
                    "cv_path": payload.get("cv_path", ""),
                    "years_of_experience": payload.get("years_of_experience", 0),
                    "location": payload.get("location", "Unknown"),
                    "score": best.score,
                    "resume_summary": "\n---\n".join(chunk_texts),
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        logger.info(f"Retrieved {len(candidates)} distinct candidates.")
        return candidates
