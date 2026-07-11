import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from api.models import ScreeningFilters
from indexer.utils import get_logger

logger = get_logger("api.retriever")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")
RETRIEVAL_TOP_N = int(os.getenv("RETRIEVAL_TOP_N", "30"))

class CVRetriever:
    def __init__(self):
        logger.info(f"Connecting CVRetriever to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION

    def search_candidates(self, query_vector: list[float], filters: ScreeningFilters, top_n: int = None) -> list[dict]:
        """
        Search candidates in Qdrant based on JD vector and filters.
        Deduplicates chunks by candidate, merging the text.
        """
        if top_n is None:
            top_n = RETRIEVAL_TOP_N
            
        # Build filter conditions
        conditions = []
        if filters:
            if filters.min_experience is not None:
                conditions.append(
                    models.FieldCondition(
                        key="years_of_experience",
                        range=models.Range(gte=filters.min_experience)
                    )
                )
            if filters.location:
                # Match location case-insensitively using MatchValue and title-case
                conditions.append(
                    models.FieldCondition(
                        key="location",
                        match=models.MatchValue(value=filters.location.strip().capitalize())
                    )
                )

        query_filter = models.Filter(must=conditions) if conditions else None

        try:
            logger.info(f"Searching Qdrant collection '{self.collection_name}' with limit {top_n}...")
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_n
            )
            logger.info(f"Retrieved {len(search_results)} raw chunk points from Qdrant.")
        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}")
            raise

        # Deduplicate and group by candidate_id
        candidates_map = {}
        for hit in search_results:
            payload = hit.payload
            cand_id = payload.get("candidate_id")
            if not cand_id:
                continue
                
            chunk_text = payload.get("chunk_text", "")
            score = hit.score
            
            if cand_id not in candidates_map:
                candidates_map[cand_id] = {
                    "candidate_id": cand_id,
                    "name": payload.get("name", "Unknown"),
                    "cv_path": payload.get("cv_path", ""),
                    "years_of_experience": payload.get("years_of_experience", 0),
                    "location": payload.get("location", "Unknown"),
                    "max_score": score,
                    "chunks": [chunk_text]
                }
            else:
                if score > candidates_map[cand_id]["max_score"]:
                    candidates_map[cand_id]["max_score"] = score
                if chunk_text not in candidates_map[cand_id]["chunks"]:
                    candidates_map[cand_id]["chunks"].append(chunk_text)

        # Merge candidate texts and format list sorted by max_score descending
        deduplicated_candidates = []
        for cand_id, data in candidates_map.items():
            merged_text = "\n---\n".join(data["chunks"])
            deduplicated_candidates.append({
                "candidate_id": data["candidate_id"],
                "name": data["name"],
                "cv_path": data["cv_path"],
                "years_of_experience": data["years_of_experience"],
                "location": data["location"],
                "score": data["max_score"],
                "resume_summary": merged_text
            })

        # Sort by initial score descending
        deduplicated_candidates.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Deduplicated to {len(deduplicated_candidates)} unique candidates.")
        return deduplicated_candidates
