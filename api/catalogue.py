"""Browsing the indexed candidate pool, independent of any search.

A recruiter needs to see who is in the pool, open one person's record, and read
their actual CV -- none of which is a similarity search. Those reads come from
Qdrant's stored payloads rather than a second database, because the payload
already holds every field involved and keeping a parallel copy in sync would be
a bug waiting to happen.

Chunks are collapsed back to one record per candidate here, the same way the
retriever does it, since the index is chunk-level and a recruiter thinks in
people.
"""

import os
from pathlib import Path

from indexer.utils import get_logger

logger = get_logger("api.catalogue")


class CandidateCatalogue:
    """Read-only views over the indexed candidates."""

    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        # Serving files straight from a client-supplied path would be a directory
        # traversal hole, so every resolved path is checked against this root.
        self.cv_root = Path(os.getenv("CV_FOLDER_PATH", "./cvs")).resolve()

    def _scroll_all(self, scroll_filter=None, batch: int = 512):
        """Yield every point, following Qdrant's pagination cursor."""
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=batch,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                yield point
            if offset is None:
                break

    def _collapse(self, points) -> dict:
        """Group chunk-level points into one record per candidate."""
        people = {}
        for point in points:
            payload = point.payload or {}
            cid = payload.get("candidate_id")
            if not cid:
                continue
            record = people.setdefault(cid, {
                "candidate_id": cid,
                "name": payload.get("name", "Unknown"),
                "location": payload.get("location", "Unknown"),
                "years_of_experience": payload.get("years_of_experience", 0),
                "cv_path": payload.get("cv_path", ""),
                "chunk_count": 0,
                "_chunks": [],
            })
            record["chunk_count"] += 1
            text = payload.get("chunk_text", "")
            if text:
                record["_chunks"].append(text)
        return people

    def list_candidates(self, limit: int = 50, offset: int = 0, location: str = None,
                        min_experience: int = None, name_contains: str = None) -> dict:
        """Paginated roster. Filters are applied after collapsing to candidates."""
        people = self._collapse(self._scroll_all())

        rows = []
        for record in people.values():
            if location and record["location"].lower() != location.strip().lower():
                continue
            if min_experience is not None and record["years_of_experience"] < min_experience:
                continue
            if name_contains and name_contains.lower() not in record["name"].lower():
                continue
            rows.append({k: v for k, v in record.items() if k != "_chunks"})

        # Stable, predictable ordering for a browsable list.
        rows.sort(key=lambda r: r["name"])
        total = len(rows)
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "candidates": rows[offset: offset + limit],
        }

    def get_candidate(self, candidate_id: str):
        """One candidate with their full indexed text, or None."""
        from qdrant_client.http import models

        points = list(self._scroll_all(
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="candidate_id",
                    match=models.MatchValue(value=candidate_id),
                )
            ])
        ))
        if not points:
            return None
        record = self._collapse(points)[candidate_id]
        chunks = record.pop("_chunks")
        record["indexed_text"] = "\n\n".join(chunks)
        return record

    def resolve_cv_file(self, candidate_id: str):
        """Absolute path to a candidate's CV, or None if unavailable.

        The stored path is validated against CV_FOLDER_PATH before being served.
        A payload is only as trustworthy as whatever wrote it, and serving an
        arbitrary path from an HTTP handler is how directory traversal happens.
        """
        record = self.get_candidate(candidate_id)
        if not record or not record.get("cv_path"):
            return None

        try:
            resolved = Path(record["cv_path"]).resolve()
        except (OSError, ValueError):
            return None

        try:
            resolved.relative_to(self.cv_root)
        except ValueError:
            # Common and legitimate: the index was built in Docker, where paths
            # are /app/cvs/..., and is now served from the host. Fall back to
            # matching the filename inside the configured CV root.
            candidate_file = self.cv_root / Path(record["cv_path"]).name
            if candidate_file.is_file():
                return candidate_file
            logger.warning(
                f"CV path for {candidate_id} resolves outside CV_FOLDER_PATH "
                f"and no matching file was found in it; refusing to serve."
            )
            return None

        return resolved if resolved.is_file() else None

    def stats(self) -> dict:
        """Pool summary: how many people, chunks, and the spread of locations."""
        people = self._collapse(self._scroll_all())
        locations, experience = {}, []
        chunks = 0
        for record in people.values():
            locations[record["location"]] = locations.get(record["location"], 0) + 1
            experience.append(record["years_of_experience"])
            chunks += record["chunk_count"]
        return {
            "candidates": len(people),
            "chunks": chunks,
            "locations": dict(sorted(locations.items(), key=lambda kv: -kv[1])),
            "experience_years": {
                "min": min(experience) if experience else 0,
                "max": max(experience) if experience else 0,
                "mean": round(sum(experience) / len(experience), 1) if experience else 0,
            },
        }
