"""Persistence for screening runs and recruiter decisions.

Why SQLite
----------
A screening run is small structured data that has to survive a restart and be
queried by id. Qdrant is a vector index, not a record store, and adding Postgres
for a handful of rows would mean another service to deploy for no benefit.
SQLite is in the standard library, needs no server, and the write pattern here
is one row per search -- nowhere near its limits.

What is stored
--------------
A run keeps the job description, the filters, the returned candidates and the
timings, so a shortlist can be reopened or shared without re-running the search
(which would cost another LLM call and could return a different order).

Decisions are kept separately from the run that produced them: a recruiter
revises a shortlist repeatedly, and the search results themselves are immutable
evidence of what the system returned at the time.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from indexer.utils import get_logger

logger = get_logger("api.store")

DECISION_VALUES = ("shortlisted", "rejected", "maybe", "undecided")

SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    job_id          TEXT PRIMARY KEY,
    job_description TEXT NOT NULL,
    filters         TEXT,
    candidates      TEXT NOT NULL,
    timings         TEXT,
    reranked        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    job_id       TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    decision     TEXT NOT NULL,
    note         TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id)
);

CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ScreeningStore:
    """Thread-safe SQLite store for screening runs and decisions."""

    def __init__(self, path: str = None):
        import os

        self.path = path or os.getenv("STORE_PATH", "./data/shortlist.db")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because uvicorn serves requests from a thread
        # pool; a single lock around writes is simpler than a connection pool and
        # adequate at this write volume.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        logger.info(f"Screening store ready at {self.path}")

    # ---- screenings ----

    def save_screening(self, job_id: str, job_description: str, filters,
                       candidates: list, timings: dict, reranked: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO screenings "
                "(job_id, job_description, filters, candidates, timings, reranked, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    str(job_id),
                    job_description,
                    json.dumps(filters) if filters else None,
                    json.dumps(candidates),
                    json.dumps(timings or {}),
                    1 if reranked else 0,
                    _now(),
                ),
            )
            self._conn.commit()

    def get_screening(self, job_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM screenings WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        if not row:
            return None
        return {
            "job_id": row["job_id"],
            "job_description": row["job_description"],
            "filters": json.loads(row["filters"]) if row["filters"] else None,
            "candidates": json.loads(row["candidates"]),
            "timings": json.loads(row["timings"]) if row["timings"] else {},
            "reranked": bool(row["reranked"]),
            "created_at": row["created_at"],
        }

    def list_screenings(self, limit: int = 20, offset: int = 0) -> list:
        """Recent runs, newest first. Candidate lists are summarised, not returned."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id, job_description, created_at, reranked, candidates "
                "FROM screenings ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            counts = dict(
                self._conn.execute(
                    "SELECT job_id, COUNT(*) FROM decisions "
                    "WHERE decision = 'shortlisted' GROUP BY job_id"
                ).fetchall()
            )
        out = []
        for row in rows:
            candidates = json.loads(row["candidates"])
            jd = row["job_description"]
            out.append({
                "job_id": row["job_id"],
                # A full JD is paragraphs long; a history list needs a label.
                "job_description_preview": (jd[:120] + "...") if len(jd) > 120 else jd,
                "candidate_count": len(candidates),
                "shortlisted_count": counts.get(row["job_id"], 0),
                "reranked": bool(row["reranked"]),
                "created_at": row["created_at"],
            })
        return out

    def count_screenings(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]

    # ---- decisions ----

    def set_decision(self, job_id: str, candidate_id: str, decision: str,
                     note: str = None) -> None:
        if decision not in DECISION_VALUES:
            raise ValueError(f"decision must be one of {DECISION_VALUES}")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions "
                "(job_id, candidate_id, decision, note, updated_at) VALUES (?,?,?,?,?)",
                (str(job_id), candidate_id, decision, note, _now()),
            )
            self._conn.commit()

    def get_decisions(self, job_id: str) -> dict:
        """candidate_id -> {decision, note, updated_at} for one run."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT candidate_id, decision, note, updated_at FROM decisions "
                "WHERE job_id = ?", (str(job_id),)
            ).fetchall()
        return {
            r["candidate_id"]: {
                "decision": r["decision"],
                "note": r["note"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
