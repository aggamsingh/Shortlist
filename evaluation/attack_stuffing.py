"""Keyword-stuffing attack against the real retriever and the serving path.

    python -m evaluation.attack_stuffing dev
    python -m evaluation.attack_stuffing test     # held-out queries; run once

For each query, one clearly irrelevant candidate adds a chunk to their CV, and we
measure where they land. Two attacks:

    V1  the job description pasted in verbatim (the lazy, common version)
    V2  its words shuffled into a keyword list, so no phrase survives

each with the defence (api/stuffing.py) off and on. Also counts false flags on
CLEAN queries, because a defence that flags honest candidates is its own failure.

Each cell is "in the top 5 / ranked first", out of the number of queries, first for
retrieval alone and then after the local cross-encoder, using the same path
POST /screen uses: retrieve, then rerank what retrieval returned.

The test split exists to be read once. Do not tune the defence against it.
"""

import argparse
import os
import random
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from api.cross_encoder import CrossEncoderReranker
from api.retriever import CVRetriever
from evaluation import run_eval
from evaluation.corpus import CANDIDATES, select_queries
from indexer.embedder import CVEmbedder


def soup(job_description: str, rng: random.Random) -> str:
    words = sorted({w for w in re.findall(r"\w+", job_description.lower()) if len(w) >= 3})
    rng.shuffle(words)
    return " ".join(words)


def rank_of(ids: list, candidate_id: str):
    return ids.index(candidate_id) + 1 if candidate_id in ids else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("split", choices=("dev", "test"))
    split = parser.parse_args().split

    by_id = {c["id"]: c for c in CANDIDATES}
    workdir = Path(tempfile.mkdtemp(prefix="resume_attack_"))
    client = QdrantClient(path=str(workdir / "qdrant"))
    try:
        embedder = CVEmbedder()
        run_eval.build_index(client, embedder, run_eval.CHUNKERS["window"])
        cross = CrossEncoderReranker()

        retriever = CVRetriever(client=client)
        retriever.collection_name = run_eval.COLLECTION
        retriever.hybrid_enabled = True
        retriever.bm25 = run_eval._BM25

        def serve(jd, victim=None, text=None):
            point = None
            if victim:
                indices, values = run_eval._BM25.encode_document(text)
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "": embedder.embed_text(text),
                        run_eval.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=indices, values=values
                        ),
                    },
                    payload={
                        "candidate_id": victim, "name": by_id[victim]["name"],
                        "cv_path": "", "chunk_text": text,
                        "years_of_experience": by_id[victim]["years"],
                        "location": by_id[victim]["location"],
                    },
                )
                client.upsert(run_eval.COLLECTION, points=[point])
            try:
                candidates = retriever.search_candidates(
                    embedder.embed_text(jd), None, query_text=jd
                )
                reranked = cross.rerank(jd, candidates, top_k=len(candidates))
                return (
                    [c["candidate_id"] for c in candidates],
                    [r["candidate_id"] for r in reranked],
                    candidates,
                )
            finally:
                if point:
                    client.delete(
                        run_eval.COLLECTION,
                        points_selector=models.PointIdsList(points=[point.id]),
                    )

        queries = select_queries(split, "within") + select_queries(split, "cross")
        rng = random.Random(11)
        victims = {
            q["id"]: rng.choice([c for c in by_id if c not in q["relevance"]])
            for q in queries
        }
        n = len(queries)
        flagged_either = lambda c: c.get("possible_stuffing") or c.get("suspicious_match")

        os.environ["STUFFING_DEFENSE"] = "true"
        false_flags = sum(
            sum(1 for c in serve(q["job_description"])[2] if flagged_either(c))
            for q in queries
        )
        print(f"[{split}] {n} queries | false flags on CLEAN queries: {false_flags}")
        print(f"[{split}] one irrelevant candidate attacks each query    "
              f"in top-5 / ranked first (of {n})")

        for attack in ("none", "V1 verbatim copy", "V2 keyword soup"):
            for defence in ("off", "on"):
                if attack == "none" and defence == "on":
                    continue
                os.environ["STUFFING_DEFENSE"] = "true" if defence == "on" else "false"
                r5 = r1 = c5 = c1 = flagged = 0
                attack_rng = random.Random(3)
                for q in queries:
                    jd, victim = q["job_description"], victims[q["id"]]
                    if attack == "none":
                        ret, crs, cands = serve(jd)
                    else:
                        text = jd if attack.startswith("V1") else soup(jd, attack_rng)
                        ret, crs, cands = serve(jd, victim, text)
                    rv, cv = rank_of(ret, victim), rank_of(crs, victim)
                    r5 += rv is not None and rv <= 5
                    r1 += rv == 1
                    c5 += cv is not None and cv <= 5
                    c1 += cv == 1
                    flagged += any(
                        flagged_either(c) and c["candidate_id"] == victim for c in cands
                    )
                print(
                    f"  {attack:17} defence {defence:3}  retrieval {r5:2}/{r1:2}   "
                    f"+cross-encoder {c5:2}/{c1:2}   flagged {flagged}"
                )
    finally:
        try:
            client.close()
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
