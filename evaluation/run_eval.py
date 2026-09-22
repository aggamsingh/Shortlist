"""Retrieval evaluation harness.

Builds a throwaway Qdrant index from the synthetic corpus, runs every labelled
job description through the pipeline and reports ranking quality.

    python -m evaluation.run_eval               # headline numbers
    python -m evaluation.run_eval --ablations   # compare design choices
    python -m evaluation.run_eval --rerank      # include the LLM reranker

Runs against an embedded Qdrant (no server needed), so it is reproducible on a
laptop and in CI. Note that embedded Qdrant ignores payload indexes; those matter
for latency on a real server, not for the ranking quality measured here.
"""

import argparse
import shutil
import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from evaluation.corpus import CANDIDATES, HARD_QUERIES, QUERIES, corpus_stats, render_cv
from evaluation.metrics import aggregate, evaluate_query, recall_at_k
from indexer.embedder import CVEmbedder
from indexer.parser import chunk_by_words, chunk_cv, clean_text

COLLECTION = "eval_resumes"


# ---------------------------------------------------------------- chunkers

def chunk_sections(text: str) -> list:
    """The shipped strategy: split on resume headings, sub-chunk long sections."""
    return chunk_cv(text)


def chunk_window(text: str) -> list:
    """Baseline: fixed sliding window, ignoring document structure."""
    return chunk_by_words(clean_text(text), chunk_size=200, overlap=50)


def chunk_whole(text: str) -> list:
    """Baseline: one vector per CV, no chunking at all."""
    return [clean_text(text)]


def chunk_window_small(text: str) -> list:
    """Baseline: a window small enough to actually split a short CV.

    The production 200-word window never splits these fixtures (a CV here is
    ~110 words), so `window` and `whole` would otherwise be the same run.
    """
    return chunk_by_words(clean_text(text), chunk_size=60, overlap=15)


CHUNKERS = {
    "section": chunk_sections,
    "window": chunk_window,
    "window60": chunk_window_small,
    "whole": chunk_whole,
}


# ---------------------------------------------------------------- indexing

def build_index(client: QdrantClient, embedder: CVEmbedder, chunker) -> int:
    """(Re)build the eval collection using the given chunking strategy."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=embedder.dimension, distance=models.Distance.COSINE
        ),
    )

    points, total_chunks = [], 0
    for candidate in CANDIDATES:
        text = render_cv(candidate)
        chunks = chunker(text)
        total_chunks += len(chunks)
        vectors = embedder.embed_texts(chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                models.PointStruct(
                    # Deterministic ids keep reruns comparable.
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{candidate['id']}-{i}")),
                    vector=vector,
                    payload={
                        "candidate_id": candidate["id"],
                        "name": candidate["name"],
                        "cv_path": f"/eval/{candidate['id']}.pdf",
                        "chunk_text": chunk,
                        "years_of_experience": candidate["years"],
                        "location": candidate["location"],
                    },
                )
            )
    client.upsert(collection_name=COLLECTION, points=points)
    return total_chunks


# ---------------------------------------------------------------- retrieval

def retrieve_grouped(client, vector, budget: int) -> list:
    """Shipped strategy: ask Qdrant for `budget` DISTINCT candidates."""
    response = client.query_points_groups(
        collection_name=COLLECTION,
        query=vector,
        group_by="candidate_id",
        limit=budget,
        group_size=3,
        with_payload=True,
    )
    ranked = []
    for group in response.groups:
        best = max(group.hits, key=lambda h: h.score)
        ranked.append((best.payload["candidate_id"], best.score))
    ranked.sort(key=lambda x: -x[1])
    return ranked


def retrieve_flat(client, vector, budget: int) -> list:
    """Prior strategy: take `budget` top CHUNKS, then collapse to candidates.

    Kept as a baseline because it is what the service did before grouping, and
    it makes the cost of chunk-level budgeting measurable rather than asserted.
    """
    hits = client.query_points(
        collection_name=COLLECTION, query=vector, limit=budget, with_payload=True
    ).points
    best_by_candidate = {}
    for hit in hits:
        cid = hit.payload["candidate_id"]
        if cid not in best_by_candidate or hit.score > best_by_candidate[cid]:
            best_by_candidate[cid] = hit.score
    ranked = sorted(best_by_candidate.items(), key=lambda x: -x[1])
    return ranked


RETRIEVERS = {"grouped": retrieve_grouped, "flat": retrieve_flat}


# ---------------------------------------------------------------- evaluation

def evaluate_config(client, embedder, retriever_name: str, budget: int, k: int,
                    reranker=None, queries=None) -> dict:
    """Run every query under one configuration and average the metrics."""
    retrieve = RETRIEVERS[retriever_name]
    queries = QUERIES if queries is None else queries
    per_query, distinct_counts = [], []

    for query in queries:
        vector = embedder.embed_text(query["job_description"])
        ranked = retrieve(client, vector, budget)
        distinct_counts.append(len(ranked))
        ranked_ids = [cid for cid, _ in ranked]
        pool_ids = list(ranked_ids)  # the set handed to the reranker

        if reranker is not None:
            candidates = [
                {
                    "candidate_id": cid,
                    "name": cid,
                    "cv_path": "",
                    "score": score,
                    "resume_summary": render_cv(
                        next(c for c in CANDIDATES if c["id"] == cid)
                    ),
                }
                for cid, score in ranked
            ]
            reranked = reranker.rerank(query["job_description"], candidates, top_k=len(candidates))
            ranked_ids = [r["candidate_id"] for r in reranked]

        metrics = evaluate_query(ranked_ids, query["relevance"], k=k)
        # Recall over the ENTIRE retrieved pool, not just the top k. In a two
        # stage system this is the ceiling on final quality: a reranker can
        # reorder what retrieval handed it, but it can never recover a relevant
        # candidate that retrieval dropped. This is the number that justifies
        # spending retrieval budget on distinct candidates.
        metrics["pool_recall"] = recall_at_k(
            pool_ids, query["relevance"], k=len(pool_ids)
        )
        per_query.append(metrics)

    result = aggregate(per_query)
    result["avg_candidates_retrieved"] = sum(distinct_counts) / len(distinct_counts)
    return result


def format_row(label: str, metrics: dict, k: int) -> str:
    return (
        f"  {label:<34} "
        f"{metrics[f'recall@{k}']:.3f}   "
        f"{metrics[f'precision@{k}']:.3f}      "
        f"{metrics[f'ndcg@{k}']:.3f}   "
        f"{metrics['mrr']:.3f}  "
        f"{metrics['pool_recall']:.3f}      "
        f"{metrics['avg_candidates_retrieved']:.1f}"
    )


def header(k: int) -> str:
    return (
        f"  {'configuration':<34} {f'recall@{k}':<8} {f'prec@{k}':<10} "
        f"{f'nDCG@{k}':<8} {'MRR':<6} {'pool_rec':<10} {'cands'}\n"
        f"  {'-' * 92}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume retrieval quality.")
    parser.add_argument("--ablations", action="store_true", help="compare design choices")
    parser.add_argument("--rerank", action="store_true", help="include the LLM reranker")
    parser.add_argument("--k", type=int, default=5, help="cutoff for @k metrics")
    parser.add_argument("--budget", type=int, default=10, help="retrieval budget")
    args = parser.parse_args()

    stats = corpus_stats()
    print("Resume Screener - retrieval evaluation")
    print(
        f"corpus: {stats['candidates']} CVs, {stats['queries']} job descriptions, "
        f"{stats['distractors']} never-relevant distractors"
    )
    print(f"retrieval budget: {args.budget}   metrics cutoff: k={args.k}\n")

    workdir = Path(tempfile.mkdtemp(prefix="resume_eval_"))
    try:
        client = QdrantClient(path=str(workdir / "qdrant"))
        embedder = CVEmbedder()

        reranker = None
        if args.rerank:
            # .env is loaded here, not at import, so the harness picks up keys
            # without mutating os.environ for anything that merely imports it.
            from dotenv import load_dotenv

            from api.reranker import CVReranker

            load_dotenv(dotenv_path=".env")
            reranker = CVReranker()
            if not reranker.is_configured:
                print("!! --rerank requested but no LLM key configured; skipping rerank.\n")
                reranker = None

        suites = [
            ("CROSS-ROLE queries (different jobs; separable by topic alone)", QUERIES),
            ("WITHIN-ROLE queries (all Python backend; the discriminating set)", HARD_QUERIES),
        ]

        if not args.ablations:
            n_chunks = build_index(client, embedder, CHUNKERS["section"])
            print(f"indexed {n_chunks} chunks (section chunking)\n")
            for title, queries in suites:
                print(title)
                print(header(args.k))
                metrics = evaluate_config(
                    client, embedder, "grouped", args.budget, args.k, reranker,
                    queries=queries,
                )
                label = "section + grouped" + (" + rerank" if reranker else "")
                print(format_row(label, metrics, args.k))
                print()
        else:
            for title, queries in suites:
                print(title)
                print(header(args.k))
                for chunker_name in ("whole", "window", "window60", "section"):
                    n_chunks = build_index(client, embedder, CHUNKERS[chunker_name])
                    for retriever_name in ("flat", "grouped"):
                        metrics = evaluate_config(
                            client, embedder, retriever_name, args.budget, args.k,
                            queries=queries,
                        )
                        print(
                            format_row(
                                f"{chunker_name:<8} + {retriever_name:<8} ({n_chunks:>3} chunks)",
                                metrics,
                                args.k,
                            )
                        )
                if reranker:
                    build_index(client, embedder, CHUNKERS["section"])
                    metrics = evaluate_config(
                        client, embedder, "grouped", args.budget, args.k, reranker,
                        queries=queries,
                    )
                    print(format_row("section  + grouped  + LLM rerank", metrics, args.k))
                print()

        print("cands = mean distinct candidates reaching the reranker (higher is better)")
    finally:
        # Embedded Qdrant holds a file lock; drop the client before cleanup.
        try:
            client.close()
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
