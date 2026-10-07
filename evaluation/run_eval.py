"""Retrieval evaluation harness.

Builds a throwaway Qdrant index from the synthetic corpus, runs every labelled
job description through the pipeline and reports ranking quality.

    python -m evaluation.run_eval                  # dev set, headline numbers
    python -m evaluation.run_eval --ablations      # dev set, compare design choices
    python -m evaluation.run_eval --rerank         # add the LLM reranker
    python -m evaluation.run_eval --rerank cross   # add the local cross-encoder
    python -m evaluation.run_eval --rerank all     # three-way comparison
    python -m evaluation.run_eval --split test     # the held-out set (read once)

Queries are split into dev and test (see evaluation/corpus.py). Everything
defaults to DEV, because every design decision in this project was made against
the dev queries and a harness that reported held-out numbers by default would
end up tuned against them. `--split test` is for confirming a configuration
already chosen on dev, not for choosing one.

Runs against an embedded Qdrant (no server needed), so it is reproducible on a
laptop and in CI. Note that embedded Qdrant ignores payload indexes; those matter
for latency on a real server, not for the ranking quality measured here.
"""

import argparse
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from evaluation.corpus import CANDIDATES, corpus_stats, render_cv, select_queries
from evaluation.metrics import aggregate, evaluate_query, recall_at_k
from indexer.embedder import CVEmbedder
from indexer.parser import chunk_by_words, chunk_cv, clean_text
from indexer.sparse import BM25Encoder

# Each build gets a FRESH collection name. delete_collection + create_collection
# does not actually purge points in embedded Qdrant: rebuilding with a different
# chunker left the previous chunker's points in place and the new ones were
# upserted on top, so every ablation row after the first silently retrieved over
# a mixture of strategies. Unique names sidestep that entirely.
COLLECTION = "eval_resumes"
_BUILD_COUNTER = 0
SPARSE_VECTOR_NAME = "text"

# Fitted in build_index and reused by the hybrid retriever below.
_BM25 = BM25Encoder()


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


def _sparse(text: str) -> models.SparseVector:
    """BM25 document vector for a chunk."""
    indices, values = _BM25.encode_document(text)
    return models.SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------- indexing

def build_index(client: QdrantClient, embedder: CVEmbedder, chunker) -> int:
    """(Re)build the eval collection using the given chunking strategy."""
    global _BM25, COLLECTION, _BUILD_COUNTER
    _BUILD_COUNTER += 1
    COLLECTION = f"eval_resumes_{_BUILD_COUNTER}"
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=embedder.dimension, distance=models.Distance.COSINE
        ),
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )

    # BM25 is refitted per chunking strategy: document frequency and average
    # length both depend on how the corpus was split.
    chunked = {c["id"]: chunker(render_cv(c)) for c in CANDIDATES}
    _BM25 = BM25Encoder().fit(ch for chunks in chunked.values() for ch in chunks)

    points, total_chunks = [], 0
    for candidate in CANDIDATES:
        text = render_cv(candidate)
        chunks = chunked[candidate["id"]]
        total_chunks += len(chunks)
        vectors = embedder.embed_texts(chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                models.PointStruct(
                    # Deterministic ids keep reruns comparable.
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{candidate['id']}-{i}")),
                    vector={
                        "": vector,
                        SPARSE_VECTOR_NAME: _sparse(chunk),
                    },
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


def retrieve_hybrid(client, vector, budget: int, query_text: str = None):
    """Dense + BM25, fused with Reciprocal Rank Fusion, grouped by candidate.

    Dense embeddings are weak on exact low-frequency technical tokens (Qdrant,
    asyncio, Terraform), which is what the within-role queries turn on. BM25 is
    strong there and weak at paraphrase, so the two are complementary.
    """
    indices, values = _BM25.encode_query(query_text or "")
    if not indices:
        return retrieve_grouped(client, vector, budget)

    prefetch_limit = max(budget * 3, 100)
    response = client.query_points_groups(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=vector, limit=prefetch_limit),
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR_NAME, limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        group_by="candidate_id", limit=budget, group_size=3, with_payload=True,
    )
    ranked = []
    for group in response.groups:
        best = max(group.hits, key=lambda h: h.score)
        ranked.append((best.payload["candidate_id"], best.score))
    ranked.sort(key=lambda x: -x[1])
    return ranked


RETRIEVERS = {
    "grouped": retrieve_grouped,
    "flat": retrieve_flat,
    "hybrid": retrieve_hybrid,
}


# ---------------------------------------------------------------- evaluation

def evaluate_config(client, embedder, retriever_name: str, budget: int, k: int,
                    reranker=None, queries=None, jd_filter=None) -> dict:
    """Run every query under one configuration and average the metrics.

    `jd_filter`, if given, strips boilerplate from the job description before
    retrieval, exactly as the service does. The reranker still receives the
    original text, as it does in the service.
    """
    retrieve = RETRIEVERS[retriever_name]
    if queries is None:
        queries = select_queries("dev", "cross")
    per_query, distinct_counts = [], []
    degraded = 0

    rerank_times = []
    for query in queries:
        search_text = (
            jd_filter.apply(query["job_description"])
            if jd_filter is not None
            else query["job_description"]
        )
        vector = embedder.embed_text(search_text)
        if retriever_name == "hybrid":
            ranked = retrieve(client, vector, budget, search_text)
        else:
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
            # Wall clock, because the point of comparing a cross-encoder to an
            # LLM is the cost of the second stage, not just its quality.
            started = time.perf_counter()
            reranked = reranker.rerank(
                query["job_description"], candidates, top_k=len(candidates)
            )
            rerank_times.append((time.perf_counter() - started) * 1000)
            ranked_ids = [r["candidate_id"] for r in reranked]
            # A rerank that silently degraded to vector scores must not be
            # reported as a reranker result. Every candidate carrying the
            # fallback reasoning means the provider never answered.
            if reranked and all(
                "not scored by reranker" in r.get("match_reasoning", "")
                for r in reranked
            ):
                degraded += 1

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
    result["queries"] = len(queries)
    result["degraded_queries"] = degraded
    result["rerank_ms"] = (
        sum(rerank_times) / len(rerank_times) if rerank_times else 0.0
    )
    return result


def format_row(label: str, metrics: dict, k: int) -> str:
    # Flag runs where the reranker never actually answered, so a degraded run
    # cannot be mistaken for a measurement.
    degraded = metrics.get("degraded_queries", 0)
    if degraded:
        label = f"{label}  !! {degraded}/{metrics.get('queries', '?')} NOT reranked"
    rerank_ms = metrics.get("rerank_ms", 0.0)
    latency = f"{rerank_ms:>8.0f}" if rerank_ms else "       -"
    return (
        f"  {label:<34} "
        f"{metrics[f'recall@{k}']:.3f}   "
        f"{metrics[f'precision@{k}']:.3f}      "
        f"{metrics[f'ndcg@{k}']:.3f}   "
        f"{metrics['mrr']:.3f}  "
        f"{metrics['pool_recall']:.3f}      "
        f"{metrics['avg_candidates_retrieved']:>4.1f} "
        f"{latency}"
    )


def header(k: int) -> str:
    return (
        f"  {'configuration':<34} {f'recall@{k}':<8} {f'prec@{k}':<10} "
        f"{f'nDCG@{k}':<8} {'MRR':<6} {'pool_rec':<10} {'cands':<5} "
        f"{'rerank_ms':>8}\n"
        f"  {'-' * 102}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume retrieval quality.")
    parser.add_argument("--ablations", action="store_true", help="compare design choices")
    parser.add_argument(
        "--rerank",
        nargs="?",
        const="llm",
        choices=("llm", "cross", "all"),
        default=None,
        help="add a reranking stage: llm (default when bare), cross, or all",
    )
    parser.add_argument("--k", type=int, default=5, help="cutoff for @k metrics")
    parser.add_argument("--budget", type=int, default=10, help="retrieval budget")
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="query split to evaluate (default: dev; test is held out)",
    )
    parser.add_argument(
        "--jd-style",
        choices=("plain", "A", "B", "C"),
        default="plain",
        help="wrap each job description in company boilerplate (C is held out: "
             "--split test only)",
    )
    parser.add_argument(
        "--query-style",
        choices=("jd", "title", "keywords"),
        default="jd",
        help="query form: the full job description (default), a short keyword "
             "phrase, or the job title alone (cross-role queries only)",
    )
    parser.add_argument(
        "--jd-filter",
        action="store_true",
        help="strip boilerplate from the job description before retrieval, as the "
             "service does",
    )
    args = parser.parse_args()

    from evaluation.jd_styles import check_allowed, wrap

    try:
        check_allowed(args.jd_style, args.split)
    except ValueError as error:
        parser.error(str(error))
    if args.query_style != "jd" and (args.jd_style != "plain" or args.jd_filter):
        parser.error("--query-style title/keywords cannot be combined with "
                     "--jd-style or --jd-filter: those act on full job descriptions")

    stats = corpus_stats(args.split)
    print("Shortlist - retrieval evaluation")
    print(
        f"corpus: {stats['candidates']} CVs, {stats['queries']} job descriptions, "
        f"{stats['min_distractors_per_query']}+ non-relevant candidates per query"
    )
    print(f"retrieval budget: {args.budget}   metrics cutoff: k={args.k}")
    print(f"split: {args.split.upper()}  ({stats['dev_queries']} dev / "
          f"{stats['test_queries']} held-out test)")
    if args.split == "dev":
        print("note: the shipped configuration was CHOSEN on these queries, so "
              "these numbers are optimistic.")
    else:
        # Said out loud every time, because the whole value of a held-out set is
        # destroyed quietly rather than loudly.
        print("note: HELD-OUT queries. Report, do not tune. If a number here "
              "prompts a change, choose the change on dev and re-measure once.")
    print()

    workdir = Path(tempfile.mkdtemp(prefix="resume_eval_"))
    try:
        client = QdrantClient(path=str(workdir / "qdrant"))
        embedder = CVEmbedder()

        # Ordered (label, reranker) pairs. Retrieval-only is always first so a
        # reranker is read as a delta against it rather than in isolation.
        stages = [("retrieval only", None)]
        if args.rerank in ("cross", "all"):
            from api.cross_encoder import CrossEncoderReranker

            cross = CrossEncoderReranker()
            if cross.is_configured:
                stages.append(("+ cross-encoder", cross))
            else:
                print("!! cross-encoder could not be loaded; skipping it.\n")
        if args.rerank in ("llm", "all"):
            # .env is loaded here, not at import, so the harness picks up keys
            # without mutating os.environ for anything that merely imports it.
            from dotenv import load_dotenv

            from api.reranker import CVReranker

            load_dotenv(dotenv_path=".env")
            llm = CVReranker()
            if llm.is_configured:
                stages.append(("+ LLM rerank", llm))
            else:
                print("!! LLM rerank requested but no key configured; skipping it.\n")

        # Only the retrieval-only row: keep the old single-row output shape.
        reranker = stages[-1][1] if len(stages) > 1 else None

        suites = [
            (
                "CROSS-ROLE queries (different jobs; separable by topic alone)",
                select_queries(args.split, "cross"),
            ),
            (
                "WITHIN-ROLE queries (all Python backend; the discriminating set)",
                select_queries(args.split, "within"),
            ),
        ]
        # Copies, so the labelled corpus is never mutated.
        suites = [
            (
                title,
                [dict(q, job_description=wrap(q["job_description"], args.jd_style))
                 for q in qs],
            )
            for title, qs in suites
        ]

        if args.query_style != "jd":
            from evaluation.short_queries import short_text

            shortened = []
            for title, qs in suites:
                kept = []
                for q in qs:
                    text = short_text(q["id"], args.query_style)
                    if text:
                        kept.append(dict(q, job_description=text))
                if len(kept) < len(qs):
                    print(f"{title.split(' (')[0]}: {len(kept)} of {len(qs)} queries "
                          f"have a '{args.query_style}' form")
                shortened.append((title, kept))
            suites = [(t, q) for t, q in shortened if q]
            print(f"query style: {args.query_style} (a short phrase, not a job description)\n")

        jd_filter = None
        if args.jd_filter:
            from api.jd_filter import JDFilter

            jd_filter = JDFilter(embedder)
        print(
            f"job descriptions: style {args.jd_style}"
            + ("  +  boilerplate filter" if jd_filter else "")
            + "\n"
        )

        if not args.ablations:
            # Must mirror the shipped default (indexer.parser.chunk_resume), or
            # the headline numbers describe a configuration nobody actually runs.
            n_chunks = build_index(client, embedder, CHUNKERS["window"])
            print(f"indexed {n_chunks} chunks (window chunking, the shipped default)\n")
            for title, queries in suites:
                print(title)
                print(header(args.k))
                for stage_label, stage in stages:
                    metrics = evaluate_config(
                        client, embedder, "hybrid", args.budget, args.k, stage,
                        queries=queries, jd_filter=jd_filter,
                    )
                    label = (
                        "window + hybrid"
                        if stage is None
                        else f"window + hybrid {stage_label}"
                    )
                    print(format_row(label, metrics, args.k))
                print()
        else:
            for title, queries in suites:
                print(title)
                print(header(args.k))
                for chunker_name in ("whole", "window", "window60", "section"):
                    n_chunks = build_index(client, embedder, CHUNKERS[chunker_name])
                    for retriever_name in ("flat", "grouped", "hybrid"):
                        metrics = evaluate_config(
                            client, embedder, retriever_name, args.budget, args.k,
                            queries=queries, jd_filter=jd_filter,
                        )
                        print(
                            format_row(
                                f"{chunker_name:<8} + {retriever_name:<8} ({n_chunks:>3} chunks)",
                                metrics,
                                args.k,
                            )
                        )
                if len(stages) > 1:
                    build_index(client, embedder, CHUNKERS["window"])
                    for stage_label, stage in stages[1:]:
                        metrics = evaluate_config(
                            client, embedder, "hybrid", args.budget, args.k, stage,
                            queries=queries, jd_filter=jd_filter,
                        )
                        print(
                            format_row(
                                f"window   + hybrid   {stage_label}", metrics, args.k
                            )
                        )
                print()

        print("cands = mean distinct candidates reaching the reranker (higher is better)")
        print("rerank_ms = mean wall-clock cost of the reranking stage per query")
    finally:
        # Embedded Qdrant holds a file lock; drop the client before cleanup.
        try:
            client.close()
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
