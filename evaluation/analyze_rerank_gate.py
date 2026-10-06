"""Can a serving-time signal predict when reranking will hurt?

The motivation: reranking costs latency (and, for the LLM, an API call) on every
request, yet it is flat-to-negative on queries retrieval already gets right. A
gate that skipped it when retrieval was confident would save that cost. This
script tests whether such a signal exists, per query, before anything is built.

    python -m evaluation.analyze_rerank_gate

DEV QUERIES ONLY, and deliberately without a --split flag. Choosing a gate is a
design decision, and design decisions are made on dev (see evaluation/corpus.py).

Uses the local cross-encoder, because it is free and deterministic. Per-query
LLM deltas would cost ~4k tokens each, so whether the same conclusion holds for
the LLM reranker is NOT established by this script.

A signal is only a candidate if it is available when a request arrives, i.e. it
is computed from retrieval output and not from relevance labels. Baseline nDCG
is reported for comparison but cannot be used: it needs labels.
"""

import shutil
import statistics
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from evaluation import run_eval
from evaluation.corpus import CANDIDATES, render_cv, select_queries
from evaluation.metrics import evaluate_query

K = 5
BUDGET = 10
# Fixed before looking at the results, so the bar cannot drift to fit the data.
ADOPTION_RHO = -0.40

SIGNALS = ("top1", "margin1", "ratio", "cos_top1", "cos_gap", "overlap5", "top1_agree")


def average_ranks(values: list) -> list:
    """Ranks 1..n, with tied values sharing the mean of the ranks they span.

    Breaking ties by position instead would make the correlation depend on input
    order, and this data is full of ties: several queries have a delta of exactly
    zero, and top1_agree is binary.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for position in range(i, j + 1):
            ranks[order[position]] = mean_rank
        i = j + 1
    return ranks


def spearman(xs: list, ys: list) -> float:
    """Spearman rank correlation. 0.0 when either input is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need two equal-length sequences of at least 2 values")
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def collect(client, embedder, reranker) -> list:
    """Per dev query: retrieval-only nDCG, reranked nDCG and candidate signals."""
    by_id = {c["id"]: c for c in CANDIDATES}
    rows = []
    for kind in ("cross", "within"):
        for query in select_queries("dev", kind):
            jd = query["job_description"]
            vector = embedder.embed_text(jd)

            hybrid = run_eval.retrieve_hybrid(client, vector, BUDGET, jd)
            dense = run_eval.retrieve_grouped(client, vector, BUDGET)

            indices, values = run_eval._BM25.encode_query(jd)
            sparse = client.query_points_groups(
                collection_name=run_eval.COLLECTION,
                query=models.SparseVector(indices=indices, values=values),
                using=run_eval.SPARSE_VECTOR_NAME,
                group_by="candidate_id", limit=BUDGET, group_size=1,
                with_payload=True,
            )
            sparse_ids = [g.hits[0].payload["candidate_id"] for g in sparse.groups]

            ids = [c for c, _ in hybrid]
            scores = [s for _, s in hybrid]
            dense_scores = [s for _, s in dense]

            base = evaluate_query(ids, query["relevance"], k=K)[f"ndcg@{K}"]
            candidates = [
                {"candidate_id": c, "name": c, "cv_path": "", "score": s,
                 "resume_summary": render_cv(by_id[c])}
                for c, s in hybrid
            ]
            reranked = reranker.rerank(jd, candidates, top_k=len(candidates))
            after = evaluate_query(
                [r["candidate_id"] for r in reranked], query["relevance"], k=K
            )[f"ndcg@{K}"]

            top5_dense = {c for c, _ in dense[:5]}
            rows.append({
                "id": query["id"],
                "kind": kind,
                "base": base,
                "delta": after - base,
                # RRF scores depend only on rank, so these three say little about
                # absolute confidence. Kept to show that, not because they work.
                "top1": scores[0],
                "margin1": (scores[0] - scores[1]) / scores[0] if scores[0] else 0.0,
                "ratio": scores[-1] / scores[0] if scores[0] else 0.0,
                # Raw cosine similarity IS an absolute quantity.
                "cos_top1": dense_scores[0],
                "cos_gap": dense_scores[0] - dense_scores[min(4, len(dense_scores) - 1)],
                # Do the two retrievers agree? Disagreement is a classic
                # query-difficulty signal.
                "overlap5": len(top5_dense & set(sparse_ids[:5])) / 5,
                "top1_agree": float(bool(dense and sparse_ids and dense[0][0] == sparse_ids[0])),
            })
    return rows


def summarise(rows: list) -> dict:
    out = {"n": len(rows), "groups": {}, "rho": {}}
    for label in ("cross", "within", "all"):
        subset = [r for r in rows if label == "all" or r["kind"] == label]
        deltas = [r["delta"] for r in subset]
        out["groups"][label] = {
            "n": len(subset),
            "mean_delta": statistics.mean(deltas),
            "helped": sum(d > 1e-9 for d in deltas),
            "hurt": sum(d < -1e-9 for d in deltas),
            "same": sum(abs(d) <= 1e-9 for d in deltas),
        }
    deltas = [r["delta"] for r in rows]
    for signal in SIGNALS + ("base",):
        out["rho"][signal] = spearman([r[signal] for r in rows], deltas)
    return out


def main() -> None:
    from api.cross_encoder import CrossEncoderReranker
    from indexer.embedder import CVEmbedder

    workdir = Path(tempfile.mkdtemp(prefix="resume_gate_"))
    client = None
    try:
        client = QdrantClient(path=str(workdir / "qdrant"))
        embedder = CVEmbedder()
        run_eval.build_index(client, embedder, run_eval.CHUNKERS["window"])
        reranker = CrossEncoderReranker()
        if not reranker.is_configured:
            print("cross-encoder could not be loaded; nothing to analyse.")
            return

        rows = collect(client, embedder, reranker)
        result = summarise(rows)

        print("Rerank gate analysis (dev queries, cross-encoder, nDCG@5)\n")
        for label, g in result["groups"].items():
            print(
                f"  {label:7} n={g['n']:>2}  mean delta {g['mean_delta']:+.3f}  "
                f"helped {g['helped']:>2}  hurt {g['hurt']:>2}  unchanged {g['same']:>2}"
            )
        print(
            f"\nSpearman(signal, delta nDCG). A gate needs rho <= {ADOPTION_RHO:+.2f}: "
            "a strongly NEGATIVE value means 'confident -> reranking does not help'.\n"
        )
        for signal in SIGNALS:
            rho = result["rho"][signal]
            verdict = "meets bar" if rho <= ADOPTION_RHO else "no"
            print(f"  {signal:11} rho {rho:+.3f}   {verdict}")
        print(
            f"\n  {'base nDCG':11} rho {result['rho']['base']:+.3f}   "
            "(needs relevance labels; not available at serving time)"
        )
        best = min(SIGNALS, key=lambda s: result["rho"][s])
        print(
            f"\nBest serving-time signal: {best} (rho {result['rho'][best]:+.3f}). "
            f"{len(SIGNALS)} signals were compared on {result['n']} queries, so even "
            "a modest value here needs a multiple-comparison discount."
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
