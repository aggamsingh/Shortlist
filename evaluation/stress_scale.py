"""How does the service scale with corpus size? Real indexer, real retriever.

    python -m evaluation.stress_scale 500 2000 6000

For each N: build a synthetic corpus of N CVs by varying the 32 labelled ones (new
names, cities, years), then measure, through the real code paths:

    full index     the first run
    no-change      a second run with nothing modified
    +1 file        one new CV added, the case a single upload would hit
    query          embed and retrieve latency over 30 job descriptions
    disk           Qdrant directory and BM25 state size

TIMING ONLY. The corpus is the same 32 documents repeated with different metadata,
so retrieval QUALITY at scale is not measured, and this says nothing about it.

Embedded Qdrant is a brute-force scan in a single process, so these are the numbers
for the default zero-setup mode, not for a Qdrant server with an HNSW index.
"""

import copy
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evaluation.corpus import CANDIDATES, render_cv, select_queries
from evaluation.export_corpus import write_docx

FIRST = ("Aarav Vivaan Aditya Arjun Reyansh Ishaan Anaya Diya Myra Saanvi Kabir Rohan "
         "Meera Tara Nikhil Kavya Rahul Neha Pooja Sanjay").split()
LAST = ("Sharma Verma Gupta Singh Patel Reddy Nair Iyer Menon Joshi Kapoor Malhotra "
        "Bose Das Rao Mehta Shah Khan").split()
CITIES = ["Bangalore", "Pune", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Noida", "Kolkata"]


def build_corpus(n: int, out: Path) -> None:
    rng = random.Random(1)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        candidate = copy.deepcopy(CANDIDATES[i % len(CANDIDATES)])
        candidate["name"] = f"{rng.choice(FIRST)} {rng.choice(LAST)} {i}"
        candidate["location"] = rng.choice(CITIES)
        candidate["years"] = rng.randint(1, 15)
        write_docx(out / f"cv_{i:05d}.docx", render_cv(candidate))


def run_indexer(env: dict) -> tuple:
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "indexer.run"], env=env,
                            capture_output=True, text=True)
    return time.perf_counter() - started, result.returncode


def measure(n: int) -> None:
    root = Path(tempfile.mkdtemp(prefix=f"shortlist_scale{n}_"))
    env = dict(os.environ)
    env.update({
        "CV_FOLDER_PATH": str(root / "cvs"), "QDRANT_PATH": str(root / "qdrant"),
        "STATE_FILE_PATH": str(root / "state.json"), "BM25_STATE_PATH": str(root / "bm25.json"),
        "METADATA_CACHE_PATH": str(root / "meta.json"), "QDRANT_COLLECTION": "scale",
        "METADATA_LLM_FALLBACK": "false", "PYTHONPATH": ".",
    })
    for key in ("QDRANT_HOST", "QDRANT_PORT"):
        env.pop(key, None)

    started = time.perf_counter()
    build_corpus(n, root / "cvs")
    generated = time.perf_counter() - started

    full, rc1 = run_indexer(env)
    unchanged, rc2 = run_indexer(env)
    write_docx(root / "cvs" / "cv_new.docx",
               render_cv(CANDIDATES[0]).replace("Ananya Rao", "Zed New Person"))
    one_new, rc3 = run_indexer(env)

    os.environ.update(env)
    from api.retriever import CVRetriever
    from indexer.embedder import CVEmbedder

    embedder, retriever = CVEmbedder(), CVRetriever()
    jds = [q["job_description"] for q in select_queries("all")][:30]
    retriever.search_candidates(embedder.embed_text(jds[0]), None, query_text=jds[0])  # warm-up
    embed_ms, retrieve_ms = [], []
    for jd in jds:
        started = time.perf_counter()
        vector = embedder.embed_text(jd)
        embed_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        retriever.search_candidates(vector, None, query_text=jd)
        retrieve_ms.append((time.perf_counter() - started) * 1000)
    retrieve_ms.sort()
    retriever.client.close()

    qdrant_mb = sum(f.stat().st_size for f in (root / "qdrant").rglob("*") if f.is_file()) / 1e6
    bm25_mb = (root / "bm25.json").stat().st_size / 1e6
    print(
        f"N={n:<5} generate {generated:4.0f}s | index: full {full:6.1f}s  no-change "
        f"{unchanged:6.1f}s  +1 file {one_new:6.1f}s (exit {rc1}{rc2}{rc3}) | query: embed "
        f"{statistics.median(embed_ms):4.0f}ms  retrieve p50 {statistics.median(retrieve_ms):6.1f}  "
        f"p95 {retrieve_ms[int(len(retrieve_ms) * .95) - 1]:6.1f} ms | qdrant {qdrant_mb:5.1f}MB  "
        f"bm25 {bm25_mb:4.2f}MB",
        flush=True,
    )


def main() -> None:
    sizes = [int(arg) for arg in sys.argv[1:]] or [500, 2000]
    for n in sizes:
        measure(n)


if __name__ == "__main__":
    main()
