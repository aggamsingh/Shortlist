"""Stress the RUNNING service: does a slow request freeze it, and is it correct under load?

    python -m evaluation.stress_server blocking      # slow reranker; does /health stall?
    python -m evaluation.stress_server concurrency   # many threads; do results stay correct?

Both build a real index in a temp directory, start a real uvicorn server as a
subprocess and drive it over HTTP. TestClient cannot do this: it gives every call
its own event loop, so it passes against exactly the bug these exist to catch.

blocking     The reranker sleeps SLOW seconds. While one /screen is in flight, time
             /health, then fire N concurrent /screen requests. If the handlers block
             the event loop, /health waits for the search and the N requests run one
             after another. Found in this project: /health took 2.4s behind a 3s
             search and six requests took 18.3s (fully serial) until the handlers
             stopped being `async def`.

concurrency  Real embedder, real cross-encoder, real Qdrant, real SQLite. A serial
             baseline for a set of job descriptions, then many concurrent requests,
             each compared with its baseline; /health while the pool is saturated;
             and concurrent decision writes to one screening. Moving handlers onto
             threads makes the embedder, the embedded Qdrant client and torch
             inference concurrent for the first time, so "no errors" is a claim that
             needed testing rather than assuming.

Throughput is whatever this machine's CPU gives. The point is correctness and
responsiveness, not an absolute number.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = 8765
KEY = "stress-key"


def prepare(backend: str, pool: int):
    root = Path(tempfile.mkdtemp(prefix="shortlist_stress_"))
    env = dict(os.environ)
    env.update({
        "CV_FOLDER_PATH": str(root / "cvs"), "QDRANT_PATH": str(root / "qdrant"),
        "STATE_FILE_PATH": str(root / "state.json"), "BM25_STATE_PATH": str(root / "bm25.json"),
        "METADATA_CACHE_PATH": str(root / "meta.json"), "STORE_PATH": str(root / "db.sqlite"),
        "QDRANT_COLLECTION": "stress", "RERANKER_BACKEND": backend,
        "METADATA_LLM_FALLBACK": "false", "API_KEY": KEY, "PYTHONPATH": ".",
        # Explicit, so a developer's own .env cannot silently change what is measured.
        "RETRIEVAL_CANDIDATES": str(pool),
    })
    for key in ("QDRANT_HOST", "QDRANT_PORT"):
        env.pop(key, None)
    for args in (["-m", "evaluation.export_corpus", "--out", str(root / "cvs")],
                 ["-m", "indexer.run"]):
        result = subprocess.run([sys.executable] + args, env=env, capture_output=True, text=True)
        if result.returncode:
            sys.exit(result.stderr[-800:])
    return root, env


def call(path, body=None, method=None, timeout=120):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method or ("POST" if body is not None else "GET"),
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null"), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()[:200], time.perf_counter() - started
    except Exception as error:  # connection reset, timeout, ...
        return str(error)[:60], None, time.perf_counter() - started


def wait_healthy(server, log_path=None):
    for _ in range(120):
        if server.poll() is not None:
            sys.exit("server exited early" + (f"\n{Path(log_path).read_text()[-800:]}" if log_path else ""))
        if call("/health", timeout=2)[0] == 200:
            return
        time.sleep(1)
    sys.exit("server never became healthy")


def blocking(slow: float, n: int) -> None:
    root, env = prepare("none", 10)
    env["SLOW_SECONDS"] = str(slow)
    launcher = root / "serve.py"
    launcher.write_text(
        "import os, time, uvicorn\n"
        "import api.main as m\n"
        "orig = m.NoOpReranker.rerank\n"
        "def slow(self, *a, **k):\n"
        "    time.sleep(float(os.environ['SLOW_SECONDS']))\n"
        "    return orig(self, *a, **k)\n"
        "m.NoOpReranker.rerank = slow\n"
        f"uvicorn.run(m.app, host='127.0.0.1', port={PORT}, log_level='warning')\n"
    )
    server = subprocess.Popen([sys.executable, str(launcher)], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_healthy(server)
        body = {"job_description": "Python backend engineer with FastAPI and vector search", "top_k": 3}
        print(f"reranker sleeps {slow}s per request")
        print(f"idle /health:                      {call('/health')[2]:.3f}s")
        status, _, latency = call("/api/v1/screen", body)
        print(f"single /screen:                    {latency:.2f}s (status {status})")

        holder = {}
        thread = threading.Thread(target=lambda: holder.update(r=call("/api/v1/screen", body)))
        thread.start()
        time.sleep(0.7)
        status, _, latency = call("/health", timeout=30)
        thread.join()
        print(f"/health DURING a slow /screen:     {latency:.2f}s (status {status}; compose timeout is 5s)")

        results = []
        workers = [threading.Thread(target=lambda: results.append(call("/api/v1/screen", body, timeout=300)))
                   for _ in range(n)]
        started = time.perf_counter()
        [w.start() for w in workers]
        [w.join() for w in workers]
        wall = time.perf_counter() - started
        ok = sum(1 for status, _, _ in results if status == 200)
        print(f"{n} concurrent /screen:               {ok} ok in {wall:.1f}s "
              f"(fully serial would be ~{n * slow:.0f}s, parallel ~{slow:.0f}s)")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


def concurrency(threads: int, requests: int, pool: int) -> None:
    root, env = prepare("cross-encoder", pool)
    print(f"candidate pool: {pool}")
    log = open(root / "server.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--log-level", "warning"],
        env=env, stdout=log, stderr=log)
    try:
        wait_healthy(server, root / "server.log")
        sys.path.insert(0, ".")
        from evaluation.corpus import select_queries
        jds = [q["job_description"] for q in select_queries("all")][:24]

        baseline, started = {}, time.perf_counter()
        for jd in jds:
            status, body, _ = call("/api/v1/screen", {"job_description": jd, "top_k": 5})
            assert status == 200, (status, body)
            baseline[jd] = [c["candidate_id"] for c in body["candidates"]]
        serial_wall = time.perf_counter() - started
        print(f"serial:      {len(jds)} requests in {serial_wall:.1f}s ({serial_wall / len(jds):.2f}s each)")

        rng = random.Random(7)
        work = [rng.choice(jds) for _ in range(requests)]
        latencies, errors, mismatches = [], [], 0

        def one(jd):
            status, body, took = call("/api/v1/screen", {"job_description": jd, "top_k": 5})
            return jd, status, body, took

        started = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            for jd, status, body, took in pool.map(one, work):
                latencies.append(took)
                if status != 200:
                    errors.append((status, str(body)[:120]))
                elif [c["candidate_id"] for c in body["candidates"]] != baseline[jd]:
                    mismatches += 1
        wall = time.perf_counter() - started
        latencies.sort()
        print(f"concurrent:  {requests} requests, {threads} threads, {wall:.1f}s, "
              f"{requests / wall:.2f} req/s (serial {len(jds) / serial_wall:.2f} req/s)")
        print(f"latency:     p50 {latencies[len(latencies) // 2]:.2f}s  "
              f"p95 {latencies[int(len(latencies) * .95) - 1]:.2f}s  max {latencies[-1]:.2f}s")
        print(f"errors: {len(errors)}   results differing from the serial baseline: {mismatches}")

        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                call("/api/v1/screen", {"job_description": jds[0], "top_k": 5})

        hammers = [threading.Thread(target=hammer) for _ in range(threads)]
        [h.start() for h in hammers]
        time.sleep(2)
        health = [call("/health", timeout=30)[2] for _ in range(5)]
        stop.set()
        [h.join() for h in hammers]
        print(f"/health with the pool saturated: max {max(health):.2f}s (compose timeout 5s)")

        status, body, _ = call("/api/v1/screen", {"job_description": jds[0], "top_k": 10})
        job, candidates = body["job_id"], [c["candidate_id"] for c in body["candidates"]]

        def decide(i):
            return call(
                f"/api/v1/screenings/{job}/candidates/{candidates[i % len(candidates)]}/decision",
                {"decision": random.choice(["shortlisted", "rejected", "maybe"]), "note": f"n{i}"},
                method="PUT")[0]

        with ThreadPoolExecutor(threads) as pool:
            codes = list(pool.map(decide, range(200)))
        _, stored, _ = call(f"/api/v1/screenings/{job}")
        decided = sum(1 for c in stored["candidates"] if c["decision"] != "undecided")
        print(f"200 concurrent decision writes: {sum(c != 200 for c in codes)} failed, "
              f"{decided} of {len(candidates)} candidates have a stored decision")

        log.flush()
        bad = [line for line in (root / "server.log").read_text().splitlines()
               if any(w in line for w in ("Already borrowed", "Traceback", "RuntimeError", "database is locked"))]
        print(f"server-side errors in the log: {len(bad)}")
        for line in bad[:3]:
            print("   ", line[:160])
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("blocking")
    b.add_argument("--slow", type=float, default=3.0, help="seconds the reranker sleeps")
    b.add_argument("-n", type=int, default=6, help="concurrent requests")
    c = sub.add_parser("concurrency")
    c.add_argument("--threads", type=int, default=12)
    c.add_argument("--requests", type=int, default=120)
    c.add_argument("--pool", type=int, default=10, help="candidates retrieved for reranking")
    args = parser.parse_args()
    if args.mode == "blocking":
        blocking(args.slow, args.n)
    else:
        concurrency(args.threads, args.requests, args.pool)


if __name__ == "__main__":
    main()
