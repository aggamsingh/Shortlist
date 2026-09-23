# Shortlist

> Two-stage resume screening: vector retrieval + LLM reranking, with a labelled evaluation harness (nDCG, recall@k, MRR).

A CPU-only microservice that takes a job description and returns the best-matching candidates from a corpus of CVs, each with a relevance score and a one-line justification.

It is a two-stage retrieval system: fast vector search over chunked resumes narrows thousands of CVs to a shortlist, then an LLM reranks that shortlist against the full job description. Retrieval quality is measured, not asserted — see [Evaluation](#evaluation).

```
CVs (PDF/DOCX) ──> parse ──> chunk ──> embed ──> Qdrant
                                                   │
job description ──> embed ──────> vector search ───┤ grouped by candidate
                                                   ▼
                                          shortlist (N distinct people)
                                                   │
                                                   ▼
                                          LLM rerank (Gemini / Groq)
                                                   │
                                                   ▼
                                       scored candidates + reasoning
```

---

## Quick start

No Docker and no database server required — Qdrant can run embedded from a local directory.

```bash
pip install -r requirements.txt
cp .env.example .env          # set API_KEY; LLM keys are optional

export QDRANT_PATH=./data/qdrant     # embedded mode
export CV_FOLDER_PATH=./cvs          # drop some .pdf / .docx CVs here

python -m indexer.run                # build the index
python -m uvicorn api.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"job_description":"Backend engineer with Python, FastAPI and vector search","top_k":5}'
```

Embedded Qdrant takes an **exclusive file lock**, so the indexer and the API cannot run at the same time: index first, then start the API. For concurrent use, run a Qdrant server and set `QDRANT_HOST` / `QDRANT_PORT` instead.

### With Docker

```bash
docker compose up -d          # Qdrant + API
docker compose run indexer    # index whatever is in CV_FOLDER_PATH
```

---

## How it works

### Indexing (`indexer/`)

1. **Incremental scan** — each file is MD5-hashed and compared against `data/index_state.json`. Unchanged CVs are skipped, so re-indexing a large corpus after adding ten CVs costs ten CVs of work.
2. **Parsing** — `pypdf` for PDFs, `python-docx` for DOCX *including table cells*, since a lot of resumes put work history in tables that a paragraph-only reader silently drops.
3. **Metadata extraction** — years of experience by regex, location against a known-city list, candidate name from the filename. This is the weakest part of the system and is discussed honestly under [Limitations](#limitations).
4. **Chunking** — split on resume section headings (`Experience`, `Skills`, …), with a 200-word sliding window as fallback for CVs with no recognisable structure.
5. **Embedding** — `all-MiniLM-L6-v2` (384-dim), pinned to CPU. Small, fast, and good enough that the reranker does the hard discrimination.
6. **Upsert** — deterministic UUID5 point ids derived from the candidate id, so re-indexing an edited CV replaces its vectors instead of accumulating duplicates. Stale vectors are deleted first.

### Serving (`api/`)

`POST /api/v1/screen` embeds the JD, applies hard filters in Qdrant, retrieves a shortlist, reranks, and returns the top `k`. `GET /health` reports Qdrant connectivity, model load, and whether an LLM is actually configured.

---

## Design decisions

### Retrieval is grouped by candidate, not by chunk

A naive top-N chunk search returns *chunks*, and one verbose CV can occupy many of the N slots. The reranker then gets a handful of people instead of N, and a candidate that vector search ranked 8th never gets the chance to be promoted.

Qdrant's `query_points_groups` groups hits by `candidate_id`, guaranteeing N **distinct** candidates while still surfacing each one's best-matching chunks. On the within-role query set, this raises the share of relevant candidates that reach the reranker from **0.617 to 0.822** at the same retrieval budget. That number is the ceiling on final quality: a reranker can reorder what it is given, but it can never recover a candidate retrieval dropped.

### Retrieval is hybrid: dense embeddings fused with BM25

The evaluation pointed here. Once reranking was measured, within-role recall@5
reached the pool recall exactly — the reranker was already promoting *every*
relevant candidate it received, so the remaining loss was entirely candidates
retrieval never surfaced. Pool recall of 0.822 meant ~18% were missing before
ranking even began.

Embeddings are weak on exact, low-frequency technical tokens — "Qdrant",
"Pinecone", "asyncio", "Terraform" — which is precisely what the within-role
queries turn on. BM25 is strong there and weak at paraphrase, so the two are
complementary. Both branches are queried and fused with Qdrant's native
Reciprocal Rank Fusion, still grouped by candidate.

Measured effect on the within-role set: **pool recall 0.822 → 0.933**, and
nDCG@5 0.566 → 0.702 on whole-CV chunking. A regression test asserts the sparse
branch still surfaces an exact term match that dense search ranks last.

Document weights carry term-frequency saturation and query weights carry IDF, so
their dot product is the BM25 score. Token ids come from CRC32 rather than a
stored vocabulary — Python's `hash()` is salted per process and would silently
produce a different index on every run.

The cost is honest: BM25 needs corpus-wide statistics, so the indexer re-parses
every CV on each run. Parsing is cheap next to embedding, which still runs only
for changed files, and unchanged CVs get their sparse vectors refreshed in place
without re-embedding. `HYBRID_RETRIEVAL=false` restores the dense-only path.

### The reranker is treated as an untrusted input

LLM output is the least reliable input in the system, so `api/reranker.py` assumes it will be malformed:

- **Identity is never taken from the model.** Scores and reasoning come from the LLM; `candidate_id`, `name` and `cv_path` always come from the index, so a hallucinated candidate cannot reach the response.
- **Score scale is inferred across the batch, not per value.** Models asked for `0.00–1.00` sometimes answer `0–10` or `0–100`. Judging each value alone would map a lone `1.5` to `0.015` and rank the model's *best* pick last; judging the batch together preserves the model's ordering, which is the part of its answer that actually matters.
- **Reranked candidates are ordered ahead of fallbacks.** An LLM score and a cosine similarity are different units, so interleaving them by raw value would let an unjudged `0.6` cosine outrank a judged `0.55`.
- **A provider outage degrades, it does not 500.** With no key or a failed call, the service returns vector-similarity ordering and says so in `match_reasoning`.
- **Prompt size is capped** per candidate and per request, because concatenated chunks are otherwise unbounded input to a metered API.
- **Rate limits are retried; quotas are not.** A per-minute cap clears in seconds, so it is worth waiting out with backoff. An exhausted daily quota reports a wait of minutes to hours, and sleeping through a short backoff for that only delays the inevitable fallback — so when a provider asks for longer than the retry ceiling, the request degrades immediately. A retired model id or a bad key never retries at all.

### Readiness is handled in the application, not by a compose healthcheck

The database and API start together, so the first connection races container startup. `connect_qdrant()` retries with backoff and verifies a real round trip. This also survives a Qdrant restart while the API is already running — something a startup-only healthcheck never sees.

### Configuration is read per request, not at import

Binding config at import time makes behaviour depend on whether `.env` loaded before the module was first imported. That bug was live here: `api/retriever.py` read `QDRANT_HOST` at import while `load_dotenv()` ran afterwards, so those `.env` values were silently ignored outside Docker.

---

## Evaluation

Retrieval quality is measured against a labelled synthetic corpus: **32 CVs, 8 job descriptions, graded relevance** (2 = would shortlist, 1 = plausible, 0 = not a fit).

```bash
python -m evaluation.run_eval --ablations
```

Two query sets, because they measure very different things:

- **Cross-role** — five different jobs (backend, frontend, ML, DevOps, data). Any embedding model separates a React CV from a Kubernetes CV, so these saturate near the ceiling.
- **Within-role** — three jobs where *every* strong candidate is a Python backend engineer and only one specific requirement separates them (production vector-database experience; deep asyncio; owning your own infra). Keyword overlap on "Python", "backend", "API" is near-uniform here and carries no signal. **This is the set that discriminates.**

A quarter of the corpus is deliberate distractors: a QA engineer whose CV is dense with Python, a technical writer who documents FastAPI, a Java engineer whose resume says "backend microservices Docker". A keyword matcher scores well without them.

### Results (retrieval only, k=5, budget=10)

| configuration | recall@5 | nDCG@5 | pool recall |
|---|---|---|---|
| **Cross-role** (8 queries) | | | |
| whole CV + grouped | 0.919 | 0.954 | 0.950 |
| whole CV + hybrid | 0.919 | **0.967** | 0.919 |
| window + grouped | 0.919 | 0.951 | 0.950 |
| **window + hybrid** | 0.919 | 0.958 | 0.950 |
| section + grouped | 0.894 | 0.902 | 0.975 |
| section + hybrid | 0.894 | 0.927 | 0.975 |
| **Within-role** (7 queries) | | | |
| whole CV + flat | 0.648 | 0.677 | 0.900 |
| whole CV + hybrid | 0.795 | 0.803 | 0.871 |
| window + flat | 0.648 | 0.677 | 0.900 |
| window + grouped | 0.648 | 0.677 | 0.900 |
| **window + hybrid** | **0.814** | **0.833** | 0.900 |
| window60 + hybrid | 0.743 | 0.756 | 0.924 |
| section + flat | 0.638 | 0.647 | 0.807 |
| section + grouped | 0.638 | 0.647 | 0.867 |
| section + hybrid | 0.726 | 0.761 | 0.867 |

**What this actually shows:**

1. **Hybrid retrieval is the largest single win.** On the within-role set it lifts nDCG@5 from 0.677 to **0.833** and recall@5 from 0.648 to **0.814**. Exact technical tokens are where dense embeddings are weakest and BM25 is strongest.

2. **Grouping's value depends on how many chunks each candidate has.** With section chunking (~9 chunks per CV) it lifts pool recall 0.807 → 0.867, because a few verbose CVs otherwise monopolise the budget. With the shipped window chunking (2 chunks per CV) flat retrieval already returns nearly distinct candidates, and grouping changes the metrics not at all — it only guarantees the property rather than leaving it to luck. An earlier, much larger figure for this came from the short-fixture corpus and no longer holds.
2. **The cross-role set is nearly useless as a benchmark.** It sits at 0.92 nDCG for almost every configuration. Reporting only these numbers would make the system look better than it is.
3. **Retrieval still loses ~10% of relevant candidates.** Pool recall on the within-role set is 0.900, so about one relevant candidate in ten never reaches the reranker and can never be recovered. An earlier draft of this README claimed 1.000; that number was an artifact of the evaluation bug described below, and is corrected here.

4. **Section chunking lost, and the default changed because of it.** Heading-based sectioning was the original design and seemed principled. On the first corpus it was indistinguishable from a sliding window, because 55-word fixtures chunk identically under any strategy. Once the corpus was rebuilt at realistic length with varied layouts — including CVs with no headings at all — the window won consistently: within-role nDCG **0.833 vs 0.761**, cross-role **0.958 vs 0.927**, across every retrieval mode. The shipped default is now the window (`CHUNK_STRATEGY=section` restores the old behaviour). This is the one conclusion in the project that reversed under better data, which is the whole reason the corpus was rebuilt.

### The benchmark had a bug that flattered it

Worth recording, because it is the failure mode evaluation harnesses are most
prone to: **the measuring instrument was wrong, and it was wrong in the
direction that made the results look better.**

`build_index` rebuilt the collection between ablation rows with
`delete_collection` followed by `create_collection`. In embedded Qdrant that
does **not** purge the points. Each new chunking strategy was upserted on top of
the previous one, so every row after the first was retrieving over a growing
mixture of all strategies — and more chunks meant more chances to match, which
inflated pool recall to a spurious 1.000.

It surfaced only because the same configuration scored differently depending on
what had been built before it. Each build now uses a fresh collection name, and
two regression tests pin it: one asserts a rebuilt collection contains exactly
its own points, the other asserts an identical configuration scores identically
regardless of what ran first.

Every number in this README was re-measured after the fix. The headline
conclusions — hybrid wins, window beats section — survived; the specific figures
did not, and the corrected ones are above.

**Caveat, stated plainly:** 3 within-role queries over 32 CVs is still a small sample, and differences of 0.03 nDCG remain inside the noise a single query could produce. The chunking and hybrid conclusions are held with more confidence than that margin because they are consistent in direction across every retrieval mode and both query sets, not because any single figure is decisive. The fixtures are now 255–304 words with three layout variants — realistic enough for chunking strategies to differ, still shorter than a real 400–800 word CV.

### Reranker evaluation — measured

Every number in the table above is LLM-free. Adding the reranker on top of
`section + grouped` gives, over **5 runs** at `temperature=0`
(Groq, `openai/gpt-oss-120b`):

These figures were taken with `section + grouped` retrieval, before hybrid
search was added.

**Within-role queries — the discriminating set**

| metric | retrieval only | + LLM rerank | change |
|---|---|---|---|
| recall@5 | 0.494 | 0.739 – 0.822 (mean 0.805) | **+0.31** |
| precision@5 | 0.467 | 0.733 – 0.800 (mean 0.787) | **+0.32** |
| nDCG@5 | 0.605 | 0.855 – 0.886 (mean 0.877) | **+0.27** |
| MRR | 0.833 | 1.000 (all 5 runs) | +0.17 |
| pool recall | 0.822 | 0.822 (unchanged) | — |

**Cross-role queries**

| metric | retrieval only | + LLM rerank | change |
|---|---|---|---|
| recall@5 | 0.830 | 0.763 (all 5 runs) | **−0.07** |
| precision@5 | 0.560 | 0.520 (all 5 runs) | **−0.04** |
| nDCG@5 | 0.799 | 0.887 – 0.899 (mean 0.895) | +0.10 |
| MRR | 0.800 | 1.000 (all 5 runs) | +0.20 |

**What this shows:**

1. **Reranking is the single largest quality lever on the hard set** — +0.27 nDCG,
   and MRR reaches a perfect 1.000 in every run, meaning the top result was
   always relevant. The embedding model does see the discriminating requirement;
   it just does not weight it heavily enough, and the LLM does.
2. **Retrieval is now the bottleneck, not ranking.** Within-role recall@5 reaches
   0.822 in 4 of 5 runs — exactly the pool recall. The reranker pulled *every*
   relevant candidate retrieval handed it into the top 5. Further gains have to
   come from retrieving better, not ranking better, which is precisely what the
   grouped-retrieval change was for.
3. **Cross-role recall and precision get worse.** This is a real trade-off, not a
   bug. The reranker promotes strong (grade-2) matches and pushes partial
   (grade-1) ones out of the top 5. Graded nDCG rewards that; binary recall@5
   (which counts grade ≥ 1) penalises it. Both numbers are true — they measure
   different things, and a system tuned for "shortlist the best" will look worse
   on "find everyone plausible".
4. **`temperature=0` is not fully deterministic.** Hosted inference still varies
   run to run, which is why ranges over 5 runs are reported rather than a single
   figure. Within-role nDCG moved ±0.03; cross-role recall and precision were
   identical in all 5.

**On the current corpus and the shipped configuration** (window + hybrid, 15
queries), one mostly-clean run measured within-role nDCG@5 **0.865** against a
retrieval-only baseline of **0.833** — a lift of about **+0.03**.

That is far smaller than the +0.27 measured earlier, and the reason is the
point: **the reranker's value depends on how weak retrieval is.** When retrieval
scored 0.605, the LLM had a great deal to fix. Now that hybrid retrieval scores
0.833, most of what the reranker used to contribute has already been done
upstream, more cheaply and without an API call. Improving retrieval did not just
raise the ceiling — it ate the reranker's margin.

Treat this as one observation, not a range. 13 of 15 queries were reranked in
that run; the other two hit the daily token quota and fell back. A full rerank
pass costs roughly 55,000 tokens, so the Groq free tier's 200,000/day allows
about three runs. Repeating this on a fresh quota is the top open item.

**Measured latency** (3 live API requests, 3 candidates each):

| stage | time |
|---|---|
| embed | 12 – 51 ms |
| retrieve | 1.3 – 2.6 ms |
| **rerank** | **1505 – 2113 ms** |
| total | 1.5 – 2.2 s |

The LLM call is ~97% of request time. Retrieval is effectively free; any latency
work belongs at the reranking stage (batching, a smaller model, or reranking
only the top slice of the shortlist).

### Reproducing this

Add a key to `.env` and verify it first:

```bash
python -m evaluation.check_llm      # ONE real call; confirms key + model id work
```

It sends three candidates whose correct ranking is unambiguous — a retrieval
engineer with production vector-database experience, a Django developer, and a
frontend developer who is given the *highest* vector score on purpose. A working
reranker must put the retrieval engineer first and the frontend developer last,
overriding the vector order. It reports exactly what came back, so a retired
model id or a mis-scoped key shows up immediately rather than as a silent empty
rerank ten minutes into an evaluation.

Then:

```bash
python -m evaluation.run_eval --ablations --rerank
```

Model ids get retired: the previous default, `llama-3.3-70b-versatile`, returned
a 404 on a current Groq account. `check_llm` reports that distinctly from an
invalid key, so you can tell the two apart immediately.

---

## Testing

```bash
python -m unittest discover -s tests -t . -v      # 134 tests
```

Qdrant runs embedded, so the end-to-end tests need no server and run in CI.

The original repository had 32 passing tests and a service that could not index a single CV: `indexer/run.py` called `chunk_cv` without importing it, and every test mocked the boundary so nothing ever executed `main()`. The suite now drives the real entry point with real `.docx` files and real vectors, and covers:

- **Pipeline** — indexing, incremental hash skipping, re-indexing an edited CV without leaving stale vectors, corrupt files not aborting a run.
- **Reranker** — hallucinated ids, duplicate ids, omitted candidates, null scores, wrong numeric scales, non-JSON responses, provider outage.
- **Metrics** — every ranking metric checked against hand-computed values, because the quality claims above rest on them.
- **API** — auth, request validation, response contract.
- **Eval harness** — including the `--rerank` branch, run against a stubbed provider so it cannot rot while no API key is available. Stubs prove plumbing only; no stub-derived number is ever reported as a quality result.

Test isolation is verified by running the suite forwards, backwards, and one module at a time. That check found a real bug: two modules patched `sentence_transformers` globally and `CVEmbedder` is a singleton, so a mocked model leaked into the end-to-end tests.

---

## Configuration

Every value lives in `.env` — see `.env.example` for the annotated list. The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | — | Shared key for `X-API-Key`. **If unset, `/screen` is unauthenticated.** |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | — | At least one enables reranking. Gemini is tried first, Groq is the fallback. |
| `QDRANT_PATH` | — | Run Qdrant embedded from this directory (no server). |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server, when not embedded. |
| `RETRIEVAL_CANDIDATES` | `30` | Distinct **candidates** retrieved for reranking (not chunks). |
| `MAX_CHARS_PER_CANDIDATE` | `1200` | Cap on resume text per candidate sent to the LLM. |

---

## Limitations

Known and deliberate, rather than hidden:

- **Metadata extraction is naive.** Years of experience is a regex over phrases like "5 years of experience"; a CV that only lists date ranges yields `0`. Location matches a hardcoded list of eleven Indian cities. Candidate name comes from the *filename*. This is a cost decision — an LLM extraction pass per CV is accurate but expensive at 200k resumes — and the right design is a cheap regex fast path with an LLM fallback only when it fails. That fallback is not built yet.
- **Location filtering is exact-match on a guessed city.** No geocoding, no radius, no remote handling.
- **The evaluation corpus is synthetic and small.** Real resumes cannot be committed to a public repo, but these fixtures are cleaner and more uniformly structured than real CVs, so the absolute numbers are optimistic. The *relative* comparisons between configurations are the useful part.
- **No OCR.** Scanned image-only PDFs extract no text and are skipped with a warning.
- **Reranker latency dominates and is not optimised.** Measured: embed 12–51 ms, retrieve 1.3–2.6 ms, rerank 1505–2113 ms — about 97% of request time, for only 3 candidates. It has not been measured with a full 30-candidate shortlist, and there is no batching, caching or timeout tuning.
- **The Docker image build is unverified.** The Docker daemon was unavailable during development, so `docker compose up` has not been executed end-to-end. Everything else here was run for real.
- **Model ids drift.** The Groq default is `openai/gpt-oss-120b`, verified working; the previous `llama-3.3-70b-versatile` now 404s. The Gemini default `gemini-2.5-flash` is **unverified** — no Gemini key was tested. Both are configurable via `GEMINI_MODEL` / `GROQ_MODEL`.
- **Results are from one model on a small corpus.** The reranker numbers come from a single provider (Groq `openai/gpt-oss-120b`) over 15 queries and 32 synthetic CVs, and the +0.03 lift rests on one mostly-clean run. They show the pipeline works and that retrieval quality erodes the reranker's margin; they are not a general claim about reranking.

## Next steps

In rough priority order:

1. Add the LLM metadata-extraction fallback for CVs where the regex finds nothing. This is the weakest remaining component and the one an interviewer would probe first.
2. Repeat the hybrid + reranker measurement across several runs, so the +0.03 lift is a range rather than a single observation.
3. Verify the Docker image build; the daemon was never available during development.
4. Grow the query set further. 15 labelled queries is enough to separate 0.07 nDCG but not 0.03.
