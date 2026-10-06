# Shortlist

[![CI](https://github.com/aggamsingh/Shortlist/actions/workflows/ci.yml/badge.svg)](https://github.com/aggamsingh/Shortlist/actions/workflows/ci.yml)

> Two-stage resume screening: hybrid vector + keyword retrieval, then LLM
> reranking — with a labelled evaluation harness measuring whether any of it
> actually works.

Give it a job description, get back ranked candidates with a one-line reason for
each. Runs on CPU, with no GPU and no external database service required.

**What makes this more than a RAG demo:** every design decision here was
measured rather than assumed, and the measurements repeatedly disagreed with the
design. Heading-based chunking lost to a plain sliding window. The LLM reranker
turned out to *hurt* on easy queries. The benchmark itself had a bug that
flattered the results. A held-out query set, added late, knocked 0.08 nDCG off
the headline — and then briefly suggested the shipped chunker was the wrong one,
which turned out to be an 8-query fluctuation that a larger dev set reversed.
All of that is documented below, including the numbers that are unflattering and
the one conclusion that was deliberately *not* acted on.

```
CVs (PDF/DOCX) ──> parse ──> chunk ──> embed ─┬─> dense vectors ─┐
                                              └─> BM25 sparse ───┤
                                                                 ▼
                                                              Qdrant
                                                                 │
job description ──> embed ──────────────────> hybrid search ─────┤ RRF fusion,
                                                                 │ grouped by
                                                                 ▼ candidate
                                                    shortlist (N distinct people)
                                                                 │
                                                                 ▼
                                     rerank: LLM | cross-encoder | none
                                                                 │
                                                                 ▼
                                                  scored candidates + reasoning
```

### Headline results

Measured on 32 labelled CVs and 54 labelled job descriptions, split into
**37 dev** and **17 held-out test** queries. The **within-role** set is the one
that matters: every strong candidate there is a Python backend engineer and the
job turns on a single requirement, so keyword overlap carries no signal.

Within-role, k=5, budget=10:

| | dev recall@5 | dev nDCG@5 | **test** recall@5 | **test** nDCG@5 |
|---|---|---|---|---|
| dense vectors only | 0.609 | 0.662 | 0.402 | 0.502 |
| **+ hybrid BM25 retrieval** | **0.719** | **0.803** | 0.594 | 0.725 |
| + local cross-encoder | 0.679 | 0.813 | 0.615 | 0.768 |
| + LLM reranking | — ¹ | — ¹ | **0.692** | **0.847** |

¹ Not measured: Groq's free tier daily quota ran out mid-run and the harness
flagged the row rather than averaging over failures. See
[The missing cell](#the-missing-cell).

Three things to read off this table:

**Hybrid retrieval is the large win, and it generalises.** +0.14 nDCG on dev,
+0.22 on queries that played no part in choosing it, and it wins under every
chunking strategy on both splits. This is the one conclusion the project holds
with real confidence.

**0.725 is the honest within-role figure.** The configuration was selected by
looking at the dev queries, so dev numbers measure the tuning as well as the
system. The held-out set is what the system scores on job descriptions it was
not shaped around — see
[Held-out results](#held-out-results-the-honest-numbers).

**LLM reranking is worth far more than previously thought — on hard queries
only.** +0.122 nDCG on held-out within-role queries, where an earlier
measurement on an easier set had put it at +0.03. On easy queries it is flat,
and a local cross-encoder *hurts* there. Reranker value turns out to be a
property of how much retrieval left on the table, not of the reranker. See
[Reranker evaluation](#reranker-evaluation-three-ways).

### Contents

- [Quick start](#quick-start) — running it in about two minutes, no Docker needed
- [How it works](#how-it-works) — the pipeline, stage by stage
- [Design decisions](#design-decisions) — what was chosen and what it cost
- [Evaluation](#evaluation) — the benchmark, the dev/test split, and the bug in it
- [Testing](#testing) — 286 tests and why the original 32 were worthless
- [Configuration](#configuration) — every environment variable
- [Limitations](#limitations) — what this does not do, stated plainly
- [Next steps](#next-steps)

---

## Try it on the evaluation corpus

The 32 labelled CVs live as Python data in `evaluation/corpus.py`, so the service
cannot read them directly. Export them as real `.docx` files and drive the whole
pipeline by hand:

```bash
python -m evaluation.export_corpus --list   # see the 32 candidates and what each query expects
python -m evaluation.export_corpus          # write them to ./cvs
python -m indexer.run                       # index
python -m uvicorn api.main:app --port 8000
```

Then search them:

```bash
curl -X POST http://localhost:8000/api/v1/screen   -H "X-API-Key: $API_KEY" -H "Content-Type: application/json"   -d '{"job_description":"Senior Python backend engineer. Must have production experience with a vector database and embedding-based semantic retrieval.","top_k":5}'
```

```
0.95  Imran Qureshi      Built production RAG pipeline with Pinecone vector DB
0.93  Ananya Rao         Designed semantic search backed by Qdrant
0.92  Pooja Desai        Implemented semantic search using Weaviate
0.75  Fatima Sheikh      embeddings and FAISS; no explicit vector DB ownership
```

`--messy-filenames` exports them as `cv_final_v2.docx` and similar, which forces
the metadata extractor onto its LLM fallback instead of reading names from
filenames.

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
docker compose up -d              # Qdrant server + API
docker compose run --rm indexer   # index whatever is in CV_FOLDER_PATH
```

Verified end to end: image builds, `torch 2.14.0+cpu` (the CPU wheel, ~2.26 GB
image — the `--index-url` pin works and no CUDA payload is pulled), Qdrant server
and API containers come up healthy, indexing and screening both run, and the API
reports `llm_configured: true` with live reranking.

Two things this surfaced that local runs never could:

- **`QDRANT_PATH` in `.env` silently overrode `QDRANT_HOST`.** `env_file` loads it
  into the containers, so both ran their own embedded database instead of the
  `qdrant` service — and then contended for the same file lock. Compose now
  clears it explicitly.
- **The state file and the database can diverge.** `./data` is bind-mounted and
  survives `docker compose down -v`, so `index_state.json` claimed three CVs were
  indexed while the fresh Qdrant server held zero points. Every file looked
  unchanged, nothing was written, and the run reported success over an empty
  index. The indexer now counts the collection and re-indexes everything if the
  state and the database disagree.

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

| | |
|---|---|
| `POST /api/v1/screen` | Rank candidates against a job description |
| `GET /api/v1/candidates` | Browse the pool, paginated and filterable |
| `GET /api/v1/candidates/stats` | What is indexed: counts, locations, experience spread |
| `GET /api/v1/candidates/{id}` | One candidate, including the text actually indexed |
| `GET /api/v1/candidates/{id}/cv` | **Download the original PDF/DOCX** |
| `GET /api/v1/screenings` | Previous runs, newest first |
| `GET /api/v1/screenings/{job_id}` | Reopen a run with its decisions |
| `PUT /api/v1/screenings/{job_id}/candidates/{id}/decision` | Shortlist / reject / maybe, with a note |
| `GET /api/v1/screenings/{job_id}/shortlist.csv` | Export as CSV |
| `GET /health` | Qdrant connectivity, model load, which reranking backend is live |

Ranking alone is not a usable tool. A recruiter needs to see who is in the pool,
read the actual resume, record a decision, and come back to it tomorrow — so:

- **`cv_path` became openable.** The screening response carries a server-side
  path a client cannot use; `/cv` serves the file. The stored path is validated
  against `CV_FOLDER_PATH` first, because serving a payload-supplied path
  unchecked is a directory traversal hole, and an index built in Docker
  (`/app/cvs/...`) then served from the host falls back to matching by filename.
- **Runs are persisted.** `job_id` used to be generated, logged, and discarded.
  Reopening returns the *stored* results rather than re-running: a second search
  costs another LLM call and can return a different order, so a shortlist would
  not be the thing the recruiter saw.
- **Decisions live separately from results.** A shortlist gets revised
  repeatedly, while the search output is immutable evidence of what the system
  returned at the time.
- **SQLite, via stdlib `sqlite3`.** A run is a handful of small rows; Qdrant is a
  vector index rather than a record store, and adding Postgres would mean
  another service to deploy for no benefit.
- **The reranking stage is swappable.** `api/reranker.py` (LLM),
  `api/cross_encoder.py` (local) and a no-op all satisfy one contract, chosen by
  `RERANKER_BACKEND`. They are interchangeable because they were built to be
  *compared* — the same harness measures all three — and `/health` reports which
  one is live, since an identical request returns a different ordering under
  each and a bug report without that field is not reproducible. A backend that
  was asked for and failed to load degrades to retrieval order and says so
  loudly; it never silently becomes a different backend.

---

## Design decisions

### Retrieval is grouped by candidate, not by chunk

A naive top-N chunk search returns *chunks*, and one verbose CV can occupy many of the N slots. The reranker then gets a handful of people instead of N, and a candidate that vector search ranked 8th never gets the chance to be promoted.

Qdrant's `query_points_groups` groups hits by `candidate_id`, guaranteeing N
**distinct** candidates while still surfacing each one's best-matching chunks.

**How much this is worth depends entirely on chunks per candidate**, and being
precise about that matters more than the headline. With section chunking (~9
chunks per CV) it lifts pool recall from **0.807 to 0.867**, because a few
verbose CVs would otherwise monopolise the budget. With the window chunking this
project ships (2 chunks per CV) flat retrieval already returns nearly distinct
candidates, and grouping changes the metrics **not at all** — it converts a
property that happened to hold into one that is guaranteed.

An earlier version of this README claimed a much larger gain (0.617 → 0.822).
That came from the original short-fixture corpus and does not survive on
realistic-length CVs.

Pool recall matters because it is the ceiling on final quality: a reranker can
reorder what it is handed, but can never recover a candidate retrieval dropped.

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

Measured effect on the within-role set, dense-only versus hybrid at the same
retrieval budget, on both splits:

| | dense only | hybrid | |
|---|---|---|---|
| **dev** (17 queries) | | | |
| recall@5 | 0.609 | **0.719** | +0.11 |
| nDCG@5 | 0.662 | **0.803** | +0.14 |
| pool recall | 0.854 | 0.869 | +0.02 |
| **held-out** (8 queries) | | | |
| recall@5 | 0.402 | **0.594** | +0.19 |
| nDCG@5 | 0.502 | **0.725** | +0.22 |
| pool recall | 0.783 | 0.829 | +0.05 |

The barely-changed pool recall is the informative part. **Hybrid retrieval is
not finding many different candidates — it is ordering much the same pool far
better.** Both strategies surface a similar fraction of relevant people within a
budget of 10; BM25
decides which of them land in the top 5. A regression test asserts the sparse
branch still promotes an exact term match that dense search ranks last.

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
- **Oversized shortlists are split, not truncated.** Providers reject a request above their per-request token cap outright (HTTP 413), and retrying cannot help because the request is deterministically too large. Capping the candidate count instead would silently discard people retrieval worked to find, so the shortlist is batched to fit `LLM_TOKEN_BUDGET` and every candidate still gets judged. The trade-off is that scores are only strictly comparable within a batch, which is why fewer, larger batches are preferred. Found by running 32 real CVs: 30 candidates at 1200 chars built an ~8300-token prompt against an 8000-token cap.
- **Rate limits are retried; quotas are not.** A per-minute cap clears in seconds, so it is worth waiting out with backoff. An exhausted daily quota reports a wait of minutes to hours, and sleeping through a short backoff for that only delays the inevitable fallback — so when a provider asks for longer than the retry ceiling, the request degrades immediately. A retired model id or a bad key never retries at all.

### Metadata extraction: cheap path first, LLM only on failure

Regex extraction is fast and free but brittle. Years of experience only matched
phrases like "5 years of experience", so a CV listing date ranges (`2018 - 2024`)
yielded 0. Location matched eleven hardcoded cities, so anyone elsewhere was
`Unknown`. The candidate's name came from the *filename*, which is wrong the
moment the file is called `cv_final_v2.pdf`.

Using an LLM for all of it would fix that and cost a call per CV — real money and
real latency for a job regex already does correctly most of the time. So the LLM
is asked only about the fields regex could not resolve, and only about those
fields: a CV missing just its location never sends a prompt mentioning name or
experience. Results are cached by content hash, so re-indexing never pays twice.

**The expensive path is therefore proportional to the failure rate of the cheap
one, not to corpus size.** On a two-CV demo where one resume defeats every regex:

```
Metadata: 1 regex-only, 0 cached, 1 LLM calls, 0 LLM failures (1/2 CVs needed the expensive path)
  cv_final_v2.docx  -> name "Priyanka Deshmukh" (from the CV body, not the filename)
                       9 years (inferred from 2015-2024 date ranges)
                       "Nagpur" (a city absent from the hardcoded list)
```

Re-running the same corpus afterwards costs **zero** calls.

Model output is not trusted: a returned name must pass a character allowlist and
a job-title check (`"Senior Backend Engineer @ Acme!!"` is rejected — a plain
letter-ratio test accepts it, since it is 90% letters), and years outside 0–60
are discarded. Anything rejected falls back to the regex answer. One hard
provider failure disables the fallback for the rest of the run, so an exhausted
quota costs one failed call rather than one per remaining CV.

### Readiness is handled in the application, not by a compose healthcheck

The database and API start together, so the first connection races container startup. `connect_qdrant()` retries with backoff and verifies a real round trip. This also survives a Qdrant restart while the API is already running — something a startup-only healthcheck never sees.

### Configuration is read per request, not at import

Binding config at import time makes behaviour depend on whether `.env` loaded before the module was first imported. That bug was live here: `api/retriever.py` read `QDRANT_HOST` at import while `load_dotenv()` ran afterwards, so those `.env` values were silently ignored outside Docker.

---

## Evaluation

Retrieval quality is measured against a labelled synthetic corpus: **32 CVs, 54 job descriptions, graded relevance** (2 = would shortlist, 1 = plausible, 0 = not a fit).

```bash
python -m evaluation.run_eval                # dev set, headline numbers
python -m evaluation.run_eval --ablations     # dev set, compare design choices
python -m evaluation.run_eval --split test    # the held-out set
```

**Everything defaults to the dev split.** That is deliberate: a harness that
reported held-out numbers unless told otherwise would be tuned against within a
week, which is exactly how the leakage below happened the first time.

Two query sets, because they measure very different things:

- **Cross-role** (29 queries) — different jobs: backend, frontend, ML, DevOps, data engineering, QA, technical writing, cloud architecture, platform engineering, accessibility, web performance, design systems, streaming, analytics engineering, monitoring, recommendations, LLM products, leadership. Any embedding model separates a React CV from a Kubernetes CV, so these saturate near the ceiling.
- **Within-role** (25 queries) — jobs where *every* strong candidate is a Python backend engineer and only one specific requirement separates them: production vector-database experience, deep asyncio, owning your own infra, Django, query optimisation, lexical search, task queues, caching, ORM migrations, gRPC contracts, instrumentation, load testing, mentoring, MySQL modelling, retrieval evaluation, internal tooling, small-team breadth, testing discipline, payments, WebSockets, consumer scale, legacy migration, hybrid search, batch-plus-light-API, seniority. Keyword overlap on "Python", "backend", "API" is near-uniform here and carries no signal. **This is the set that discriminates.**

A quarter of the corpus is deliberate distractors: a QA engineer whose CV is dense with Python, a technical writer who documents FastAPI, a Java engineer whose resume says "backend microservices Docker". A keyword matcher scores well without them. Two held-out queries deliberately invert dev labels — the ETL developer who was a *partial* match for the dev data-engineering query is the *strong* match for a warehousing query, and the vector-search specialists who win the dev vector query are the wrong answer for a lexical-search query. A system that has memorised "data means Spark" or "search means embeddings" gets both backwards.

### The dev/test split

Originally there were 15 queries, and all 15 were used to choose the chunking
strategy, the window size, dense vs hybrid and the retrieval budget — and then
the same 15 were used to report the headline numbers. That is tuning on the test
set. It does not make the pipeline wrong, but it makes every reported figure an
upper bound of unknown tightness.

Partitioning those 15 would not have fixed it: all of them had already
influenced the configuration, so neither half was clean. The fix was to author
17 new queries against the existing candidates **without looking at any
retrieval output**, and hold them back. The dev set was then expanded to 37, so
that design questions too small to settle on 15 queries could be settled there
instead of on the held-out set. They also probe axes the dev set never
touches — seniority, business domain, legacy modernisation, accessibility,
testing discipline, and the *conjunction* of lexical and vector search — so they
test generalisation to new kinds of job description rather than new wordings of
the old ones.

`tests/test_query_splits.py` enforces the parts that fail quietly: the two sets
share no query (by id *and* by object identity, since one dict in both lists
would be stamped with whichever split applied last), the CLI default stays
`dev`, every held-out query has at least one grade-2 answer, and every
within-role query's strong answers stay inside the Python cluster. Those guards
were themselves mutation-tested — each invariant was broken in turn to confirm
the matching test fails.

### Results on the dev set (retrieval only, k=5, budget=10)

37 dev queries. These are the numbers the configuration was **chosen** on, so
read them as an upper bound; the held-out figures follow.

| configuration | recall@5 | nDCG@5 | pool recall |
|---|---|---|---|
| **Cross-role dev** (20 queries) | | | |
| whole CV + grouped | 0.809 | 0.829 | 0.868 |
| whole CV + hybrid | 0.863 | **0.895** | 0.938 |
| window + grouped | 0.772 | 0.798 | 0.876 |
| **window + hybrid** (shipped) | 0.847 | 0.890 | 0.909 |
| window60 + hybrid | 0.863 | 0.870 | 0.930 |
| section + grouped | 0.812 | 0.829 | 0.919 |
| section + hybrid | 0.832 | 0.882 | **0.953** |
| **Within-role dev** (17 queries) | | | |
| whole CV + flat | 0.609 | 0.662 | 0.854 |
| whole CV + hybrid | 0.662 | 0.758 | 0.842 |
| window + flat | 0.609 | 0.662 | 0.854 |
| window + grouped | 0.609 | 0.662 | 0.854 |
| **window + hybrid** (shipped) | **0.719** | **0.803** | 0.869 |
| window60 + flat | 0.543 | 0.593 | 0.751 |
| window60 + hybrid | 0.701 | 0.759 | 0.861 |
| section + flat | 0.551 | 0.557 | 0.750 |
| section + grouped | 0.551 | 0.557 | 0.816 |
| section + hybrid | 0.597 | 0.633 | 0.786 |

**What this actually shows:**

1. **Hybrid retrieval is the largest single win.** On the within-role set it
   lifts nDCG@5 from 0.662 to **0.803** and recall@5 from 0.609 to **0.719**,
   and it is the best choice under every chunker on both query sets. Exact
   technical tokens are where dense embeddings are weakest and BM25 is
   strongest, and the within-role queries turn on exactly those tokens.

2. **Chunk size is settled, and the shipped default wins.** The window scores
   **0.803** within-role against 0.759 for `window60`, 0.758 for whole-CV and
   0.633 for section chunking. This replaces an earlier, weaker version of this
   result — see [How the chunking question was actually
   settled](#how-the-chunking-question-was-actually-settled), which is the most
   instructive sequence in the project.

3. **Grouping's value depends on how many chunks each candidate has.** With
   section chunking (~9 chunks per CV) it lifts pool recall 0.750 → 0.816,
   because a few verbose CVs otherwise monopolise the budget. With the shipped
   window chunking (2 chunks per CV) flat retrieval already returns nearly
   distinct candidates, and grouping barely moves the metrics — it guarantees
   the property rather than leaving it to luck. It is also what keeps
   `cands` at a full 10: under `flat`, section chunking delivers only 7.2
   distinct people to the reranker instead of 10.

4. **The cross-role set is nearly useless as a benchmark.** It sits near 0.88
   nDCG for almost every configuration. Reporting only these numbers would make
   the system look considerably better than it is.

5. **Retrieval still loses ~13% of relevant candidates.** Within-role pool
   recall is 0.869, so roughly one relevant candidate in eight never reaches the
   reranker and can never be recovered. An earlier draft of this README claimed
   1.000; that was an artifact of the evaluation bug described below.


### Held-out results (the honest numbers)

Measured once, with the configuration already fixed on dev. `--split test`.
These numbers have never been used to choose anything.

| configuration | recall@5 | nDCG@5 | pool recall |
|---|---|---|---|
| **Cross-role test** (9 queries) | | | |
| whole CV + hybrid | 0.796 | 0.894 | 0.889 |
| window + flat | 0.694 | 0.829 | 0.861 |
| **window + hybrid** (shipped) | **0.833** | **0.896** | 0.833 |
| window60 + hybrid | 0.741 | 0.838 | 0.833 |
| section + hybrid | 0.759 | 0.850 | 0.889 |
| **Within-role test** (8 queries) | | | |
| whole CV + flat | 0.452 | 0.536 | 0.833 |
| whole CV + hybrid | 0.667 | 0.732 | 0.846 |
| window + flat | 0.402 | 0.502 | 0.783 |
| **window + hybrid** (shipped) | **0.594** | **0.725** | 0.829 |
| window60 + hybrid | 0.588 | 0.775 | 0.798 |
| section + hybrid | 0.558 | 0.652 | 0.723 |

**Dev vs held-out, shipped configuration** (dev = 37 queries, test = 17):

| | dev | test | gap |
|---|---|---|---|
| Cross-role recall@5 | 0.847 | 0.833 | −0.014 |
| Cross-role nDCG@5 | 0.890 | 0.896 | **+0.006** |
| Cross-role pool recall | 0.909 | 0.833 | −0.076 |
| Within-role recall@5 | 0.719 | 0.594 | −0.125 |
| Within-role nDCG@5 | 0.803 | 0.725 | −0.078 |
| Within-role pool recall | 0.869 | 0.829 | −0.040 |

**What the held-out set showed:**

1. **Hybrid retrieval generalises, and is the one result held with real
   confidence.** Within-role nDCG goes 0.502 → 0.725 with BM25 fusion on
   queries that had no part in choosing it: **+0.22**, larger than the +0.14 on
   dev, and hybrid wins under every chunker on both splits. Direction,
   magnitude and consistency all agree.

2. **Within-role performance is genuinely worse than dev suggests.** 0.725
   against 0.803: a −0.078 nDCG gap, and −0.125 on recall@5. **0.725 is the
   honest within-role number** and the one to quote. Some of that gap is
   residual optimism from selecting on dev; some is simply that 8 queries over
   32 CVs is a noisy estimate. It is not possible to separate the two with a
   corpus this size, and this README does not pretend otherwise.

3. **Cross-role performance generalises essentially perfectly** — 0.890 dev
   against 0.896 test. That is not a sign of a well-tuned system so much as
   confirmation that the cross-role task is too easy to distinguish anything.

4. **Retrieval, not reranking, is the ceiling.** Within-role pool recall is
   0.829 on held-out queries: roughly one relevant candidate in six never
   reaches the reranker, and no amount of reranking recovers them. Stage two can
   only reorder the 0.829 that stage one hands it.

### How the chunking question was actually settled

This sequence is the most instructive thing in the project, so it is recorded in
full rather than summarised into the answer.

| stage | evidence | within-role nDCG | conclusion drawn |
|---|---|---|---|
| 1. Original design | none — it just seemed principled | — | section chunking ships |
| 2. First benchmark | 32 CVs of ~55 words | identical | no difference detectable |
| 3. Realistic corpus | 15 dev queries, 255–304 words | window 0.833, section 0.761 | **default changed to window** |
| 4. Held-out set | 8 held-out queries | window 0.725, **window60 0.775** | window ranks *third of four* |
| 5. Expanded dev set | 17 dev queries | **window 0.803**, window60 0.759 | window confirmed; stage 4 was noise |

**At stage 4 the default was deliberately left alone.** `window60` had won the
only clean measurement available, and the temptation was to switch. Switching
would have been the exact mistake the split exists to prevent: it would have
spent the held-out set on a decision and left nothing clean to check the result
against. So the finding was recorded as *unresolved* and the dev set was
expanded instead — from 15 queries to 37 — which is the only legitimate way to
settle a question of that size.

At stage 5, with more than twice the within-role queries, the window wins by
0.044 over `window60` and the stage-4 ordering disappears. The held-out result
was an 8-query fluctuation, and it is visible as such only because nothing was
changed in response to it.

Two things to take from this:

- **A 0.05 nDCG difference on 8 queries is not a finding.** Three of the four
  chunkers sat inside that band, and their ordering changed with the sample. The
  hybrid result (+0.14 to +0.22, every chunker, both splits) is a different kind
  of claim entirely, and only one of the two belongs in a summary.
- **Section chunking is the one part that held throughout.** It loses on both
  splits and under every retrieval mode, by a wide margin (0.633 vs 0.803 on the
  expanded dev set). The shipped default remains the window;
  `CHUNK_STRATEGY=section` restores the old behaviour.

**Caveat, stated plainly:** 17 within-role dev queries over 32 CVs is still a
small sample. The 0.044 gap between the window and `window60` is more credible
than the stage-4 reversal it overturns, but it is not decisive, and if a future
measurement moves it that would not be surprising. What *is* now well supported
is the gap between either of them and section chunking.


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

**On the fixtures:** they are 255–304 words with three layout variants, including CVs with no recognisable headings at all — realistic enough for chunking strategies to differ, still shorter than a real 400–800 word CV. The short 55-word originals are what made every chunking strategy look identical, which is why the corpus was rebuilt.

### Reranker evaluation: three ways

Everything above is LLM-free. There are now three ways to reorder the shortlist,
selected with `RERANKER_BACKEND`:

| backend | what it is | cost per query | reasoning text |
|---|---|---|---|
| `none` | hybrid retrieval order is final | 0 | no |
| `cross-encoder` | local `ms-marco-MiniLM-L-6-v2`, 22M params | ~0.55 s, no API call | quotes the best-matching section |
| `llm` (default) | Groq `openai/gpt-oss-120b`, `temperature=0` | ~1.5–2 s inference + an API call | yes, generated |

A cross-encoder is the standard middle ground: unlike the bi-encoder used for
retrieval, it reads the job description and a CV passage *together*, so it can
condition on the query — but it is a small classifier rather than a generative
model, so it needs no key, no quota and no network.

```bash
python -m evaluation.run_eval --rerank all --split test
```

#### nDCG@5, all three, both splits

| | retrieval only | + cross-encoder | + LLM |
|---|---|---|---|
| **Cross-role dev** (20 q) | 0.890 | **0.942** | 0.935 |
| **Within-role dev** (17 q) | 0.803 | 0.813 | *not measured — see below* |
| **Cross-role test** (9 q) | **0.896** | 0.843 | 0.906 |
| **Within-role test** (8 q) | 0.725 | 0.768 | **0.847** |

Held-out detail, the set that actually discriminates:

| within-role test | retrieval only | + cross-encoder | + LLM |
|---|---|---|---|
| recall@5 | 0.594 | 0.615 | **0.692** |
| precision@5 | 0.450 | 0.475 | **0.525** |
| nDCG@5 | 0.725 | 0.768 | **0.847** |
| MRR | 0.938 | 0.906 | **1.000** |
| mean rerank latency | — | 562 ms | 28.7 s wall clock¹ |

¹ Dominated by rate-limit backoff, not inference — see [Latency](#latency).

**1. On hard queries the LLM wins clearly, and by more than previously
believed.** +0.122 nDCG on held-out within-role queries (0.725 → 0.847), with
recall, precision and MRR all improving together and MRR reaching a perfect
1.000 — the right person was first for all eight queries. An earlier version of
this README reported the lift as **+0.03**. That figure came from the old
15-query dev set where retrieval scored 0.833; on held-out queries retrieval
scores 0.725, which leaves far more for the reranker to fix.

**2. That is the same mechanism as before, pointing the other way.** The
previously documented finding was that the reranker's value *shrank* as
retrieval improved (+0.27 → +0.03). The corollary, now observed directly, is
that it *grows* where retrieval is weaker. Reranker lift is not a property of
the reranker; it is a property of how much stage one left on the table.

**3. The cross-encoder captures roughly a third of the LLM's gain, at roughly a
third of the inference time and none of the API cost.** +0.043 against +0.122 on
held-out within-role nDCG; ~0.55 s locally against 1.5–2 s plus a billable call
(and against ~25 s once a free tier starts throttling). If the choice
is framed as quality per unit cost it wins comfortably; if it is framed as
shortlist quality for a recruiter who has to live with the ranking, it does not.
**It is not the shipped default** for that reason, but it is the right default
for anyone without an API key, and `RERANKER_BACKEND=cross-encoder` makes it one
environment variable away.

**4. On easy queries, neither reranker is worth running.** Cross-role is where
this is clearest: the LLM is flat (+0.045 dev, +0.010 test — inside noise), and
the cross-encoder flips sign between splits (+0.052 dev, **−0.053** test). A
result that changes direction between splits is not an effect; it is noise with
a confident face on it. Retrieval already scores ~0.89 there, so there is
nothing to fix and every judgement is a fresh chance to break something already
correct.

**5. The obvious optimisation does not work, and that was tested before
anything was built.** Reranking costs an API call on *every* request including
the ones it makes worse, so the natural idea is a confidence gate: skip stage two
when retrieval looks sure of itself. See
[Why there is no confidence gate](#why-there-is-no-confidence-gate) — no
serving-time signal predicted which queries reranking would help.

#### Why there is no confidence gate

The case for a gate looked strong: reranking is flat-to-negative on easy
queries and costs an API call every time. So before building one,
`evaluation/analyze_rerank_gate.py` asked a prior question — **does any signal
available when a request arrives predict that reranking will help or hurt?**

```bash
python -m evaluation.analyze_rerank_gate
```

It runs on the 37 **dev** queries only (there is deliberately no `--split` flag;
choosing a gate is a design decision, and design decisions are made on dev),
using the local cross-encoder because it is free and deterministic. For each
query it records the change in nDCG@5 from reranking, and the Spearman rank
correlation between that change and seven candidate signals. The bar for
adopting a gate, **ρ ≤ −0.40**, was fixed before looking at any result.

| per query, dev, cross-encoder | n | mean Δ nDCG@5 | helped | hurt | unchanged |
|---|---|---|---|---|---|
| cross-role | 20 | +0.052 | 5 | 6 | 9 |
| within-role | 17 | +0.010 | 9 | 8 | 0 |
| **all** | **37** | **+0.033** | **14** | **14** | **9** |

| signal (all available at serving time) | ρ with Δ nDCG |
|---|---|
| top-1 RRF score | −0.298 |
| top-1 vs top-2 RRF margin | −0.033 |
| top-10 / top-1 RRF ratio | +0.299 |
| **top-1 dense cosine similarity** | **−0.344** (best) |
| dense cosine, top-1 minus top-5 | −0.264 |
| overlap of dense and BM25 top 5 | −0.238 |
| dense and BM25 agree on the top result | −0.170 |
| *baseline nDCG — needs relevance labels, so unusable* | *−0.436* |

**No serving-time signal meets the bar.** The best, top-1 dense cosine at
−0.344, has an uncorrected p-value of roughly 0.04 on 37 queries; after
correcting for having compared seven signals it is about 0.26, which is not
evidence of anything. The only variable that does correlate — how badly
retrieval did — is the one that cannot be observed without labels.

What the per-query picture says is more useful than the correlations. The
cross-encoder helped 14 queries and hurt 14, and its +0.033 average is carried by
a few large wins on queries where retrieval had failed badly (a recommendations
query went from nDCG 0.15 to 0.88). **Reranking pays off precisely where
retrieval is wrong, and nothing cheap tells you where that is.** RRF scores
depend only on rank, so a wide score gap says how decisively the two retrievers
agreed with each other, not whether the top result is right.

**What this does and does not establish.** It establishes that a score-based gate
cannot be justified for the cross-encoder on this corpus. It does *not* establish
the same for the LLM reranker, whose per-query behaviour was not measured: that
would cost about 4,000 tokens per query, and the free quota that already failed
once cannot cover 37 of them. A better signal than the seven tried may exist, and
a corpus of 32 CVs may be too small for any relationship to show. The honest
position is "no evidence for a gate", not "gates cannot work".

An earlier draft of this README listed the gate as the best-supported next step.
That claim rested on the reranker being flat on *average* over easy queries,
which does not imply it is predictable *per* query.

#### The missing cell

Within-role dev has no LLM figure because **the measurement failed and is not
being reported as though it succeeded.** Groq's free tier allows 200,000 tokens
per day; evaluating 54 queries across two splits exhausted it mid-run:

```
Rate limit reached ... tokens per day (TPD): Limit 200000, Used 197974
window + hybrid + LLM rerank  !! 13/17 NOT reranked   0.804
```

13 of 17 queries fell back to retrieval order, so the 0.804 that row printed is
*retrieval's* score wearing a reranker's label — it is within 0.001 of the
retrieval-only 0.803, which is the tell. The harness flags this explicitly
rather than averaging silently over failures, which is the entire reason that
check exists: the number looked completely plausible.

It is left blank until it can be measured properly. The held-out within-role
figure is the one that matters anyway, and that run completed cleanly with all 8
queries reranked.

**This is also a finding about the design, not just about a free tier.** A
reranker that costs ~4,000 tokens per query cannot evaluate a 54-query benchmark
twice a day on a free plan. That is a real constraint on iterating, and it is a
large part of why a local cross-encoder is worth having even at a third of the
quality.

#### What the recall/precision pattern shows

On the held-out hard set all four metrics move the same way under the LLM, which
is the clean case. The cross-encoder is less tidy: on **dev** within-role it
*raises* nDCG (0.803 → 0.813) and MRR (0.931 → 0.971) while *lowering* recall@5
(0.719 → 0.679) and precision@5 (0.612 → 0.576).

That combination is not a contradiction. It means the cross-encoder pulls the
single best candidate to the top — graded nDCG and MRR reward that — while
pushing some merely-relevant people out of the top 5, which binary recall@5
punishes. For a recruiter reading a ranked list top-down, "best person first,
one fewer decent person on page one" is usually the better trade. It is worth
knowing that is the trade being made, rather than reading a 0.01 nDCG gain as an
unambiguous improvement.

#### Latency

| stage | cost | share |
|---|---|---|
| embedding the job description | 12–51 ms | <1% |
| hybrid retrieval | 1.3–2.6 ms | <1% |
| cross-encoder rerank | ~550 ms | ~95% when enabled |
| LLM rerank (inference only) | 1.5–2.1 s | ~97% when enabled |
| LLM rerank (observed wall clock) | 22–29 s | — |

**The 22–29 s figure needs its caveat stated, not buried.** It is what the
harness measured, and it is real on this account, but it is mostly the retry
logic sleeping through 60-second TPM windows on a free tier — not model latency.
A single uncontended call takes 1.5–2.1 s. Quote 1.5–2 s for a paid tier and
~25 s for a free one running a benchmark back-to-back; neither number alone is
honest on its own.

The fail-fast path is visible in the same data: once the *daily* quota was gone,
mean rerank time dropped to 6.4 s rather than climbing, because a quota error
asking the caller to wait 15 minutes is detected and abandoned instead of slept
through. Retries are for per-minute limits only.

#### Confidence in these numbers

- **One run per configuration.** No repeats, so none of these figures has an
  error bar. The LLM runs at `temperature=0` and the cross-encoder is fully
  deterministic, so the *models* do not vary between runs — but which queries
  hit a rate limit does.
- **8 held-out within-role queries.** Enough to see +0.122 clearly. Not enough
  to trust +0.043, and nowhere near enough to trust ±0.01.
- **One LLM, one cross-encoder.** `openai/gpt-oss-120b` and
  `ms-marco-MiniLM-L-6-v2`. These compare two specific models, not two
  architectures in general.
- **The cross-encoder is uncalibrated.** Its 0-1 score is a sigmoid of an
  ms-marco relevance logit, trained for passage relevance rather than resume
  fit. Observed values are extreme in both directions and sensitive to passage
  length: on one hand-checked query a strong match scored 0.955 and a keyword-
  dense distractor 0.016, while a shorter excerpt of the same strong CV scored
  0.04. Ordering is meaningful; the absolute number is not a percentage match
  and is not comparable to the LLM's. Calibration is deferred rather than
  faked.


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
python -m unittest discover -s tests -t . -v      # 286 tests
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
| `GEMINI_API_KEY` / `GROQ_API_KEY` | — | At least one enables LLM reranking. Gemini is tried first, Groq is the fallback. |
| `RERANKER_BACKEND` | `llm` | `llm`, `cross-encoder` (local, no key) or `none`. See [Reranker evaluation](#reranker-evaluation-three-ways). |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder weights, downloaded once (~90MB) and cached. |
| `QDRANT_PATH` | — | Run Qdrant embedded from this directory (no server). |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server, when not embedded. |
| `RETRIEVAL_CANDIDATES` | `30` | Distinct **candidates** retrieved for reranking (not chunks). |
| `MAX_CHARS_PER_CANDIDATE` | `1200` | Cap on resume text per candidate sent to the LLM. |

---

## Limitations

Known and deliberate, rather than hidden:

- **Metadata extraction still leans on regex first.** That is deliberate (see above), but the regex path itself is unchanged: years from phrases like "5 years of experience", location from eleven hardcoded Indian cities, name from the filename. The LLM fallback covers the failures rather than improving the fast path, and its accuracy has been checked on a handful of CVs, not measured across a labelled set.
- **Location filtering is exact-match on a guessed city.** No geocoding, no radius, no remote handling.
- **The evaluation corpus is synthetic and small.** Real resumes cannot be committed to a public repo, but these fixtures are cleaner and more uniformly structured than real CVs, so the absolute numbers are optimistic. The *relative* comparisons between configurations are the useful part.
- **No OCR.** Scanned image-only PDFs extract no text and are skipped with a warning.
- **Reranker latency dominates and is not optimised.** Measured: embed 12–51 ms, retrieve 1.3–2.6 ms, rerank 1505–2113 ms — about 97% of request time, for only 3 candidates. It has not been measured with a full 30-candidate shortlist, and there is no batching, caching or timeout tuning.
- **Model ids drift.** The Groq default is `openai/gpt-oss-120b`, verified working; the previous `llama-3.3-70b-versatile` now 404s. The Gemini default `gemini-2.5-flash` is **unverified** — no Gemini key was tested. Both are configurable via `GEMINI_MODEL` / `GROQ_MODEL`.
- **Results are from one model on a small corpus.** The reranker numbers come from a single provider (Groq `openai/gpt-oss-120b`) over the *original* 15 dev queries and 32 synthetic CVs, and the +0.03 lift rests on one mostly-clean run. They show the pipeline works and that retrieval quality erodes the reranker's margin; they are not a general claim about reranking.
- **Every reranker figure is a single run.** The LLM runs at `temperature=0` and the cross-encoder is deterministic, so the models do not vary — but there are no error bars, and one within-role dev cell is missing entirely because a daily quota ran out mid-measurement.
- **Reranking is compared across two specific models**, `openai/gpt-oss-120b` and `ms-marco-MiniLM-L-6-v2`. These are not general claims about LLMs versus cross-encoders.
- **The cross-encoder's scores are uncalibrated.** Ordering is meaningful; the absolute 0–1 value is a sigmoid of a relevance logit, not a percentage match, and it is sensitive to how long the matched passage is.
- **17 held-out queries is a thin test set.** It is enough to show a 0.22 nDCG effect and not enough to resolve a 0.05 one — as demonstrated when its 8 within-role queries ranked the shipped chunker third, and 17 dev queries then put it first. Treat the chunker ordering in the held-out table as noise, and single-digit differences anywhere in this README as undetermined.
- **Relevance labels are author-assigned.** One person graded all 54 queries, with no second annotator and no inter-rater agreement measured. The grades encode a defensible reading of each role, not a consensus one.

## Next steps

In rough priority order:

1. **Fill the missing within-role dev LLM cell.** It needs one clean run on a
   quota that is not already spent — see [The missing cell](#the-missing-cell).
   Blocked on quota, not on code.
2. **Decide the reranking policy on cost, not on a gate.** A score-based gate
   was tested and found unsupported (see
   [Why there is no confidence gate](#why-there-is-no-confidence-gate)). What
   remains is a plain choice: `llm` where quality on hard queries justifies the
   spend, `cross-encoder` where it does not, `none` where neither does. Measuring
   per-query LLM deltas, once there is quota to do it, is the only thing that
   could reopen the gate question for the LLM.
3. **Repeat each reranker measurement several times.** Every figure in the
   reranker tables is a single run. The models are deterministic; which queries
   hit a rate limit is not.
4. **Calibrate the cross-encoder's scores.** ms-marco logits are trained for
   passage relevance, not resume fit, and the sigmoid of a raw logit is not a
   percentage. Fit a monotonic calibration on dev so the API's `score` means
   something absolute and comparable across backends.
5. Measure metadata-extraction accuracy against a labelled set, rather than spot-checking it.
6. Pin `qdrant-client` more tightly. The Docker build resolved 1.19.1 against a `~=1.18` pin; it works, but a loose pin is exactly what broke `.search()` before.
