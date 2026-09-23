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
flattered the results. All of that is documented below, including the numbers
that are unflattering.

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
                                                    LLM rerank (Groq / Gemini)
                                                                 │
                                                                 ▼
                                                  scored candidates + reasoning
```

### Headline results

Measured on 32 labelled CVs and 15 job descriptions. The **within-role** set is
the one that matters: every strong candidate there is a Python backend engineer
and the job turns on a single requirement, so keyword overlap carries no signal.

| | recall@5 | nDCG@5 |
|---|---|---|
| dense vectors only | 0.648 | 0.677 |
| **+ hybrid BM25 retrieval** | **0.814** | **0.833** |
| + LLM reranking | 0.843 | 0.865 |

Hybrid retrieval is the large win. Reranking adds little here — and actively
hurts on easy queries — which is itself the most interesting finding in the
project. See [Reranker evaluation](#reranker-evaluation).

### Contents

- [Quick start](#quick-start) — running it in about two minutes, no Docker needed
- [How it works](#how-it-works) — the pipeline, stage by stage
- [Design decisions](#design-decisions) — what was chosen and what it cost
- [Evaluation](#evaluation) — the benchmark, the results, and the bug in it
- [Testing](#testing) — 183 tests and why the original 32 were worthless
- [Configuration](#configuration) — every environment variable
- [Limitations](#limitations) — what this does not do, stated plainly
- [Next steps](#next-steps)

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

`POST /api/v1/screen` embeds the JD, applies hard filters in Qdrant, retrieves a shortlist, reranks, and returns the top `k`. `GET /health` reports Qdrant connectivity, model load, and whether an LLM is actually configured.

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
retrieval budget:

| | dense only | hybrid | |
|---|---|---|---|
| recall@5 | 0.648 | **0.814** | +0.17 |
| nDCG@5 | 0.677 | **0.833** | +0.16 |
| pool recall | 0.900 | 0.900 | unchanged |

The unchanged pool recall is the informative part. **Hybrid retrieval is not
finding different candidates — it is ordering the same pool far better.** Both
strategies surface the same ~90% of relevant people within a budget of 10; BM25
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

### Reranker evaluation

Everything above is LLM-free. Adding the reranker on top of the shipped
configuration (window + hybrid, 15 queries, Groq `openai/gpt-oss-120b`,
`temperature=0`):

| | retrieval only | + reranking | |
|---|---|---|---|
| **Cross-role** (the easy set) | | | |
| recall@5 | 0.919 | 0.863 | **worse** |
| precision@5 | 0.450 | 0.400 | **worse** |
| nDCG@5 | 0.958 | 0.922 | **worse** |
| MRR | 1.000 | 1.000 | — |
| **Within-role** (the hard set) | | | |
| recall@5 | 0.814 | 0.843 | better |
| precision@5 | 0.743 | 0.771 | better |
| nDCG@5 | 0.833 | 0.865 | better |
| MRR | 0.905 | 0.905 | — |

**Reranking helps on hard queries and hurts on easy ones.** That is the most
useful result in the project, and it is not a subtle effect.

On cross-role queries retrieval already scores 0.958. A React CV and a
Kubernetes CV are trivially different, so there is nothing left for the LLM to
fix and every judgement it makes is another chance to be wrong about something
already correct. On within-role queries retrieval scores 0.833 — sixteen Python
backend engineers where the job turns on one requirement — and there the LLM
reads that requirement properly.

**The implication is a real optimisation, not just an observation.** Reranking
costs ~1.5 s and an API call on *every* request, including the ones it makes
slightly worse. A confidence gate — skip the LLM when the top retrieval score is
clearly separated from the rest — would cut both latency and spend on exactly
the queries where reranking is not earning its cost. That is the next thing this
project should do.

#### The reranker's value shrank as retrieval improved

An earlier measurement put the within-role lift at **+0.27 nDCG**. It is now
**+0.03**. Nothing regressed; retrieval got better.

| | retrieval nDCG@5 | reranker lift |
|---|---|---|
| section chunking, dense-only, 8 queries | 0.605 | +0.27 |
| window chunking, hybrid, 15 queries | 0.833 | +0.03 |

When retrieval scored 0.605 the LLM had a great deal to fix. Now most of that
work happens upstream — cheaper, faster, and with no API call. **Improving stage
one did not compound with stage two; it ate its margin.** The value of an
expensive reranker is a function of how weak the cheap stage is, which is worth
knowing before committing to one architecturally.

#### What the recall drop does and does not mean

Cross-role recall@5 falls because the reranker promotes strong (grade 2) matches
above partial (grade 1) ones, and recall@5 counts both as merely "relevant".
Graded nDCG rewards that reordering; binary recall punishes it.

The demoted candidates are **not discarded** — they move below the k=5 cutoff and
reappear at a higher `top_k`. Whether the behaviour is correct depends on the
task: for a shortlist of the best few to interview it is what you want, for a
longlist of everyone worth a look it is not.

#### Confidence in these numbers

One mostly-clean run: 13 of 15 queries were reranked, the other two hit the
daily token quota and fell back. **Treat it as a single observation, not a
range.**

Repeating it has not been possible. A full rerank pass costs roughly 55,000
tokens and the Groq free tier allows 200,000/day, so about three runs — and the
ablation sweeps consumed the budget before a clean repeat could be taken.

Because a fully degraded run prints numbers shaped exactly like a result, the
harness now labels them: a row reading `!! 7/7 NOT reranked` is retrieval-only
output. That guard exists because such runs were briefly mistaken for data.

#### Latency

Measured over live API requests with 3 candidates:

| stage | time |
|---|---|
| embed | 12 – 51 ms |
| retrieve | 1.3 – 2.6 ms |
| **rerank** | **1505 – 2113 ms** |
| total | 1.5 – 2.2 s |

The LLM is ~97% of request time; retrieval is effectively free. Any latency work
belongs at the reranking stage — and per the table above, the cheapest win is
not doing it at all when retrieval is already confident.

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
python -m unittest discover -s tests -t . -v      # 183 tests
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

- **Metadata extraction still leans on regex first.** That is deliberate (see above), but the regex path itself is unchanged: years from phrases like "5 years of experience", location from eleven hardcoded Indian cities, name from the filename. The LLM fallback covers the failures rather than improving the fast path, and its accuracy has been checked on a handful of CVs, not measured across a labelled set.
- **Location filtering is exact-match on a guessed city.** No geocoding, no radius, no remote handling.
- **The evaluation corpus is synthetic and small.** Real resumes cannot be committed to a public repo, but these fixtures are cleaner and more uniformly structured than real CVs, so the absolute numbers are optimistic. The *relative* comparisons between configurations are the useful part.
- **No OCR.** Scanned image-only PDFs extract no text and are skipped with a warning.
- **Reranker latency dominates and is not optimised.** Measured: embed 12–51 ms, retrieve 1.3–2.6 ms, rerank 1505–2113 ms — about 97% of request time, for only 3 candidates. It has not been measured with a full 30-candidate shortlist, and there is no batching, caching or timeout tuning.
- **Model ids drift.** The Groq default is `openai/gpt-oss-120b`, verified working; the previous `llama-3.3-70b-versatile` now 404s. The Gemini default `gemini-2.5-flash` is **unverified** — no Gemini key was tested. Both are configurable via `GEMINI_MODEL` / `GROQ_MODEL`.
- **Results are from one model on a small corpus.** The reranker numbers come from a single provider (Groq `openai/gpt-oss-120b`) over 15 queries and 32 synthetic CVs, and the +0.03 lift rests on one mostly-clean run. They show the pipeline works and that retrieval quality erodes the reranker's margin; they are not a general claim about reranking.

## Next steps

In rough priority order:

1. Repeat the hybrid + reranker measurement across several runs, so the +0.03 lift is a range rather than a single observation.
2. Grow the query set further. 15 labelled queries is enough to separate 0.07 nDCG but not 0.03.
3. Measure metadata-extraction accuracy against a labelled set, rather than spot-checking it.
4. Pin `qdrant-client` more tightly. The Docker build resolved 1.19.1 against a `~=1.18` pin; it works, but a loose pin is exactly what broke `.search()` before.
