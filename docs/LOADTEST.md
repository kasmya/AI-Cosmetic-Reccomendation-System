# Load test

Tool: [Locust](https://locust.io). Script: [loadtest/locustfile.py](../loadtest/locustfile.py). Run: `make loadtest` (server already running).

**Traffic mix** (weights): consultations 60, product score 15, product detail + review summary 10, catalog search 5,
feedback 10, face analysis 4, makeup try-on 2. Each simulated user waits 0.2-1.0 s between actions. 60 s per run.

**Setup:** Apple M4 (10 cores, 16 GB). The load generator runs on the SAME machine as the server, so both compete for CPU.
PostgreSQL runs were against a local throwaway Postgres 17 cluster, not a networked database. Single runs, no repeats.

## What the first run found (and the fix)

20 simulated users, 1 worker, SQLite. A consultation took **11 s at the median** (single requests took 0.2 s) and
`/products/{id}/score` took 1.6 s. Profiling showed almost all of it was one function: the calibrated model re-run 7 times
per consultation on identical inputs (about 33 ms each), plus ingredient-text re-scans and pandas filters repeated
thousands of times. The model never changes, so there are only about 55 distinct inputs (skin type x tone). Fixes:

| Change | Effect |
|---|---|
| cache `predict_like` per (skin type, tone) | consult 221 ms to 41 ms, `/score` 25 ms to 1 ms |
| cache per-concern ingredient arrays, precompute review counts, reuse concern fit from the ranking stage | consult 41 ms to 23 ms |

None of the caches is saved in a model artifact, and tests pin each one to the slow computation, so a cache cannot change an answer.

## Results after the fix

| Setup | Users | Requests | Req/s | Failures | Consult p50 / p95 / p99 |
|---|---|---|---|---|---|
| Before: 1 worker, SQLite | 20 | 212 | 3.6 | 0 | **11,000** / 12,000 / 12,000 ms |
| 1 worker, SQLite | 20 | 2,067 | 34.9 | 0 | 40 / 91 / 170 ms |
| 4 workers, PostgreSQL 17 | 50 | 4,849 | 82.0 | 0 | 35 / 250 / 590 ms |
| 4 workers, SQLite | 50 | 4,873 | 82.4 | 0 | 35 / 160 / 1,100 ms |

Other endpoints at 50 users on PostgreSQL: score p50 2 ms (p95 64), feedback p50 4 ms (p95 87), face analysis p50 11 ms
(p95 86), makeup try-on p50 26 ms (p95 130).

## How to read this

- **Not a capacity limit.** With think time, 50 users offer only about 82 req/s, and the server kept up with zero errors.
  The test shows the system is healthy at that load, not where it breaks. Finding the breaking point needs a no-think-time
  run on a separate load-generator machine.
- **SQLite vs PostgreSQL:** throughput and median were the same; SQLite's p99 tail was longer here (1.1 s vs 0.6 s), plausibly
  write-lock waits from event logging. One run each, so treat it as a hint, not a result. PostgreSQL is the production choice.
- **Photo endpoints are cheap here** because the test image is small; real phone photos (up to 5 MB, decoded and resized) will cost more.
- **Not tested:** the full docker-compose stack under load, network latency, or more than 4 workers.
