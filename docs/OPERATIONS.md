# Operations

## Architecture

```
 Web UI ──► FastAPI (N workers) ──► constraints ► candidate retrieval ► ranker ► brand re-rank
                │   │                                (calibrated GBM + ingredient taste + concern fit)
                │   └─► PostgreSQL / SQLite: recommendation events, feedback (no personal data)
                │                      │
                │                      └─► retrain job ─► model registry ─► champion ─► API (on restart)
                └─► /metrics ─► Prometheus ─► Grafana          (drift PSI is computed from the event log)
```

## Run it

| What | Command |
|---|---|
| Local API | `make run` (SQLite in `var/app.db`) |
| Whole stack | `cp .env.example .env && docker compose up --build` (API :8000, Grafana :3000, Prometheus :9090) |
| Tests / lint | `make test`, `make lint` |
| Load test | `make loadtest` ([results](LOADTEST.md)) |

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://user:pass@host:5432/db` for PostgreSQL; unset means SQLite under `var/` |
| `TAVILY_API_KEY` | enables live web search (off without it) |
| `OPS_TOKEN` | if set, `/metrics` and `/api/v1/ops/*` require `Authorization: Bearer <token>` or `X-Ops-Token` |
| `WEB_CONCURRENCY` | server workers in the Docker image (default 2) |
| `PROMETHEUS_MULTIPROC_DIR` | set by the Docker image so metrics aggregate across workers; clear it at start |
| `COSMETIC_REGISTRY_DIR` | where model versions live (default `models/registry`) |

Secrets go in `.env` (git-ignored, excluded from images by `.dockerignore`). The repository is public: never commit keys.

## The model lifecycle

1. **Train** a challenger: `make train`. It trains on every review except the *gold holdout* (a fixed 20% of users that no
   version ever trains on), registers the version, and runs the gate.
2. **The gate** compares the challenger with the deployed champion, frozen, on the gold holdout: AUC non-inferiority (paired
   bootstrap over users, lower bound of the 95% interval at least -0.01), Brier not worse (+0.002), calibration error not
   worse (+0.01). Every decision and its reasons are appended to `models/registry/history.jsonl`.
3. **Learn from feedback:** `make retrain` adds stored thumbs up/down (thumbs up = 5 stars, down = 1 star, one pseudo-user per
   consultation) and runs the same gate.
4. **Serve:** the API loads the champion at start-up. After a promotion, restart the API (rolling restart in production).
5. **Inspect / override:** `python -m cosmetic_rec.registry list | history | promote VERSION [--force] | rollback | prune`.
   `--force` is recorded as forced in the history.

**Know the gate's limits.** The holdout has about 1,700 reviews and the signal is weak (AUC about 0.59), so the AUC
interval is roughly +-0.01 to 0.03 wide. The gate catches gross regressions (a crippled or poisoned model is rejected), cannot
see small improvements or small regressions, and sits right at its noise level: a harmless retrain on 600 synthetic votes was
rejected because the interval's lower bound was -0.0102 against the -0.01 tolerance. That is the safe direction (keep the
current model), and `promote --force` exists for a human decision.

`python scripts/simulate_feedback.py --n 400` fills the feedback table with **synthetic** votes (replayed training reviews,
no new information) to exercise this loop; `--clear` removes them. They never touch the gold holdout.

## Monitoring

- **Metrics** (`/metrics`): request rate and latency histogram by route template, 5xx share, verdict mix, budget misses,
  feedback votes, candidates per call, serving model version and holdout AUC, drift PSI. The Grafana dashboard
  "Cosmetic Recommender" is provisioned automatically in compose.
- **Drift** (`/api/v1/ops/drift`, also refreshed on every scrape): PSI of the model probabilities shown and of the skin-type mix,
  newest half of the event log vs the older half. Under 0.1 stable, 0.1-0.25 moderate, over 0.25 significant. It needs about
  60 events, compares the log with itself (there are no production labels), and a genuine change in who uses the site
  also shows as drift. A significant reading means "look", not "retrain".
- **Alerts** are not configured. Suggested first rules: 5xx share above 1% for 5 minutes, consult p95 above 1 s,
  drift PSI above 0.25 for 1 hour.

## Data and privacy

Stored: skin type, concern names, budget, the products shown, and thumbs up/down. Not stored: photos (analysed in memory),
free text, IP addresses, accounts. The request id is a random UUID. There is no retention policy or delete-my-data endpoint
yet; add one before a public launch.

## Not done (be honest in interviews)

- Docker Compose has been validated and built in CI, but the full stack with Prometheus and Grafana has not been brought up end to end.
- No database migrations (tables are created at start-up), no authentication on write endpoints, no rate limiting,
  no horizontal scaling test, no alerting, and the champion is only picked up after a restart.
