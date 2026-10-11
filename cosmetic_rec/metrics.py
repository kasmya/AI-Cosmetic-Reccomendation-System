"""Prometheus metrics. Labels use the route TEMPLATE (/products/{product_id}), never the raw URL, so the number of
time series stays bounded no matter how many products or users there are.

With several server workers each process has its own memory, so a scrape would see only the worker that answered it.
Set PROMETHEUS_MULTIPROC_DIR (an empty, writable directory, cleared at start-up) before the app starts and every
worker writes to shared files that /metrics aggregates (docker-compose does this)."""
import os
import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency in seconds", ["route"], buckets=LATENCY_BUCKETS)
VERDICTS = Counter("consult_verdict_total", "Verdicts of the skincare picks shown in consultations", ["verdict"])
BUDGET_MISSES = Counter("consult_budget_infeasible_total", "Consultations whose budget could not be met")
FEEDBACK = Counter("feedback_total", "Thumbs up/down votes", ["helpful"])
CANDIDATES = Histogram("recommendation_candidates", "Candidates entering the ranking stage per recommend call",
                       buckets=(5, 10, 25, 50, 100, 200, 400))
MODEL_INFO = Gauge("model_info", "1 for the champion model version being served", ["version"],
                   multiprocess_mode="max")
MODEL_HOLDOUT_AUC = Gauge("model_holdout_auc", "Gold-holdout AUC of the serving model", multiprocess_mode="max")
DRIFT_PSI = Gauge("drift_psi", "Population stability index, recent vs reference window", ["feature"],
                  multiprocess_mode="max")


def route_label(request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def render() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST


_drift_checked = 0.0


def refresh_drift(compute, ttl: float = 60.0) -> None:
    """Update the drift gauges from the event log at most once per `ttl` seconds (called when /metrics is scraped, so
    Grafana always has a current value without anyone calling /ops/drift). `compute` returns the drift result dict."""
    global _drift_checked
    if time.monotonic() - _drift_checked < ttl:
        return
    _drift_checked = time.monotonic()
    try:
        for name, f in compute()["features"].items():
            DRIFT_PSI.labels(name).set(f["psi"])
    except Exception:                                    # noqa: BLE001  monitoring must never break a scrape
        pass
