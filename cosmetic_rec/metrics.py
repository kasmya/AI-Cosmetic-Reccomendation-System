"""Prometheus metrics. Labels use the route TEMPLATE (/products/{product_id}), never the raw URL, so the number of
time series stays bounded no matter how many products or users there are."""
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency in seconds", ["route"], buckets=LATENCY_BUCKETS)
VERDICTS = Counter("consult_verdict_total", "Verdicts of the skincare picks shown in consultations", ["verdict"])
BUDGET_MISSES = Counter("consult_budget_infeasible_total", "Consultations whose budget could not be met")
FEEDBACK = Counter("feedback_total", "Thumbs up/down votes", ["helpful"])
CANDIDATES = Histogram("recommendation_candidates", "Candidates entering the ranking stage per recommend call",
                       buckets=(5, 10, 25, 50, 100, 200, 400))
MODEL_INFO = Gauge("model_info", "1 for the champion model version being served", ["version"])
MODEL_HOLDOUT_AUC = Gauge("model_holdout_auc", "Gold-holdout AUC of the serving model")
DRIFT_PSI = Gauge("drift_psi", "Population stability index, recent vs reference window", ["feature"])


def route_label(request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
