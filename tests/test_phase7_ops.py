import numpy as np
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app
from cosmetic_rec.drift import MIN_PER_WINDOW, _continuous_psi, compute_drift, psi, status
from cosmetic_rec.state import state


# ---------- PSI ----------
def test_psi_is_zero_for_identical_shares_and_positive_otherwise():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0, abs=1e-9)
    assert psi([10, 20, 30], [30, 20, 10]) > 0.25
    assert psi([10, 20, 30], [1, 2, 3]) == pytest.approx(0, abs=1e-9)           # only the shares matter


def test_psi_survives_empty_bins():
    assert np.isfinite(psi([50, 50, 0], [0, 50, 50]))


@pytest.mark.parametrize("value,label", [(0.02, "stable"), (0.1, "moderate"), (0.24, "moderate"), (0.3, "significant")])
def test_status_thresholds(value, label):
    assert status(value) == label


def test_continuous_psi_separates_same_from_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.75, 0.05, 3000)
    assert _continuous_psi(ref, rng.normal(0.75, 0.05, 3000)) < 0.1
    assert _continuous_psi(ref, rng.normal(0.65, 0.05, 3000)) > 0.25


# ---------- drift over events ----------
def events(n, p_mu, skins, seed=0):
    rng = np.random.default_rng(seed)
    return [{"skin_type": skins[rng.integers(len(skins))],
             "shown": [{"p_like": float(np.clip(rng.normal(p_mu, 0.04), 0, 1))} for _ in range(3)]} for _ in range(n)]


def test_drift_needs_enough_events():
    assert compute_drift(events(10, 0.75, ["Dry"]))["status"] == "insufficient_data"
    assert compute_drift(events(2 * MIN_PER_WINDOW - 2, 0.75, ["Dry"]))["status"] == "insufficient_data"


def test_stable_traffic_is_stable_and_a_shift_is_flagged_per_feature():
    stable = compute_drift(events(100, 0.75, ["Dry", "Oily"], 1) + events(100, 0.75, ["Dry", "Oily"], 2))
    assert stable["status"] == "stable"
    newest_first = events(100, 0.62, ["Combination"], 3) + events(100, 0.75, ["Dry", "Oily"], 4)
    shifted = compute_drift(newest_first)
    assert shifted["status"] == "significant"
    assert {f["status"] for f in shifted["features"].values()} == {"significant"}


# ---------- API: metrics ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def metric_lines(client, prefix):
    return [ln for ln in client.get("/metrics").text.splitlines() if ln.startswith(prefix)]


def test_metrics_expose_requests_latency_verdicts_and_model_info(client):
    j = client.post("/api/v1/consult", json={"skin_type": "Dry", "per_category": 2}).json()
    client.post("/api/v1/feedback", json={"product_id": j["skincare"]["Cleanser"][0]["product_id"], "helpful": True,
                                          "request_id": j["request_id"]})
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/api/v1/consult",status="200"}' in text
    assert 'http_request_duration_seconds_bucket{le="0.5",route="/api/v1/consult"}' in text
    assert "consult_verdict_total{" in text and 'feedback_total{helpful="true"}' in text
    assert f'model_info{{version="{state["meta"]["version"]}"}} 1.0' in text
    assert "recommendation_candidates_count" in text and "model_holdout_auc" in text


def test_route_labels_use_templates_so_series_do_not_explode(client):
    ids = [p["Product_id"] for p in client.get("/skincare/products?limit=12").json()]
    for i in ids:
        client.get(f"/api/v1/products/{i}")
    client.get("/definitely/not/a/route")
    routes = {ln.split('route="')[1].split('"')[0] for ln in metric_lines(client, "http_requests_total")}
    assert "/api/v1/products/{product_id}" in routes and "unmatched" in routes
    product_routes = {r for r in routes if r.startswith("/api/v1/products/")}
    assert all("{" in r for r in product_routes)          # templates only, never a raw /products/17 URL


# ---------- API: ops ----------
def test_ops_stats(client):
    j = client.get("/api/v1/ops/stats").json()
    assert {"events", "feedback", "helpful", "helpful_rate", "database", "model_version"} <= j.keys()
    assert j["database"] == "sqlite" and j["events"] >= 1


def test_ops_drift_reports_a_shift_and_sets_the_gauge(client, monkeypatch):
    shifted = events(100, 0.62, ["Combination"], 3) + events(100, 0.75, ["Dry", "Oily"], 4)
    monkeypatch.setattr(state["db"], "recent_events", lambda limit=2000: shifted)
    j = client.get("/api/v1/ops/drift").json()
    assert j["status"] == "significant" and j["features"]["skin_type"]["psi"] > 0.25
    assert any(ln.startswith('drift_psi{feature="skin_type"}') for ln in metric_lines(client, "drift_psi"))


def test_ops_drift_with_little_data_says_so(client, monkeypatch):
    monkeypatch.setattr(state["db"], "recent_events", lambda limit=2000: events(5, 0.7, ["Dry"]))
    assert client.get("/api/v1/ops/drift").json()["status"] == "insufficient_data"


# ---------- access token ----------
@pytest.mark.parametrize("path", ["/metrics", "/api/v1/ops/stats", "/api/v1/ops/drift"])
def test_ops_token_guards_operational_endpoints(client, monkeypatch, path):
    monkeypatch.setenv("OPS_TOKEN", "s3cret")
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-Ops-Token": "wrong"}).status_code == 401
    assert client.get(path, headers={"X-Ops-Token": "s3cret"}).status_code == 200
    assert client.get(path, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_ops_endpoints_are_open_without_a_token_and_normal_endpoints_never_need_one(client, monkeypatch):
    monkeypatch.delenv("OPS_TOKEN", raising=False)
    assert client.get("/metrics").status_code == 200
    monkeypatch.setenv("OPS_TOKEN", "s3cret")
    assert client.get("/health").status_code == 200 and client.get("/api/v1/options").status_code == 200
