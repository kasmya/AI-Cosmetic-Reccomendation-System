"""Cross-checks for the deployment files (Docker is not available on every machine, so CI runs `docker compose config`
and a container smoke test; these tests catch typos and drift between the files and the code first)."""
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY

from cosmetic_rec import metrics  # noqa: F401  (registers the metrics)

ROOT = Path(__file__).resolve().parent.parent
compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
dashboard = json.loads((ROOT / "deploy/grafana/dashboards/cosmetic.json").read_text())


# ---------- compose ----------
def test_services_and_dependencies():
    assert set(compose["services"]) == {"db", "api", "prometheus", "grafana"}
    api = compose["services"]["api"]
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    assert "healthcheck" in compose["services"]["db"]


def test_api_talks_to_postgres_with_the_installed_driver_on_the_service_host():
    url = compose["services"]["api"]["environment"]["DATABASE_URL"]
    assert url.startswith("postgresql+psycopg://") and "@db:5432/" in url
    assert "psycopg" in (ROOT / "requirements.txt").read_text()


def test_every_named_volume_is_declared_and_used():
    declared = set(compose["volumes"])
    mounts = [v for s in compose["services"].values() for v in s.get("volumes", [])]
    used = {v.split(":")[0] for v in mounts if not v.startswith(".")}
    assert used == declared


def test_every_compose_variable_is_documented_in_env_example():
    variables = set(re.findall(r"\$\{([A-Z_]+)", (ROOT / "docker-compose.yml").read_text()))
    documented = {ln.split("=")[0] for ln in (ROOT / ".env.example").read_text().splitlines() if "=" in ln}
    assert variables <= documented, variables - documented


def test_env_example_contains_no_real_secrets():
    for ln in (ROOT / ".env.example").read_text().splitlines():
        if "=" in ln and not ln.startswith("#"):
            assert ln.split("=", 1)[1] in ("", "2"), ln


def test_prometheus_scrapes_the_api_service_and_grafana_mounts_line_up():
    prom = yaml.safe_load((ROOT / "deploy/prometheus.yml").read_text())
    assert prom["scrape_configs"][0]["static_configs"][0]["targets"] == ["api:8000"]
    assert prom["scrape_configs"][0]["metrics_path"] == "/metrics"
    provider = yaml.safe_load((ROOT / "deploy/grafana/provisioning/dashboards/dashboards.yml").read_text())
    path = provider["providers"][0]["options"]["path"]
    assert any(v.endswith(f":{path}:ro") for v in compose["services"]["grafana"]["volumes"])
    ds = yaml.safe_load((ROOT / "deploy/grafana/provisioning/datasources/prometheus.yml").read_text())["datasources"][0]
    assert ds["url"] == "http://prometheus:9090" and ds["uid"] == "prometheus"


# ---------- dashboard ----------
def registered_metric_names() -> set[str]:
    names = set()
    for collector in list(REGISTRY._collector_to_names):
        for n in REGISTRY._collector_to_names[collector]:
            names.add(n)
    return names


def test_every_metric_in_the_dashboard_exists_in_the_code():
    known = registered_metric_names()
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            idents = set(re.findall(r"\b([a-z][a-z0-9_]*(?:_total|_bucket|_sum|_count|_auc|_psi|_info|_candidates))\b",
                                    target["expr"]))
            assert idents, panel["title"]
            for ident in idents:
                assert ident in known or ident.removesuffix("_bucket") in known, (panel["title"], ident)


def test_dashboard_panels_have_unique_ids_titles_and_the_provisioned_datasource():
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids)) and len({p["title"] for p in dashboard["panels"]}) == len(ids)
    assert all(p["datasource"]["uid"] == "prometheus" for p in dashboard["panels"])


# ---------- Dockerfile ----------
def test_dockerfile_is_non_root_has_a_healthcheck_and_multiprocess_metrics():
    text = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^USER app$", text, re.M) and "HEALTHCHECK" in text
    assert "PROMETHEUS_MULTIPROC_DIR" in text and "COPY .env" not in text


def test_dockerignore_keeps_secrets_and_local_state_out_of_images():
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert {".env", "var", "models/registry", ".git"} <= set(ignored)


# ---------- multi-process metrics (real separate processes) ----------
def test_counters_from_separate_processes_are_summed(tmp_path):
    prog = textwrap.dedent("""
        import sys
        from cosmetic_rec import metrics
        n = int(sys.argv[1])
        for _ in range(n):
            metrics.REQUESTS.labels("GET", "/x", "200").inc()
        if sys.argv[2] == "render":
            body, _ = metrics.render()
            prefix = 'http_requests_total{method="GET",route="/x"'
            print([ln for ln in body.decode().splitlines() if ln.startswith(prefix)][0].split()[-1])
    """)
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    for n in (3, 4):                                                    # two "workers" write, then exit
        subprocess.run([sys.executable, "-c", prog, str(n), "no"], env=env, cwd=ROOT, check=True)
    out = subprocess.run([sys.executable, "-c", prog, "5", "render"], env=env, cwd=ROOT, check=True,
                         capture_output=True, text=True)
    assert out.stdout.strip() == "12.0"                                 # 3 + 4 + 5 across three processes


@pytest.mark.parametrize("gauge", ["MODEL_INFO", "MODEL_HOLDOUT_AUC", "DRIFT_PSI"])
def test_gauges_have_an_explicit_multiprocess_mode(gauge):
    """Without one, a gauge set by a single worker would be reported as the default 'all' (one series per process)."""
    assert getattr(metrics, gauge)._multiprocess_mode == "max"
