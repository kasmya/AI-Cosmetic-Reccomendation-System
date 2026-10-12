import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from cosmetic_rec.db import normalise_url

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("given,expected", [
    ("postgres://u:p@h/d?sslmode=require", "postgresql+psycopg://u:p@h/d?sslmode=require"),
    ("postgresql://u:p@h/d", "postgresql+psycopg://u:p@h/d"),
    ("postgresql+psycopg://u:p@h/d", "postgresql+psycopg://u:p@h/d"),
    ("sqlite:///x.db", "sqlite:///x.db")])
def test_database_urls_are_normalised(given, expected):
    assert normalise_url(given) == expected


def test_render_blueprint_is_free_docker_with_health_check():
    svc = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    assert svc["runtime"] == "docker" and svc["plan"] == "free" and svc["healthCheckPath"] == "/health"
    keys = {e["key"] for e in svc["envVars"]}
    assert {"DATABASE_URL", "OPS_TOKEN", "WEB_CONCURRENCY"} <= keys


def test_dockerfile_listens_on_port_env():
    text = (ROOT / "Dockerfile").read_text()
    assert re.search(r"--port \$\{PORT:-8000\}", text)


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_pages_worker():
    r = subprocess.run(["node", str(ROOT / "tests/js/worker.test.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
