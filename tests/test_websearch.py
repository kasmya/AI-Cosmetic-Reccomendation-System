import json

import httpx
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app
from cosmetic_rec.state import state
from cosmetic_rec.websearch import WebSearch, load_dotenv


def make(handler, **kw):
    return WebSearch(api_key="test-key", transport=httpx.MockTransport(handler), **kw)


def test_disabled_without_key():
    assert not WebSearch(api_key="").enabled


def test_search_sends_bearer_filters_bad_urls_and_caches():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [
            {"title": "Good", "url": "https://example.com/a", "content": "x" * 500},
            {"title": "Bad", "url": "javascript:alert(1)", "content": "nope"}]})

    ws = make(handler)
    out = ws.search("cerave cleanser")
    assert [r["url"] for r in out] == ["https://example.com/a"] and len(out[0]["snippet"]) == 300
    assert calls[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(calls[0].content)["query"] == "cerave cleanser"
    ws.search("cerave cleanser")
    assert len(calls) == 1                      # second call served from cache


def test_call_limit_and_http_errors():
    ws = make(lambda r: httpx.Response(500), max_calls_per_hour=1)
    with pytest.raises(RuntimeError):
        ws.search("a")
    with pytest.raises(RuntimeError, match="limit"):
        ws.search("b")


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nTAVILY_TEST_A=from_file\nTAVILY_TEST_B='quoted'\n")
    monkeypatch.setenv("TAVILY_TEST_A", "already")
    monkeypatch.delenv("TAVILY_TEST_B", raising=False)
    load_dotenv(f)
    import os
    assert os.environ["TAVILY_TEST_A"] == "already" and os.environ["TAVILY_TEST_B"] == "quoted"
    monkeypatch.delenv("TAVILY_TEST_B")


def test_endpoint_disabled_live_and_error():
    with TestClient(app) as c:
        pid = c.get("/skincare/products?limit=1").json()[0]["Product_id"]
        state["web"] = WebSearch(api_key="")
        assert c.get(f"/api/v1/products/{pid}/web").json()["status"] == "disabled"
        state["web"] = make(lambda r: httpx.Response(200, json={"results": [
            {"title": "T", "url": "https://example.com", "content": "c"}]}))
        j = c.get(f"/api/v1/products/{pid}/web").json()
        assert j["status"] == "live" and j["results"][0]["url"] == "https://example.com"
        state["web"] = make(lambda r: httpx.Response(401))
        assert c.get(f"/api/v1/products/{pid}/web").json()["status"] == "error"
        assert c.get("/api/v1/products/999999/web").status_code == 404
