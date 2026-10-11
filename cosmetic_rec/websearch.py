"""Optional live web search through Tavily (https://tavily.com).

Disabled unless TAVILY_API_KEY is set, either in the environment or in a git-ignored `.env` file at the repo root.
Results are untrusted web text: they are returned as data for display and never fed to the models.
"""
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger("cosmetic_rec.websearch")
API_URL = "https://api.tavily.com/search"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Minimal .env reader (KEY=VALUE lines). Never overrides variables that are already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class WebSearch:
    def __init__(self, api_key: str | None = None, ttl: float = 3600, max_calls_per_hour: int = 60,
                 transport: httpx.BaseTransport | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self.ttl, self.max_calls = ttl, max_calls_per_hour
        self._transport = transport
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._calls: list[float] = []

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """[{title, url, snippet}]. Cached for `ttl` seconds; raises RuntimeError on quota/API failure."""
        now = time.time()
        hit = self._cache.get(query)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        self._calls = [t for t in self._calls if now - t < 3600]
        if len(self._calls) >= self.max_calls:
            raise RuntimeError("web search call limit reached, try again later")
        self._calls.append(now)
        try:
            with httpx.Client(timeout=10, transport=self._transport) as client:
                r = client.post(API_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                json={"query": query, "max_results": max_results, "search_depth": "basic"})
                r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("tavily request failed: %s", type(e).__name__)
            raise RuntimeError("web search request failed") from e
        results = []
        for item in r.json().get("results", []):
            url = item.get("url", "")
            if urlparse(url).scheme not in ("http", "https"):     # never pass javascript:/data: links on
                continue
            results.append({"title": str(item.get("title", ""))[:200], "url": url,
                            "snippet": str(item.get("content", ""))[:300]})
        self._cache[query] = (now, results)
        return results
