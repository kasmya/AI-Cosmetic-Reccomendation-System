"""Test setup: never touch the real database or call real web search."""
import os
import tempfile

# Must be set before cosmetic_rec.api is imported by any test module.
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='cosmetic_test_')}/test.db"
os.environ.pop("TAVILY_API_KEY", None)
