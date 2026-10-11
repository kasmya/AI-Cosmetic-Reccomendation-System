import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from cosmetic_rec.api import app
from cosmetic_rec.db import Database, Feedback, RecommendationEvent
from cosmetic_rec.state import state

URLS = [f"sqlite:///{tempfile.mkdtemp(prefix='db_')}/t.db"]
if os.environ.get("TEST_POSTGRES_URL"):                 # e.g. postgresql+psycopg://user@127.0.0.1:5432/testdb
    URLS.append(os.environ["TEST_POSTGRES_URL"])
SHOWN = [{"product_id": 1, "score": 80, "p_like": 0.8, "verdict": "Recommended"}]


@pytest.fixture(params=URLS, ids=lambda u: u.split(":")[0].split("+")[0])
def db(request):
    d = Database(request.param)
    with d.session() as s:                               # clean slate on a shared server
        s.query(Feedback).delete()
        s.query(RecommendationEvent).delete()
    return d


def test_events_and_feedback_round_trip(db):
    eid = db.log_event("v1", "Dry", ["acne", "redness"], 90.0, SHOWN)
    assert eid and len(eid) == 36
    first = db.add_feedback(1, True, eid)
    assert first["updated"] is False and isinstance(first["id"], int)
    rows = db.feedback_rows()
    assert len(rows) == 1 and rows[0]["helpful"] is True and rows[0]["skin_type"] == "Dry"
    assert db.recent_events()[0]["shown"] == SHOWN


def test_a_second_vote_replaces_the_first(db):
    eid = db.log_event("v1", None, [], None, SHOWN)
    db.add_feedback(1, True, eid)
    assert db.add_feedback(1, False, eid)["updated"] is True
    assert db.counts() == {"events": 1, "feedback": 1, "helpful": 0}


def test_votes_without_a_consultation_are_allowed_but_unknown_ids_are_not(db):
    assert db.add_feedback(5, True)["updated"] is False
    with pytest.raises(KeyError):
        db.add_feedback(5, True, "00000000-0000-0000-0000-000000000000")


def test_logging_never_raises(db, monkeypatch):
    monkeypatch.setattr(db, "session", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    assert db.log_event("v1", "Dry", [], None, SHOWN) is None


def test_schema_has_no_personal_data_columns(db):
    cols = {c["name"] for t in ("recommendation_events", "feedback") for c in inspect(db.engine).get_columns(t)}
    assert not cols & {"ip", "email", "name", "photo", "image", "user_agent", "text", "review"}


# ---------- API ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_consult_returns_a_request_id_and_logs_what_was_shown(client):
    before = state["db"].counts()["events"]
    j = client.post("/api/v1/consult", json={"skin_type": "Dry", "concerns": ["dryness"], "per_category": 1}).json()
    assert j["request_id"] and state["db"].counts()["events"] == before + 1
    ev = state["db"].recent_events(1)[0]
    assert ev["skin_type"] == "Dry" and len(ev["shown"]) == 3 and all("p_like" in s for s in ev["shown"])


def test_feedback_endpoint_validation_and_replacement(client):
    j = client.post("/api/v1/consult", json={"skin_type": "Oily", "per_category": 1}).json()
    pid = j["skincare"]["Cleanser"][0]["product_id"]
    ok = client.post("/api/v1/feedback", json={"product_id": pid, "helpful": True, "request_id": j["request_id"]})
    assert ok.status_code == 200 and ok.json() == {"ok": True, "updated": False}
    again = client.post("/api/v1/feedback", json={"product_id": pid, "helpful": False, "request_id": j["request_id"]})
    assert again.json()["updated"] is True
    assert client.post("/api/v1/feedback", json={"product_id": 999999, "helpful": True}).status_code == 404
    bad = {"product_id": pid, "helpful": True, "request_id": "00000000-0000-0000-0000-000000000000"}
    assert client.post("/api/v1/feedback", json=bad).status_code == 404
    assert client.post("/api/v1/feedback", json={"helpful": True}).status_code == 422


def test_consult_still_works_if_the_database_is_down(client, monkeypatch):
    monkeypatch.setattr(state["db"], "session", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    r = client.post("/api/v1/consult", json={"skin_type": "Dry", "per_category": 1})
    assert r.status_code == 200 and r.json()["request_id"] is None
