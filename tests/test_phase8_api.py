"""API additions made so that every capability built in earlier phases is reachable from the web page."""
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app
from cosmetic_rec.state import state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def consult(client, **body):
    r = client.post("/api/v1/consult", json={"skin_type": "Combination", "per_category": 5, **body})
    assert r.status_code == 200, r.text
    return r.json()


def picks(j):
    return [i for v in j["skincare"].values() for i in v]


# ---------- products the user already loves ----------
def a_product_with_ingredients(client):
    rec = state["rec"]
    return int(rec.products.index[rec.products["Ingredients"] != ""][0])


def test_liked_products_are_never_recommended_back_and_steer_the_picks(client):
    liked = a_product_with_ingredients(client)
    plain, steered = consult(client), consult(client, liked_product_ids=[liked])
    assert liked not in {i["product_id"] for i in picks(steered)}
    assert any("ingredient similarity to your liked products" in i["why"] for i in picks(steered))
    assert not any("liked products" in i["why"] for i in picks(plain))
    assert [i["product_id"] for i in picks(plain)] != [i["product_id"] for i in picks(steered)]


def test_liked_ids_are_validated_deduplicated_and_unknown_ones_ignored(client):
    liked = a_product_with_ingredients(client)
    assert consult(client, liked_product_ids=[liked, liked, 10**9])["request_id"] is not None
    too_many = client.post("/api/v1/consult", json={"liked_product_ids": list(range(11))})
    assert too_many.status_code == 422
    assert client.post("/api/v1/consult", json={"liked_product_ids": ["x"]}).status_code == 422


# ---------- foundation by colour and brand ----------
def test_brands_and_brand_filtered_matches(client):
    brands = client.get("/api/v1/makeup/brands").json()
    assert "MAC" in brands and len(brands) > 20
    j = client.get("/api/v1/makeup/match", params={"hex": "#c68642", "brand": "MAC", "k": 3}).json()
    assert j["skin_hex"] == "#c68642" and 1 <= len(j["matches"]) <= 3
    assert all(m["brand"] == "MAC" for m in j["matches"])
    deltas = [m["delta_e"] for m in j["matches"]]
    assert deltas == sorted(deltas)


def test_match_accepts_hex_without_hash_and_more_shades_without_a_brand(client):
    few = client.get("/api/v1/makeup/match", params={"hex": "c68642", "k": 2}).json()
    many = client.get("/api/v1/makeup/match", params={"hex": "c68642", "k": 12}).json()
    assert few["skin_hex"] == "#c68642" and len(few["matches"]) == 2 and len(many["matches"]) == 12
    assert few["matches"][0] == many["matches"][0]


@pytest.mark.parametrize("params", [{"hex": "nope"}, {"hex": "#c68642", "k": 0}, {"hex": "#c68642", "k": 99}, {}])
def test_match_rejects_bad_input(client, params):
    assert client.get("/api/v1/makeup/match", params=params).status_code == 422


# ---------- public system status ----------
def test_system_status_shape_and_consistency(client):
    j = client.get("/api/v1/system").json()
    info = client.get("/model-info").json()
    assert j["model"]["version"] == info["version"] == j["registry"]["champion"]
    assert {"auc", "brier", "ece", "n_reviews"} <= j["model"]["holdout"].keys()
    assert j["model"]["review_auc"] > 0.9
    assert j["data"]["products_scored"] == 315 and j["data"]["foundation_shades"] > 600
    assert {s["source"] for s in j["data"]["catalog"]["sources"]} == {"sephora_reviews", "open_beauty_facts"}
    assert j["pipeline"]["stages"] == ["hard constraints", "candidate retrieval", "ranking", "brand re-rank"]
    assert j["pipeline"]["verdict_bands"] == {"recommended": 80, "okay": 70}
    assert j["pipeline"]["score_weights"]["with_concerns"]["concern"] == 0.2
    counts = state["db"].counts()
    assert j["monitoring"]["consultations"] == counts["events"]
    assert j["monitoring"]["feedback_votes"] == counts["feedback"]
    assert j["monitoring"]["drift"]["status"] in {"insufficient_data", "stable", "moderate", "significant"}


def test_system_status_exposes_no_personal_or_secret_data(client, monkeypatch):
    monkeypatch.setenv("OPS_TOKEN", "s3cret")      # ops endpoints locked, /system stays public
    r = client.get("/api/v1/system")
    assert r.status_code == 200 and client.get("/api/v1/ops/stats").status_code == 401
    text = r.text.lower()
    for forbidden in ("s3cret", "tavily", "database_url", "password", "skin_hex", "request_id", "ip_address"):
        assert forbidden not in text


# ---------- makeup suggestions without a consultation ----------
def test_makeup_suggest_returns_the_same_shape_as_a_consultation_and_logs_nothing(client):
    before = state["db"].counts()["events"]
    j = client.post("/api/v1/makeup/suggest", json={"skin_hex": "#c68642", "skin_tone": "Tan",
                                                    "colour_preferences": ["berry", "bold"]}).json()
    assert j["depth"] and j["undertone"] and j["foundation"]["status"] == "live" and len(j["foundation"]["items"]) == 5
    for kind in ("lips", "blush", "eyes"):
        assert len(j[kind]["palette"]) == 4
    assert any("berry" in p["why"] for p in j["lips"]["palette"])
    assert state["db"].counts()["events"] == before


def test_makeup_suggest_matches_the_consultation_for_the_same_inputs(client):
    body = {"skin_hex": "#8d5524", "undertone": "cool", "colour_preferences": ["nude"]}
    a = client.post("/api/v1/makeup/suggest", json=body).json()
    b = client.post("/api/v1/consult", json={**body, "per_category": 1}).json()["makeup"]
    assert a == b


def test_makeup_suggest_with_only_a_tone_and_validation(client):
    j = client.post("/api/v1/makeup/suggest", json={"skin_tone": "Deep"}).json()
    assert j["depth"] == "deep" and j["foundation"]["status"] == "planned" and j["lips"]["palette"]
    nothing = client.post("/api/v1/makeup/suggest", json={}).json()
    assert nothing["lips"]["status"] == "planned" and nothing["depth"] is None
    for bad in ({"skin_hex": "zzz"}, {"skin_tone": "Purple"}, {"undertone": "olive"}, {"colour_preferences": ["neon"]}):
        assert client.post("/api/v1/makeup/suggest", json=bad).status_code == 422
