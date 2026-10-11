import pytest
from fastapi.testclient import TestClient

from cosmetic_rec import Recommender, load_reviews
from cosmetic_rec.api import app
from cosmetic_rec.reviews_nlp import (
    ASPECTS,
    ReviewIntel,
    ReviewStats,
    dedupe_reviews,
    make_sentiment_pipeline,
    sentences,
)
from cosmetic_rec.score import PENALTY_PER_FLAG, combine, confidence, review_component, verdict


@pytest.fixture(scope="module")
def reviews():
    return load_reviews()


@pytest.fixture(scope="module")
def intel(reviews):
    return ReviewIntel().fit(reviews)


# ---------- score formula ----------
def test_score_is_monotonic_and_bounded():
    low, high = combine(0.6, 0.6)["score"], combine(0.85, 0.9)["score"]
    assert 0 <= low < high <= 100
    assert combine(1.0, 1.0)["score"] == 100 and combine(0.0, 0.0)["score"] == 0


def test_points_explain_the_score_exactly():
    r = combine(0.8, 0.7, concern_fit=0.5, n_flags=1)
    assert r["weights"] == {"model": 0.4, "reviews": 0.4, "concern": 0.2}
    assert round(sum(r["points"].values()) - r["penalty"]) == r["score"] and r["penalty"] == PENALTY_PER_FLAG
    base = combine(0.8, 0.7)
    assert base["weights"] == {"model": 0.5, "reviews": 0.5} and "concern" not in base["points"]


def test_flag_penalty_is_capped():
    assert combine(0.8, 0.8, 0.5, 2)["penalty"] == combine(0.8, 0.8, 0.5, 9)["penalty"]


def test_verdict_bands_and_confidence():
    assert [verdict(s) for s in (85, 80, 75, 70, 69)] == ["Recommended", "Recommended", "Okay", "Okay", "Caution"]
    assert [confidence(n) for n in (3, 20, 60)] == ["low", "medium", "high"]


def test_review_component_smooths_small_samples_and_uses_skin_match_only_when_enough():
    tiny = {"n": 3, "pos": 3, "n_skin": 0, "pos_skin": 0}
    big = {"n": 200, "pos": 200, "n_skin": 0, "pos_skin": 0}
    assert review_component(tiny, 0.75)[0] < review_component(big, 0.75)[0]
    few = {"n": 20, "pos": 15, "n_skin": 2, "pos_skin": 0}
    enough = {"n": 20, "pos": 15, "n_skin": 6, "pos_skin": 0}
    assert review_component(few, 0.75, "Dry")[1] == "all reviewers"
    share, basis = review_component(enough, 0.75, "Dry")
    assert basis == "dry-skin reviewers" and share < review_component(enough, 0.75)[0]


# ---------- calibration ----------
def test_probabilities_are_calibrated_not_overconfident(reviews):
    p = Recommender().fit(reviews).predict_like("Dry")
    assert 0.65 < p.mean() < 0.85 and p.max() < 0.95


# ---------- NLP ----------
def test_sentiment_model_separates_obvious_cases(reviews):
    r = dedupe_reviews(reviews)
    m = make_sentiment_pipeline().fit(r["Review"], (r["Rating_Stars"] >= 4).astype(int))
    good, bad = m.predict_proba(["I love this, my skin feels amazing and so soft.",
                                 "Terrible, it broke me out and burned my face."])[:, 1]
    assert good > 0.8 > 0.3 > bad


def test_duplicate_reviews_are_dropped_before_counting(reviews):
    assert len(dedupe_reviews(reviews)) < len(reviews)
    stats = ReviewStats().fit(reviews)
    pid = int(reviews["Product_id"].iloc[0])
    assert stats.counts(pid)["n"] <= int((reviews["Product_id"] == pid).sum())


def test_sentence_splitter_drops_fragments():
    text = "Great! I love how smooth it makes my skin feel. Ok."
    assert sentences(text) == ["I love how smooth it makes my skin feel."]


def test_themes_are_relative_and_cover_both_directions(intel):
    verdicts = {t.verdict for pid in intel.reviews["Product_id"].unique() for t in intel.themes(int(pid))}
    assert verdicts == {"praised", "criticised", "typical"}
    assert all(t.aspect in ASPECTS and t.mentions >= 3 for t in intel.themes(int(intel.reviews["Product_id"].iloc[0])))


def test_quotes_are_short_and_aspect_specific(intel):
    for pid in map(int, intel.reviews["Product_id"].unique()[:30]):
        for q in intel.quotes(pid):
            assert 40 <= len(q["text"]) <= 170 and q["sentiment"] in ("positive", "critical")


# ---------- API ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _pid(client):
    return client.get("/skincare/products?q=cream&limit=1").json()[0]["Product_id"]


def test_review_summary_endpoint_is_live_and_skin_aware(client):
    pid = _pid(client)
    j = client.get(f"/api/v1/products/{pid}/reviews/summary", params={"skin_type": "Dry"}).json()
    assert j["status"] == "live" and j["n_reviews"] > 5 and 0 <= j["sentiment"] <= 1
    assert j["themes"] and j["quotes"] and j["confidence"] in ("low", "medium", "high")
    assert j["skin_type_reviews"] >= 0
    assert client.get(f"/api/v1/products/{pid}/reviews/summary", params={"skin_type": "Purple"}).status_code == 422
    assert client.get("/api/v1/products/999999/reviews/summary").status_code == 404


def test_score_endpoint_breakdown_adds_up(client):
    pid = _pid(client)
    j = client.get(f"/api/v1/products/{pid}/score", params={"skin_type": "Oily"}).json()
    b = j["breakdown"]
    assert 0 <= j["aicosmetic_score"] <= 100 and b["review_basis"] and "model" in b["points"]
    assert round(sum(b["points"].values()) - b["penalty"]) == j["aicosmetic_score"]


def test_concerns_change_the_score_through_fit_and_flags(client):
    pid = _pid(client)
    plain = client.get(f"/api/v1/products/{pid}/score", params={"skin_type": "Dry"}).json()
    params = {"skin_type": "Dry", "concerns": ["sensitivity", "acne"]}
    with_c = client.get(f"/api/v1/products/{pid}/score", params=params).json()
    assert "concern" in with_c["breakdown"]["points"] and "concern" not in plain["breakdown"]["points"]
    assert client.get(f"/api/v1/products/{pid}/score", params={"concerns": ["bogus"]}).status_code == 422


def test_consult_scores_match_the_score_endpoint(client):
    j = client.post("/api/v1/consult", json={"skin_type": "Dry", "per_category": 1}).json()
    item = j["skincare"]["Moisturizer"][0]
    assert item["score_status"] == "live" and j["skincare_status"] == "live"
    again = client.get(f"/api/v1/products/{item['product_id']}/score", params={"skin_type": "Dry"}).json()
    assert again["aicosmetic_score"] == item["aicosmetic_score"]
    assert round(sum(item["breakdown"]["points"].values()) - item["breakdown"]["penalty"]) == item["aicosmetic_score"]


def test_verdict_bands_split_the_catalog_sensibly(client):
    """Over all products, about a fifth should be Recommended and some Caution (top picks alone are all Recommended)."""
    ids = [p["Product_id"] for p in client.get("/skincare/products?limit=400").json()]
    verdicts = [client.get(f"/api/v1/products/{i}/score", params={"skin_type": "Dry"}).json()["verdict"] for i in ids]
    share = {v: verdicts.count(v) / len(verdicts) for v in ("Recommended", "Okay", "Caution")}
    assert all(v > 0.05 for v in share.values()) and 0.10 < share["Recommended"] < 0.35
