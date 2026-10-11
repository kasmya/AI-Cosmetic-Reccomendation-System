import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec import Recommender, UserProfile, load_reviews
from cosmetic_rec.api import app
from cosmetic_rec.concerns import CONCERNS, assess, helpful_hits, public_questionnaire, validate_answers, watch_hits
from cosmetic_rec.skin_hints import blemish_score, redness_score, shine_score

skimage_data = pytest.importorskip("skimage.data")
rng = np.random.default_rng(0)


# ---------- questionnaire scoring ----------
def test_questionnaire_covers_every_concern_and_all_answers_validate():
    qs = public_questionnaire()
    assert len(qs) >= 6
    validate_answers({q["id"]: q["options"][0]["value"] for q in qs})
    scored = {c for q in __import__("cosmetic_rec.concerns", fromlist=["QUESTIONS"]).QUESTIONS for o in q["options"]
              for c in o["concerns"]}
    assert scored == set(CONCERNS)


def test_answers_infer_skin_type_and_concerns_with_reasons():
    skin, source, found = assess({"midday": "tight", "breakouts": "often", "reaction": "sometimes"})
    assert skin == "Dry" and source == "questionnaire"
    by = {c.concern: c for c in found}
    assert by["acne"].level == "likely" and by["dryness"].level == "likely" and by["sensitivity"].level == "possible"
    assert all(c.reasons for c in found)


def test_user_stated_skin_type_wins_and_selected_concerns_included():
    skin, source, found = assess({"midday": "tight"}, ["dark_spots"], skin_type="Oily")
    assert skin == "Oily" and source == "you told us"
    assert any(c.concern == "dark_spots" and c.level == "selected" for c in found)


def test_unknown_answer_rejected():
    with pytest.raises(ValueError):
        assess({"midday": "purple"})


def test_photo_hints_alone_never_exceed_possible():
    _, _, found = assess({}, [], {"redness": {"level": "high"}, "shine": {"level": "high"},
                                  "blemishes": {"level": "high"}})
    assert found and all(c.level == "possible" for c in found)
    assert all("low confidence" in c.reasons[0] for c in found)


# ---------- ingredient matching ----------
def test_ingredient_matching():
    assert helpful_hits("acne", "water, niacinamide, salicylic acid, glycerin") == ["salicylic acid", "niacinamide"]
    assert helpful_hits("dryness", "water, ceramide np, sodium hyaluronate") == ["ceramides", "hyaluronic acid"]
    assert watch_hits("sensitivity", "water, fragrance, glycerin") == ["fragrance"]
    assert watch_hits("dark_spots", "fragrance") == []


# ---------- OpenCV photo hints on synthetic patches ----------
def patch(h=50, w=45):
    return np.clip(np.full((h, w, 3), (166, 189, 224), np.uint8) + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)


def test_clean_skin_scores_low_everywhere():
    cheeks = [patch(), patch()]
    assert redness_score(cheeks)["level"] == "low" and blemish_score(cheeks)["count"] == 0
    assert shine_score(patch(100, 80))["level"] == "low"


def test_spots_redness_and_highlights_are_detected():
    spotty = [patch(), patch()]
    for q in spotty:
        for _ in range(6):
            cv2.circle(q, (int(rng.integers(8, 37)), int(rng.integers(8, 42))), 2, (135, 140, 205), -1)
    assert blemish_score(spotty)["level"] == "high"
    red = [patch(), patch()]
    for q in red:
        q[:, :14] = (120, 120, 215)
    assert redness_score(red)["level"] == "high"
    t = patch(100, 80)
    for y, x in [(15, 20), (30, 50), (60, 30), (75, 55), (45, 10), (85, 15)]:
        t[y:y + 9, x:x + 9] = (248, 248, 248)
    assert shine_score(t)["level"] == "high"


# ---------- concerns change the ranking ----------
@pytest.fixture(scope="module")
def rec():
    return Recommender().fit(load_reviews())


def _hits(rec, ids, concern):
    return float(np.mean([len(helpful_hits(concern, rec.products.at[i, "Ingredients"])) for i in ids]))


def test_acne_concern_promotes_helpful_ingredients(rec):
    base = rec.recommend(UserProfile(skin_type="Oily"), k=10, explain=False)
    acne = rec.recommend(UserProfile(skin_type="Oily", concerns=["acne"]), k=10, explain=False)
    assert _hits(rec, acne.index, "acne") > _hits(rec, base.index, "acne")


def test_sensitivity_concern_demotes_fragrance_products(rec):
    def flagged(profile):
        out = rec.recommend(profile, k=10, explain=False)
        return sum(bool(watch_hits("sensitivity", rec.products.at[i, "Ingredients"])) for i in out.index)

    assert flagged(UserProfile(concerns=["sensitivity"])) < flagged(UserProfile())


def test_explanation_mentions_concern_ingredients(rec):
    out = rec.recommend(UserProfile(concerns=["acne"]), k=10)
    assert out["why"].str.contains("often used for acne").any()


# ---------- API ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_questionnaire_endpoint(client):
    qs = client.get("/api/v1/skin/questionnaire").json()
    assert qs[0]["id"] == "midday" and "types" not in qs[0]["options"][0]


def test_assess_endpoint_full(client):
    r = client.post("/api/v1/skin/assess", json={"answers": {"midday": "oily_all", "breakouts": "often"},
                                                 "concerns": ["dark_spots"]})
    j = r.json()
    assert r.status_code == 200 and j["status"] == "live" and j["skin_type"] == "Oily"
    acne = next(c for c in j["concerns"] if c["concern"] == "acne")
    assert acne["level"] == "likely" and "salicylic acid" in acne["helpful_ingredients"] and acne["reasons"]
    assert client.post("/api/v1/skin/assess", json={"answers": {"midday": "nope"}}).status_code == 422
    assert client.post("/api/v1/skin/assess", json={"concerns": ["bogus"]}).status_code == 422


def test_consult_uses_inferred_skin_type_and_concerns(client):
    j = client.post("/api/v1/consult", json={"answers": {"midday": "tight", "breakouts": "often"},
                                             "per_category": 2}).json()
    assert j["profile"]["skin_type"] == "Dry"
    assert {c["concern"] for c in j["profile"]["concerns"]} >= {"acne", "dryness"}
    items = [i for v in j["skincare"].values() for i in v]
    assert any(i["breakdown"]["concern_fit"] for i in items)


def test_face_analyze_returns_hints_and_they_feed_assess(client):
    ok, png = cv2.imencode(".png", cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR))
    face = client.post("/api/v1/face/analyze", content=png.tobytes()).json()
    assert {"redness", "shine", "blemishes", "confidence"} <= face["hints"].keys()
    assert face["hints"]["confidence"] == "low"
    r = client.post("/api/v1/skin/assess", json={"photo_hints": face["hints"]}).json()
    assert r["photo_hints_used"] is True
