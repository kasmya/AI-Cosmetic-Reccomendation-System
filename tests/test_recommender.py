import pytest
from fastapi.testclient import TestClient

from cosmetic_rec import Recommender, UserProfile, load_reviews
from cosmetic_rec.api import app
from cosmetic_rec.evaluate import split_leave_one_out


@pytest.fixture(scope="module")
def reviews():
    return load_reviews()


@pytest.fixture(scope="module")
def rec(reviews):
    return Recommender().fit(reviews)


def test_filters_are_hard_constraints(rec):
    out = rec.recommend(UserProfile(category="Cleanser", max_price=30, avoid_ingredients=["fragrance"]), k=20)
    assert len(out) > 0
    assert (out["Category"] == "Cleanser").all() and (out["Price"] <= 30).all()
    assert not rec.products.loc[out.index, "Ingredients"].str.lower().str.contains("fragrance").any()


def test_results_sorted_explained_and_probabilities_valid(rec):
    out = rec.recommend(UserProfile(skin_type="Dry"), k=5)
    assert out["score"].is_monotonic_decreasing and out["why"].str.len().gt(0).all()
    assert out["p_like"].between(0, 1).all()


def test_excluded_products_never_returned(rec):
    pid = int(rec.products.index[0])
    out = rec.recommend(UserProfile(exclude_product_ids=[pid], liked_product_ids=[pid]), k=300)
    assert pid not in out.index


def test_liked_product_pulls_similar_ingredients(rec):
    pid = int(rec.products.index[0])
    with_like = rec.recommend(UserProfile(liked_product_ids=[pid], exclude_product_ids=[pid]), k=300, explain=False)
    assert with_like["similarity"].iloc[:10].mean() > with_like["similarity"].mean()


def test_split_has_no_target_leak(reviews):
    train, held = split_leave_one_out(reviews)
    seen = set(zip(train["User_id"], train["Product_id"]))
    assert not any((u, p) in seen for u, p in zip(held["User_id"], held["Product_id"]))
    assert held["Rating_Stars"].ge(4).all()


def test_api():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/recommend", json={"skin_type": "Oily", "category": "Moisturizer", "k": 3})
        assert r.status_code == 200 and r.json()["count"] == 3
        assert c.post("/recommend", json={"skin_type": "Purple"}).status_code == 422
        assert c.get("/").status_code == 200


# ---- OpenCV skin-tone analysis ----
import cv2
import numpy as np

from cosmetic_rec.vision import NoFaceError, ita_to_tone, estimate_skin_tone, tone_index


def test_ita_mapping_is_monotonic():
    tones = [ita_to_tone(a) for a in (70, 55, 45, 30, 22, 15, 0, -20, -50)]
    assert [tone_index(t) for t in tones] == list(range(9))


def test_tone_changes_model_input_but_not_validity(rec):
    base = rec.recommend(UserProfile(skin_type="Dry"), k=5, explain=False)
    dark = rec.recommend(UserProfile(skin_type="Dry", skin_tone="Deep"), k=5, explain=False)
    assert dark["p_like"].between(0, 1).all() and len(base) == len(dark) == 5


def test_face_photo_and_no_face():
    data = pytest.importorskip("skimage.data")
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    out = estimate_skin_tone(face)
    assert out["tone"] in ("Porcelain", "Fair", "Light") and out["skin_pixels"] > 100
    with pytest.raises(NoFaceError):
        estimate_skin_tone(cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR))


def test_photo_endpoint():
    data = pytest.importorskip("skimage.data")
    ok, png = cv2.imencode(".png", cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    with TestClient(app) as c:
        r = c.post("/analyze-photo", content=png.tobytes())
        assert r.status_code == 200 and "tone" in r.json()
        assert c.post("/analyze-photo", content=b"not an image").status_code == 400
        assert c.post("/recommend", json={"skin_tone": "Purple"}).status_code == 422
