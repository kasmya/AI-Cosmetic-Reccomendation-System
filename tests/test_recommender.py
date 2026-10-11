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


def test_results_sorted_and_explained(rec):
    out = rec.recommend(UserProfile(skin_type="Dry"), k=5)
    assert out["score"].is_monotonic_decreasing and out["why"].str.len().gt(0).all()


def test_excluded_products_never_returned(rec):
    pid = int(rec.products.index[0])
    out = rec.recommend(UserProfile(exclude_product_ids=[pid], liked_product_ids=[pid]), k=300)
    assert pid not in out.index


def test_smoothed_quality_in_range(rec):
    assert rec.products["quality"].between(1, 5).all()


def test_split_never_leaks_heldout_into_train(reviews):
    train, held = split_leave_one_out(reviews)
    assert not train.index.isin(held.index).any()
    assert held["Rating_Stars"].ge(4).all()


def test_api():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/recommend", json={"skin_type": "Oily", "category": "Moisturizer", "k": 3})
        assert r.status_code == 200 and r.json()["count"] == 3
        assert c.post("/recommend", json={"skin_type": "Purple"}).status_code == 422
