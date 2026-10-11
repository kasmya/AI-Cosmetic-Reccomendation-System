import pytest

from cosmetic_rec import Recommender, UserProfile, load_reviews
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


def test_sorted_explained_and_valid_probabilities(rec):
    out = rec.recommend(UserProfile(skin_type="Dry"), k=5)
    assert out["score"].is_monotonic_decreasing and out["why"].str.len().gt(0).all()
    assert out["p_like"].between(0, 1).all()


def test_excluded_products_never_returned(rec):
    pid = int(rec.products.index[0])
    out = rec.recommend(UserProfile(exclude_product_ids=[pid], liked_product_ids=[pid]), k=300)
    assert pid not in out.index


def test_liked_product_pulls_similar_ingredients(rec):
    pid = int(rec.products.index[rec.products["Ingredients"] != ""][0])
    out = rec.recommend(UserProfile(liked_product_ids=[pid], exclude_product_ids=[pid]), k=300, explain=False)
    assert out["similarity"].iloc[:10].mean() > out["similarity"].mean()


def test_skin_tone_input_is_accepted(rec):
    out = rec.recommend(UserProfile(skin_type="Dry", skin_tone="Deep"), k=5, explain=False)
    assert len(out) == 5 and out["p_like"].between(0, 1).all()


def test_split_has_no_target_leak(reviews):
    train, held = split_leave_one_out(reviews)
    seen = set(zip(train["User_id"], train["Product_id"], strict=True))
    assert not any((u, p) in seen for u, p in zip(held["User_id"], held["Product_id"], strict=True))
    assert held["Rating_Stars"].ge(4).all()
