import numpy as np
import pytest

from cosmetic_rec import Recommender, UserProfile, load_reviews


@pytest.fixture(scope="module")
def rec():
    return Recommender().fit(load_reviews())


PROFILES = [UserProfile(), UserProfile(skin_type="Dry", concerns=["dryness"]),
            UserProfile(skin_type="Oily", category="Cleanser", concerns=["acne"])]


def test_trace_reports_every_stage(rec):
    out = rec.recommend(UserProfile(skin_type="Dry", category="Moisturizer"), k=5, candidate_limit=20)
    t = out.attrs["trace"]
    assert t["eligible"] >= t["candidates"] == 20 and t["returned"] == 5


def test_no_cut_when_fewer_eligible_than_the_limit(rec):
    out = rec.recommend(UserProfile(category="Face Mask"), k=3, candidate_limit=100)
    assert out.attrs["trace"]["candidates"] == out.attrs["trace"]["eligible"]


@pytest.mark.parametrize("profile", PROFILES)
def test_cutting_candidates_never_changes_a_products_score(rec, profile):
    full = rec.recommend(profile, k=400, explain=False)
    cut = rec.recommend(profile, k=400, explain=False, candidate_limit=60)
    assert len(cut) <= 60
    assert np.allclose(full.loc[cut.index, "score"], cut["score"])      # scores normalised over the whole catalog


@pytest.mark.parametrize("profile", PROFILES)
def test_a_limit_of_100_keeps_almost_the_whole_top_10(rec, profile):
    full = set(rec.recommend(profile, k=10, explain=False).index)
    cut = set(rec.recommend(profile, k=10, explain=False, candidate_limit=100).index)
    assert len(full & cut) >= 9


def test_brand_cap_is_respected_and_still_returns_k_results(rec):
    prof = UserProfile(skin_type="Dry", category="Cleanser")
    out = rec.recommend(prof, k=10, explain=False, max_per_brand=2)
    assert len(out) == 10 and out["Brand"].value_counts().max() <= 2


def test_brand_cap_keeps_score_order_among_kept_products(rec):
    out = rec.recommend(UserProfile(skin_type="Oily"), k=8, explain=False, max_per_brand=2)
    assert out["score"].is_monotonic_decreasing


def test_without_a_cap_the_rerank_is_a_plain_top_k(rec):
    prof = UserProfile(skin_type="Normal")
    assert list(rec.recommend(prof, k=7, explain=False).index) == list(
        rec.recommend(prof, k=400, explain=False).index[:7])


def test_a_tight_cap_falls_back_to_fill_k_when_few_brands_exist(rec):
    out = rec.recommend(UserProfile(category="Face Mask"), k=3, explain=False, max_per_brand=1)
    assert len(out) == min(3, out.attrs["trace"]["eligible"])


def test_ties_are_ordered_deterministically(rec):
    a = rec.recommend(UserProfile(category="Cleanser", max_price=30), k=40, explain=False)
    b = rec.recommend(UserProfile(category="Cleanser", max_price=30), k=40, explain=False)
    assert list(a.index) == list(b.index)


def test_consult_picks_respect_the_brand_cap_and_candidate_limit():
    from fastapi.testclient import TestClient

    from cosmetic_rec.api import app
    with TestClient(app) as c:
        j = c.post("/api/v1/consult", json={"skin_type": "Dry", "per_category": 5}).json()
    for items in j["skincare"].values():
        brands = [i["brand"] for i in items]
        assert max(brands.count(b) for b in set(brands)) <= 2
