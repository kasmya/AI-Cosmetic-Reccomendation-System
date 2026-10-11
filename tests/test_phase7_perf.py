"""The load test found consultations dominated by repeated work. These tests pin each optimisation to the slow,
obviously-correct computation so a cache can never change an answer."""
import io

import joblib
import numpy as np
import pytest

from cosmetic_rec import Recommender, load_reviews
from cosmetic_rec.concerns import CONCERNS, helpful_hits, watch_hits
from cosmetic_rec.reviews_nlp import ReviewStats, dedupe_reviews


@pytest.fixture(scope="module")
def reviews():
    return load_reviews()


@pytest.fixture(scope="module")
def rec(reviews):
    return Recommender().fit(reviews)


# ---------- predict_like cache ----------
def test_cached_predictions_equal_a_fresh_computation(rec):
    first = rec.predict_like("Dry", "Deep")
    rec.__dict__["_like_cache"].clear()
    second = rec.predict_like("Dry", "Deep")
    assert np.allclose(first, second)


def test_cache_returns_copies_so_callers_cannot_corrupt_it(rec):
    a = rec.predict_like("Oily")
    a[:] = 0.0
    assert rec.predict_like("Oily").max() > 0.5


def test_each_skin_type_and_tone_gets_its_own_entry(rec):
    assert not np.allclose(rec.predict_like("Dry"), rec.predict_like("Oily"))
    assert len({k for k in rec.__dict__["_like_cache"]}) >= 2


# ---------- concern arrays ----------
@pytest.mark.parametrize("concern", CONCERNS)
def test_cached_concern_arrays_match_the_direct_calculation(rec, concern):
    fit, watch = rec._concern_arrays(concern)
    ing = rec.products["Ingredients"]
    assert np.array_equal(fit, [min(len(helpful_hits(concern, t)), 2) / 2 for t in ing])
    assert np.array_equal(watch, [float(bool(watch_hits(concern, t))) for t in ing])


# ---------- review count table ----------
def test_count_table_matches_brute_force_for_every_product_and_skin_type(reviews):
    stats = ReviewStats().fit(reviews)
    r = dedupe_reviews(reviews)
    for pid in r["Product_id"].unique()[:80]:
        for skin in (None, "Dry", "Oily", "Combination", "Normal"):
            p = r[r["Product_id"] == pid]
            pos = p["Rating_Stars"] >= 4
            expect = {"n": len(p), "pos": int(pos.sum()), "n_skin": 0, "pos_skin": 0}
            if skin:
                m = p["Skin_Type"] == skin
                expect["n_skin"], expect["pos_skin"] = int(m.sum()), int((pos & m).sum())
            assert stats.counts(int(pid), skin) == expect


def test_unknown_product_has_zero_counts(reviews):
    assert ReviewStats().fit(reviews).counts(10**9, "Dry") == {"n": 0, "pos": 0, "n_skin": 0, "pos_skin": 0}


# ---------- caches are never saved with a model ----------
def test_caches_are_not_pickled_into_artifacts(rec, reviews):
    rec.predict_like("Dry")
    rec._concern_arrays("acne")
    buf = io.BytesIO()
    joblib.dump(rec, buf)
    restored = joblib.load(io.BytesIO(buf.getvalue()))
    assert "_like_cache" not in restored.__dict__ and "_concern_cache" not in restored.__dict__
    assert np.allclose(restored.predict_like("Dry"), rec.predict_like("Dry"))
    stats = ReviewStats().fit(reviews)
    stats.counts(1)
    assert "_count_table" not in joblib.load(io.BytesIO(_dump(stats))).__dict__


def _dump(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()
