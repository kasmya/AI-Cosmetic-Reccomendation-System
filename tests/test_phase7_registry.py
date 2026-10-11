import numpy as np
import pandas as pd
import pytest

from cosmetic_rec import registry, train
from cosmetic_rec.data import build_products, load_reviews
from cosmetic_rec.registry import ece, gate, paired_auc_interval, split_gold


# ---------- gold holdout ----------
def test_gold_holdout_is_fixed_disjoint_and_about_a_fifth():
    reviews = load_reviews()
    tr, gold = split_gold(reviews)
    assert not set(tr["User_id"]) & set(gold["User_id"])
    assert 0.15 < len(gold) / len(reviews) < 0.25
    assert split_gold(reviews)[1].index.equals(gold.index)


# ---------- metrics ----------
def test_ece_is_zero_for_perfect_calibration_and_large_for_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    assert ece((rng.uniform(size=p.size) < p).astype(int), p) < 0.02
    assert ece(np.zeros(1000, int), np.full(1000, 0.9)) > 0.8


def test_paired_bootstrap_interval_behaves():
    rng = np.random.default_rng(1)
    users = np.repeat(np.arange(300), 4)
    y = rng.integers(0, 2, users.size)
    good = y * 0.4 + rng.normal(0, 0.3, y.size)
    noisy = rng.normal(0, 1, y.size)
    mean, lo, hi = paired_auc_interval(y, good, noisy, users, n_boot=200)
    assert lo > 0 and mean > 0.2                          # clearly better: whole interval above zero
    mean, lo, hi = paired_auc_interval(y, good, good, users, n_boot=200)
    assert mean == 0 and lo == 0 == hi                    # identical predictions: no difference


# ---------- gate ----------
OLD = {"auc": 0.60, "brier": 0.180, "ece": 0.020}


def verdict(new, diff):
    return gate(new, OLD, diff)


def test_gate_promotes_an_equal_model_but_does_not_call_it_better():
    d = verdict({"auc": 0.601, "brier": 0.180, "ece": 0.020}, (0.001, -0.008, 0.01))
    assert d["promote"] and not d["strictly_better"]


def test_gate_flags_a_strictly_better_model():
    assert verdict({"auc": 0.62, "brier": 0.178, "ece": 0.015}, (0.02, 0.005, 0.035))["strictly_better"]


@pytest.mark.parametrize("new,diff,failed", [
    ({"auc": 0.57, "brier": 0.180, "ece": 0.020}, (-0.03, -0.05, -0.01), "auc_non_inferior"),
    ({"auc": 0.60, "brier": 0.190, "ece": 0.020}, (0.0, -0.005, 0.005), "brier_not_worse"),
    ({"auc": 0.60, "brier": 0.180, "ece": 0.050}, (0.0, -0.005, 0.005), "calibration_holds"),
])
def test_gate_rejects_each_kind_of_regression(new, diff, failed):
    d = verdict(new, diff)
    assert not d["promote"] and d["checks"][failed] is False


# ---------- storage, promotion, rollback (stub model, no training) ----------
class StubModel:
    """Returns fixed probabilities so registry mechanics can be tested without training."""
    def __init__(self, p):
        self.p = p

    def _design(self, skin, pid, tone):
        return pd.DataFrame(index=range(len(pid)))

    @property
    def clf(self):
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - self.p[: len(X)], self.p[: len(X)]])


@pytest.fixture
def reg(tmp_path):
    rng = np.random.default_rng(0)
    gold = pd.DataFrame({"User_id": np.repeat(np.arange(200), 5), "Product_id": 1, "Skin_Type": "Dry",
                         "Skin_Tone": np.nan, "Rating_Stars": rng.integers(1, 6, 1000)})
    return tmp_path / "reg", gold, (gold["Rating_Stars"] >= 4).to_numpy().astype(float)


def put(root, name, p, gold):
    blob = {"model": StubModel(p), "intel": None, "meta": {"n_reviews": len(gold)}}
    return registry.register(blob, {"auc": 0}, {}, name, root)


def test_first_model_becomes_champion_then_a_worse_one_is_rejected_and_rollback_works(reg):
    root, gold, y = reg
    good = np.clip(y * 0.5 + 0.25, 0.01, 0.99)
    bad = np.clip(0.75 - y * 0.5, 0.01, 0.99)            # anti-correlated with the truth
    put(root, "v-good", good, gold)
    assert registry.evaluate_and_promote("v-good", gold, root=root)["promote"] and registry.champion(root) == "v-good"
    put(root, "v-bad", bad, gold)
    d = registry.evaluate_and_promote("v-bad", gold, root=root)
    assert not d["promote"] and registry.champion(root) == "v-good"
    put(root, "v-good2", good * 0.99 + 0.005, gold)
    assert registry.evaluate_and_promote("v-good2", gold, root=root)["promote"]
    assert registry.rollback(root) == "v-good"
    events = [e["event"] for e in registry.history(root)]
    assert events.count("rejected") == 1 and "rollback" in events


def test_force_promotes_a_rejected_model_and_records_it(reg):
    root, gold, y = reg
    put(root, "v-a", np.clip(y * 0.5 + 0.25, 0.01, 0.99), gold)
    registry.evaluate_and_promote("v-a", gold, root=root)
    put(root, "v-b", np.clip(0.75 - y * 0.5, 0.01, 0.99), gold)
    d = registry.evaluate_and_promote("v-b", gold, force=True, root=root)
    assert d["promote"] and d["forced"] and registry.champion(root) == "v-b"


def test_prune_keeps_the_champion_and_the_newest(reg):
    root, gold, y = reg
    for name in ("v1", "v2", "v3", "v4"):
        put(root, name, np.clip(y * 0.5 + 0.25, 0.01, 0.99), gold)
    registry.evaluate_and_promote("v1", gold, root=root)
    assert set(registry.prune(keep=2, root=root)) == {"v2"}
    assert set(registry.versions(root)) == {"v1", "v3", "v4"}


def test_rollback_with_nothing_earlier_raises(reg):
    root, gold, y = reg
    put(root, "only", y * 0.5 + 0.25, gold)
    registry.evaluate_and_promote("only", gold, root=root)
    with pytest.raises(RuntimeError):
        registry.rollback(root)


# ---------- feedback becomes training data ----------
def test_feedback_rows_become_pseudo_reviews_with_negative_ids():
    products = build_products(load_reviews())
    pid = int(products.index[0])
    rows = [{"product_id": pid, "helpful": True, "skin_type": "Dry", "event_id": "e1"},
            {"product_id": pid, "helpful": False, "skin_type": None, "event_id": "e1"},
            {"product_id": 999999, "helpful": True, "skin_type": "Dry", "event_id": "e2"}]
    out = train.feedback_to_reviews(rows, products)
    assert len(out) == 2 and (out["User_id"] < 0).all()
    assert out["User_id"].nunique() == 1                       # one consultation = one pseudo-user
    assert list(out["Rating_Stars"]) == [5, 1]


# ---------- end to end with real training (slow: three models) ----------
def test_real_training_registers_promotes_and_rejects_poisoned_feedback(tmp_path):
    root = tmp_path / "registry"
    first = train.train(root=root)
    assert first["decision"]["promote"] and registry.champion(root) == first["version"]
    assert registry.version_metrics(first["version"], root)["holdout"]["n_reviews"] > 1000
    reviews = load_reviews()
    tr, _ = split_gold(reviews)
    sample = tr.sample(2500, random_state=0)
    poisoned = [{"product_id": int(p), "helpful": bool(s < 4), "skin_type": k if isinstance(k, str) else None,
                 "event_id": f"e{i}"} for i, (p, s, k) in enumerate(zip(sample.Product_id, sample.Rating_Stars,
                                                                        sample.Skin_Type, strict=True))]
    second = train.train(feedback_rows=poisoned, root=root)
    assert second["feedback_rows_used"] == 2500
    assert not second["decision"]["promote"] and registry.champion(root) == first["version"]
    blob = train.load_all(root)
    assert blob["meta"]["version"] == first["version"] and blob["meta"]["holdout"]["auc"] == first["holdout"]["auc"]
