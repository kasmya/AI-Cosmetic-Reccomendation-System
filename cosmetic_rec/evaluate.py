"""Honest offline evaluation (run: python -m cosmetic_rec.evaluate).

A. Classifier: 5-fold CV grouped by user, ROC-AUC for "rated >= 4 stars", vs. baselines.
B. Retrieval: for each user with >=2 distinct reviewed products, hide one liked product, rank everything the
   user has not reviewed, report HitRate@10 / NDCG@10 (random is ~3% at 10 of ~300).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import SKIN_TYPES, load_reviews
from .recommender import Recommender, UserProfile
from .reviews_nlp import ReviewStats, dedupe_reviews, make_sentiment_pipeline
from .score import combine, review_component

warnings.filterwarnings("ignore")


def _ece(y, p, bins=10) -> float:
    """Expected calibration error: gap between predicted probability and observed rate, averaged over bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    gaps = [abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()]
    return float(sum(gaps))


def classifier_cv(reviews, seed=0):
    y = (reviews["Rating_Stars"] >= 4).astype(int).to_numpy()
    groups = reviews["User_id"].to_numpy()
    base = Recommender(seed=seed).fit(reviews)
    X = base._design(reviews["Skin_Type"], reviews["Product_id"].to_numpy(), reviews["Skin_Tone"])
    site = base.pfeat.loc[reviews["Product_id"], "site_rating"].to_numpy()

    def hgb():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)

    models = {
        "baseline: site rating": None,
        "logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                              LogisticRegression(C=0.1, max_iter=2000)),
        "neural net (MLP)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier(
            (64, 32), alpha=0.1, early_stopping=True, max_iter=300, random_state=seed)),
        "gradient boosting": hgb(),
        "gradient boosting + calibration (used)": "calibrated",
    }
    res = {}
    for name, m in models.items():
        auc, brier, ece = [], [], []
        for tr, te in GroupKFold(5).split(X, y, groups):
            if m is None:
                s = site[te]
            else:
                model = m
                if m == "calibrated":
                    inner = list(GroupKFold(3).split(X.iloc[tr], y[tr], groups[tr]))
                    model = CalibratedClassifierCV(hgb(), method="sigmoid", cv=inner)
                s = model.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
                brier.append(brier_score_loss(y[te], s))
                ece.append(_ece(y[te], s))
            auc.append(roc_auc_score(y[te], s))
        res[name] = {"AUC": np.mean(auc), "std": np.std(auc), "Brier": np.mean(brier) if brier else np.nan,
                     "ECE": np.mean(ece) if ece else np.nan}
    return pd.DataFrame(res).T


def sentiment_cv(reviews, seed=0):
    """Review text -> rated 4-5 stars? Users never appear in both train and test."""
    r = dedupe_reviews(reviews).reset_index(drop=True)
    y = (r["Rating_Stars"] >= 4).astype(int).to_numpy()
    auc, acc, f1 = [], [], []
    for tr, te in GroupKFold(5).split(r, y, r["User_id"]):
        p = make_sentiment_pipeline(seed).fit(r["Review"].iloc[tr], y[tr]).predict_proba(r["Review"].iloc[te])[:, 1]
        auc.append(roc_auc_score(y[te], p))
        acc.append(accuracy_score(y[te], p > 0.5))
        f1.append(f1_score(y[te], p > 0.5, pos_label=0))
    return {"AUC": np.mean(auc), "accuracy": np.mean(acc), "majority-class accuracy": max(y.mean(), 1 - y.mean()),
            "F1 (negative reviews)": np.mean(f1), "reviews": len(r)}


def score_validity(reviews, seed=0):
    """Does the AICosmetic score predict held-out reviews? Every part is computed from the training users only."""
    y = (reviews["Rating_Stars"] >= 4).astype(int).to_numpy()
    products = Recommender(seed=seed).fit(reviews).products
    names = ("model", "reviews (all)", "reviews (skin-matched)", "AICosmetic score")
    parts = {k: np.zeros(len(reviews)) for k in names}
    for tr, te in GroupKFold(5).split(reviews, y, reviews["User_id"]):
        train, test = reviews.iloc[tr], reviews.iloc[te]
        rec = Recommender(seed=seed).fit(train, products)
        stats = ReviewStats().fit(train)
        like = {st: rec.predict_like(st) for st in [*SKIN_TYPES, None]}
        for pos, (pid, st) in zip(te, zip(test["Product_id"], test["Skin_Type"], strict=True), strict=True):
            st = st if isinstance(st, str) else None
            p = float(like[st][pid])
            c = stats.counts(pid, st)
            all_share, _ = review_component(c, stats.global_share, None)
            skin_share, _ = review_component(c, stats.global_share, st)
            parts["model"][pos], parts["reviews (all)"][pos] = p, all_share
            parts["reviews (skin-matched)"][pos] = skin_share
            parts["AICosmetic score"][pos] = combine(p, skin_share)["score"]
    out = {k: roc_auc_score(y, v) for k, v in parts.items()}
    out["baseline: site rating"] = roc_auc_score(y, products.loc[reviews["Product_id"], "Rating"].to_numpy())
    return pd.Series(out, name="AUC")


def split_leave_one_out(reviews, seed=0):
    """Hide one liked product per user, removing *every* review that user wrote for it
    (the dataset has repeat user/product reviews that would otherwise leak the target)."""
    distinct = reviews.groupby("User_id")["Product_id"].nunique()
    eligible = reviews["User_id"].isin(distinct[distinct >= 2].index) & (reviews["Rating_Stars"] >= 4)
    held = reviews[eligible].groupby("User_id").sample(n=1, random_state=seed)
    pairs = pd.MultiIndex.from_frame(held[["User_id", "Product_id"]])
    drop = pd.MultiIndex.from_frame(reviews[["User_id", "Product_id"]]).isin(pairs)
    return reviews[~drop], held


def _metrics(ranked_lists, targets, k=10):
    hit, ndcg = [], []
    for ranked, t in zip(ranked_lists, targets, strict=True):
        pos = np.where(ranked == t)[0]
        r = pos[0] + 1 if len(pos) else None
        hit.append(float(bool(r and r <= k)))
        ndcg.append(1 / np.log2(r + 1) if r and r <= k else 0.0)
    return {f"HitRate@{k}": np.mean(hit), f"NDCG@{k}": np.mean(ndcg)}


def retrieval(reviews, seed=0, k=10):
    train, held = split_leave_one_out(reviews, seed)
    seen = train.groupby("User_id")["Product_id"].apply(list).to_dict()
    liked = train[train["Rating_Stars"] >= 4].groupby("User_id")["Product_id"].apply(list).to_dict()
    skin = reviews.dropna(subset=["Skin_Type"]).groupby("User_id")["Skin_Type"].first().to_dict()
    targets = held["Product_id"].tolist()
    users = held["User_id"].tolist()
    products = Recommender().fit(reviews).products
    rng = np.random.default_rng(seed)

    results = {"random": _metrics(
        [np.array([i for i in rng.permutation(products.index) if i not in seen[u]]) for u in users], targets, k)}
    variants = {"ML P(like) only": dict(taste=0), "ingredient similarity only": dict(model=0),
                "hybrid (used)": dict()}
    for name, wts in variants.items():
        rec = Recommender(weights=wts, seed=seed).fit(train, products)
        def rank(u, rec=rec):
            profile = UserProfile(skin_type=skin.get(u), liked_product_ids=liked.get(u, []),
                                  exclude_product_ids=seen[u], include_makeup=True)
            return rec.recommend(profile, k=len(products), explain=False).index.to_numpy()

        ranks = [rank(u) for u in users]
        results[name] = _metrics(ranks, targets, k)
    return pd.DataFrame(results).T, len(users)


if __name__ == "__main__":
    df = load_reviews()
    print("A. Will this user rate it >= 4 stars? (5-fold CV grouped by user)\n")
    print(classifier_cv(df).round(3).to_string())
    print("\nC. Review text sentiment (positive = 4-5 stars), 5-fold CV grouped by user\n")
    print(pd.Series(sentiment_cv(df)).round(3).to_string())
    print("\nD. Does the AICosmetic score predict held-out reviews? ROC-AUC, 5-fold CV grouped by user\n")
    print(score_validity(df).round(3).to_string())
    table, n = retrieval(df)
    print(f"\nB. Hidden-product retrieval ({n} users, ~{df.Product_id.nunique()} candidates)\n")
    print(table.round(3).to_string())
