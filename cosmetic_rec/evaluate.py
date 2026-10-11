"""Honest offline evaluation (run: python -m cosmetic_rec.evaluate).

A. Classifier: 5-fold CV grouped by user, ROC-AUC for "rated >= 4 stars", vs. baselines.
B. Retrieval: for each user with >=2 distinct reviewed products, hide one liked product, rank everything the
   user has not reviewed, report HitRate@10 / NDCG@10 (random is ~3% at 10 of ~300).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_reviews
from .recommender import Recommender, UserProfile

warnings.filterwarnings("ignore")


def classifier_cv(reviews, seed=0):
    y = (reviews["Rating_Stars"] >= 4).astype(int).to_numpy()
    groups = reviews["User_id"].to_numpy()
    base = Recommender(seed=seed).fit(reviews)
    X = base._design(reviews["Skin_Type"], reviews["Product_id"].to_numpy(), reviews["Skin_Tone"])
    site = base.pfeat.loc[reviews["Product_id"], "site_rating"].to_numpy()
    models = {
        "baseline: site rating": None,
        "logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(C=0.1, max_iter=2000)),
        "neural net (MLP)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier(
            (64, 32), alpha=0.1, early_stopping=True, max_iter=300, random_state=seed)),
        "gradient boosting (used)": base.clf,
    }
    res = {}
    for name, m in models.items():
        aucs = []
        for tr, te in GroupKFold(5).split(X, y, groups):
            s = site[te] if m is None else m.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], s))
        res[name] = {"AUC": np.mean(aucs), "std": np.std(aucs)}
    return pd.DataFrame(res).T


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
        ranks = [rec.recommend(UserProfile(skin_type=skin.get(u), liked_product_ids=liked.get(u, []),
                                           exclude_product_ids=seen[u]), k=len(products), explain=False)
                 .index.to_numpy() for u in users]
        results[name] = _metrics(ranks, targets, k)
    return pd.DataFrame(results).T, len(users)


if __name__ == "__main__":
    df = load_reviews()
    print("A. Will this user rate it >= 4 stars? (5-fold CV grouped by user)\n")
    print(classifier_cv(df).round(3).to_string())
    table, n = retrieval(df)
    print(f"\nB. Hidden-product retrieval ({n} users, ~{df.Product_id.nunique()} candidates)\n")
    print(table.round(3).to_string())
