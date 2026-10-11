"""Leakage-free offline evaluation.

Protocol: for every user with >=2 reviews and at least one 4-5 star review, hold out
one liked product. Fit on the remaining reviews, rank all products the user has not
reviewed, and check where the held-out product lands. Chance level for HitRate@10
over ~300 candidates is about 3%.
"""
import numpy as np
import pandas as pd

from .data import load_reviews
from .recommender import Recommender, UserProfile


def split_leave_one_out(reviews: pd.DataFrame, seed=0):
    rng = np.random.default_rng(seed)
    counts = reviews.groupby("User_id").size()
    eligible = counts[counts >= 2].index
    liked = reviews[reviews["User_id"].isin(eligible) & (reviews["Rating_Stars"] >= 4)]
    held = liked.groupby("User_id").sample(n=1, random_state=int(rng.integers(1 << 31)))
    return reviews.drop(held.index), held


def _metrics(rank_lists, targets, k=10):
    hits, ndcg, rr = [], [], []
    for ranked, t in zip(rank_lists, targets):
        pos = np.where(ranked == t)[0]
        r = pos[0] + 1 if len(pos) else None
        hits.append(float(r is not None and r <= k))
        ndcg.append(1 / np.log2(r + 1) if r and r <= k else 0.0)
        rr.append(1 / r if r else 0.0)
    return {f"HitRate@{k}": np.mean(hits), f"NDCG@{k}": np.mean(ndcg), "MRR": np.mean(rr)}


def evaluate(reviews: pd.DataFrame, k=10, seed=0, variants=None):
    train, held = split_leave_one_out(reviews, seed)
    full = Recommender().fit(reviews)  # supplies the product table incl. every product
    products = full.products[["Product", "Brand", "Category", "Price", "Ingredients",
                              "Combination", "Dry", "Normal", "Oily", "Sensitive"]]
    seen = train.groupby("User_id")["Product_id"].apply(list).to_dict()
    liked_train = (train[train["Rating_Stars"] >= 4].groupby("User_id")["Product_id"].apply(list).to_dict())
    skin = reviews.dropna(subset=["Skin_Type"]).groupby("User_id")["Skin_Type"].first().to_dict()
    n = len(products)

    variants = variants or {
        "popularity (review count)": None,
        "quality only": dict(skin_fit=0, label_fit=0, taste=0),
        "quality + skin fit": dict(taste=0),
        "full model": dict(),
    }
    rng = np.random.default_rng(seed)
    results = {"random": _metrics(
        [np.array([t for t in rng.permutation(products.index) if t not in seen[u]])
         for u, t in zip(held["User_id"], held["Product_id"])], held["Product_id"].tolist(), k)}

    for name, weights in variants.items():
        rec = Recommender(weights=weights).fit(train, products[["Product", "Brand", "Category", "Price",
                          "Ingredients", "Combination", "Dry", "Normal", "Oily", "Sensitive"]])
        if weights is None:  # popularity baseline
            order = rec.products["n_reviews"].sort_values(ascending=False, kind="stable").index.to_numpy()
            ranks = [order[~np.isin(order, seen[u])] for u in held["User_id"]]
        else:
            ranks = []
            for u in held["User_id"]:
                prof = UserProfile(skin_type=skin.get(u), liked_product_ids=liked_train.get(u, []),
                                   exclude_product_ids=seen[u])
                ranks.append(rec.recommend(prof, k=n, explain=False).index.to_numpy())
        results[name] = _metrics(ranks, held["Product_id"].tolist(), k)
    return pd.DataFrame(results).T, len(held)


def leakage_demo(reviews: pd.DataFrame):
    """Shows why the notebook's 100% accuracy was meaningless."""
    target = (reviews["Rating_Stars"] == 5).astype(int)
    good = (reviews["Good_Stuff"] == target).mean()
    return good


if __name__ == "__main__":
    df = load_reviews()
    print(f"Good_Stuff == (Rating_Stars == 5) for {leakage_demo(df):.1%} of rows -> the old 100% was leakage.\n")
    table, n = evaluate(df)
    print(f"Leave-one-out over {n} users, ~{df.Product_id.nunique()} candidate products\n")
    print(table.round(3).to_string())
