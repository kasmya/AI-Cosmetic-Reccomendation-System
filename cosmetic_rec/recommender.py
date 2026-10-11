"""Transparent, review-driven product recommender.

Score for a user = weighted sum of
  quality   - Bayesian-smoothed mean star rating over all reviews
  skin_fit  - the same, but only over reviewers who share the user's skin type
              (shrunk toward `quality` when few such reviews exist)
  label_fit - the product's own "suitable for <skin type>" flag
  taste     - TF-IDF ingredient similarity to products the user already liked
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import SKIN_TYPES, build_products

DEFAULT_WEIGHTS = {"quality": 1.0, "skin_fit": 0.6, "label_fit": 0.3, "taste": 1.0}


@dataclass
class UserProfile:
    skin_type: str | None = None          # Combination / Dry / Normal / Oily
    sensitive: bool = False
    category: str | None = None           # Cleanser / Moisturizer / Treatment / Face Mask
    max_price: float | None = None
    avoid_ingredients: list[str] = field(default_factory=list)
    liked_product_ids: list[int] = field(default_factory=list)
    exclude_product_ids: list[int] = field(default_factory=list)


def _bayes_mean(sum_, n, prior_mean, prior_n):
    return (sum_ + prior_mean * prior_n) / (n + prior_n)


def _z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


class Recommender:
    def __init__(self, prior_n=10, skin_prior_n=5, weights=None):
        self.prior_n = prior_n
        self.skin_prior_n = skin_prior_n
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def fit(self, reviews: pd.DataFrame, products: pd.DataFrame | None = None):
        self.products = (products if products is not None else build_products(reviews)).copy()
        ids = self.products.index
        stars = reviews.groupby("Product_id")["Rating_Stars"].agg(["sum", "count"]).reindex(ids).fillna(0)
        self.global_mean = reviews["Rating_Stars"].mean()
        self.products["n_reviews"] = stars["count"].astype(int)
        self.products["quality"] = _bayes_mean(stars["sum"], stars["count"], self.global_mean, self.prior_n)

        # Skin-type-specific smoothed rating, shrunk toward the product's overall quality.
        self.skin_quality = {}
        self.skin_counts = {}
        for st in SKIN_TYPES:
            sub = reviews[reviews["Skin_Type"] == st].groupby("Product_id")["Rating_Stars"].agg(["sum", "count"])
            sub = sub.reindex(ids).fillna(0)
            self.skin_quality[st] = _bayes_mean(sub["sum"], sub["count"], self.products["quality"], self.skin_prior_n)
            self.skin_counts[st] = sub["count"].astype(int)

        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda s: [t.strip().lower() for t in s.split(",") if t.strip()],
            token_pattern=None, lowercase=False,
        )
        self.ing_matrix = self.vectorizer.fit_transform(self.products["Ingredients"])
        self._pos = {pid: i for i, pid in enumerate(ids)}
        return self

    def recommend(self, profile: UserProfile, k: int = 10, explain: bool = True) -> pd.DataFrame:
        p, w = self.products, self.weights
        mask = np.ones(len(p), dtype=bool)
        if profile.category:
            mask &= (p["Category"] == profile.category).to_numpy()
        if profile.max_price is not None:
            mask &= (p["Price"] <= profile.max_price).to_numpy()
        for ing in profile.avoid_ingredients:
            mask &= ~p["Ingredients"].str.contains(ing, case=False, regex=False).to_numpy()
        if profile.exclude_product_ids:
            mask &= ~p.index.isin(profile.exclude_product_ids)
        if profile.sensitive:
            mask &= (p["Sensitive"] == 1).to_numpy()

        score = w["quality"] * _z(p["quality"])
        skin_gain = np.zeros(len(p))
        label_gain = np.zeros(len(p))
        taste = np.zeros(len(p))
        st = profile.skin_type if profile.skin_type in SKIN_TYPES else None
        if st:
            skin_gain = w["skin_fit"] * (_z(self.skin_quality[st]) - _z(p["quality"]))
            label_gain = w["label_fit"] * p[st].to_numpy()
        liked = [self._pos[i] for i in profile.liked_product_ids if i in self._pos]
        if liked:
            sim = (self.ing_matrix @ self.ing_matrix[liked].T).toarray().mean(axis=1)
            taste = w["taste"] * _z(sim)
        score = score + skin_gain + label_gain + taste

        out = p.assign(score=score)[mask].sort_values("score", ascending=False).head(k)
        if explain:
            out["why"] = [self._explain(pid, profile, st) for pid in out.index]
        return out.drop(columns=["Ingredients"])

    def _explain(self, pid, profile, st):
        row = self.products.loc[pid]
        reasons = [f"{row.quality:.2f}/5 smoothed rating from {row.n_reviews} reviews"]
        if st and self.skin_counts[st][pid] >= 3:
            reasons.append(f"{self.skin_counts[st][pid]} reviewers with {st.lower()} skin, "
                           f"{self.skin_quality[st][pid]:.2f}/5")
        if st and row[st] == 1:
            reasons.append(f"labelled suitable for {st.lower()} skin")
        if profile.liked_product_ids:
            reasons.append("similar ingredients to products you liked")
        return "; ".join(reasons)
