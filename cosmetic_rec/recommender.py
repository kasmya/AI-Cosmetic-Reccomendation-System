"""Hybrid recommender made of two small ML models.

1. Gradient boosting predicts P(user rates the product >= 4 stars) from the user's skin
   type and the product's category, brand, price, site rating, suitability labels and
   ingredient embedding.
2. TF-IDF + truncated SVD (LSA) embeds ingredient lists, so products similar to ones the
   user already liked can be found by cosine similarity.

The final score is a z-scored blend of the two. Hard filters are applied first.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import LABELS, SKIN_TYPES, build_products
from .vision import tone_index

DEFAULT_WEIGHTS = {"model": 1.0, "taste": 1.0}


@dataclass
class UserProfile:
    skin_type: str | None = None
    skin_tone: str | None = None          # e.g. from the photo analyser
    sensitive: bool = False
    category: str | None = None
    max_price: float | None = None
    avoid_ingredients: list[str] = field(default_factory=list)
    liked_product_ids: list[int] = field(default_factory=list)
    exclude_product_ids: list[int] = field(default_factory=list)


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)


class Recommender:
    def __init__(self, weights=None, svd_dim=20, seed=0):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.svd_dim, self.seed = svd_dim, seed

    # ---------- features ----------
    def _product_features(self, products: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(self.embed, index=products.index).add_prefix("ing")
        f["price"] = products["Price"]
        rated = products["Rating"] > 0
        f["site_rating"] = products["Rating"].where(rated, products.loc[rated, "Rating"].mean())
        for c in LABELS:
            f["label_" + c] = products[c]
        f = f.join(pd.get_dummies(products["Category"], prefix="cat").astype(int))
        f = f.join(pd.get_dummies(products["Brand"], prefix="brand").astype(int))
        return f

    def _user_features(self, skin_types: pd.Series) -> pd.DataFrame:
        u = pd.DataFrame({f"skin_{t}": (skin_types == t).astype(int) for t in SKIN_TYPES})
        u["skin_unknown"] = (~skin_types.isin(SKIN_TYPES)).astype(int)
        return u

    def _design(self, skin_types: pd.Series, product_ids, tones: pd.Series | None = None) -> pd.DataFrame:
        pf = self.pfeat.loc[product_ids].reset_index(drop=True)
        uf = self._user_features(skin_types.reset_index(drop=True))
        X = pd.concat([uf, pf], axis=1)
        X["skin_tone"] = float("nan") if tones is None else tones.map(tone_index).to_numpy()
        X["skin_match"] = sum(X[f"skin_{t}"] * X[f"label_{t}"] for t in SKIN_TYPES)
        return X.reindex(columns=self.columns, fill_value=0) if hasattr(self, "columns") else X

    # ---------- training ----------
    def fit(self, reviews: pd.DataFrame, products: pd.DataFrame | None = None):
        self.products = (products if products is not None else build_products(reviews)).copy()
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda s: [t.strip().lower() for t in s.split(",") if t.strip()],
            token_pattern=None, lowercase=False)
        tfidf = self.vectorizer.fit_transform(self.products["Ingredients"])
        svd = TruncatedSVD(self.svd_dim, random_state=self.seed)
        self.embed = svd.fit_transform(tfidf)
        norms = np.linalg.norm(self.embed, axis=1, keepdims=True)
        self.unit = self.embed / np.where(norms == 0, 1, norms)
        self.pfeat = self._product_features(self.products)
        self._pos = {pid: i for i, pid in enumerate(self.products.index)}

        X = self._design(reviews["Skin_Type"], reviews["Product_id"].to_numpy(), reviews["Skin_Tone"])
        self.columns = list(X.columns)
        y = (reviews["Rating_Stars"] >= 4).astype(int).to_numpy()
        self.clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=self.seed).fit(X, y)
        return self

    # ---------- inference ----------
    def predict_like(self, skin_type: str | None, skin_tone: str | None = None) -> pd.Series:
        ids = self.products.index.to_numpy()
        X = self._design(pd.Series([skin_type] * len(ids)), ids, pd.Series([skin_tone] * len(ids)))
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=ids)

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

        p_like = self.predict_like(profile.skin_type, profile.skin_tone)
        score = w["model"] * _z(p_like)
        sim = np.zeros(len(p))
        liked = [self._pos[i] for i in profile.liked_product_ids if i in self._pos]
        if liked:
            sim = (self.unit @ self.unit[liked].T).mean(axis=1)
            score = score + w["taste"] * _z(sim)

        out = p.assign(score=score, p_like=p_like.to_numpy(), similarity=sim)[mask]
        out = out.sort_values("score", ascending=False).head(k)
        if explain:
            out["why"] = [self._explain(r, profile) for r in out.itertuples()]
        return out.drop(columns=["Ingredients"])

    @staticmethod
    def _explain(row, profile):
        who = f"{profile.skin_type.lower()} skin" if profile.skin_type else "a typical user"
        parts = [f"model estimates a {row.p_like:.0%} chance of a 4-5 star experience for {who}"]
        if profile.skin_type and getattr(row, profile.skin_type) == 1:
            parts.append(f"labelled suitable for {profile.skin_type.lower()} skin")
        if profile.liked_product_ids:
            parts.append(f"ingredient similarity to your liked products: {row.similarity:.2f}")
        return "; ".join(parts)
