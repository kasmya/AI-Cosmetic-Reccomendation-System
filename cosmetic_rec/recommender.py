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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from .categories import looks_like_makeup
from .concerns import helpful_hits, watch_hits
from .data import LABELS, SKIN_TYPES, build_products
from .vision import tone_index


def split_ingredients(text: str) -> list[str]:
    """Tokenizer for TF-IDF: one token per comma-separated ingredient (module-level so models can be pickled)."""
    return [t.strip().lower() for t in text.split(",") if t.strip()]


# concern: z-scored share of helpful ingredients present
# watch: flat penalty for ingredients often reported as aggravating. 2.5 because ~59% of products contain one for
# sensitivity (mostly fragrance); at 0.7 the top 10 was unchanged, at 2.5 it drops from 6/10 flagged to 1/10
DEFAULT_WEIGHTS = {"model": 1.0, "taste": 1.0, "concern": 0.8, "watch": 2.5}


@dataclass
class UserProfile:
    skin_type: str | None = None
    skin_tone: str | None = None          # e.g. from the photo analyser
    sensitive: bool = False
    concerns: list[str] = field(default_factory=list)
    category: str | None = None
    max_price: float | None = None
    avoid_ingredients: list[str] = field(default_factory=list)
    liked_product_ids: list[int] = field(default_factory=list)
    exclude_product_ids: list[int] = field(default_factory=list)
    include_makeup: bool = False          # products whose name says concealer / tinted / BB cream etc. are skipped


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)


class Recommender:
    DEFAULT_HGB = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200}

    def __init__(self, weights=None, svd_dim=20, seed=0, hgb_params=None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.svd_dim, self.seed = svd_dim, seed
        self.hgb_params = {**self.DEFAULT_HGB, **(hgb_params or {})}

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
        self.products["is_makeup"] = self.products["Product"].map(looks_like_makeup)
        self.vectorizer = TfidfVectorizer(
            tokenizer=split_ingredients, token_pattern=None, lowercase=False)
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
        # Platt-scaled with user-grouped folds, so the probabilities are fitted on users the model did not see.
        folds = list(GroupKFold(5).split(X, y, reviews["User_id"]))
        self.clf = CalibratedClassifierCV(
            HistGradientBoostingClassifier(**getattr(self, "hgb_params", self.DEFAULT_HGB), random_state=self.seed),
            method="sigmoid", cv=folds).fit(X, y)
        return self

    # ---------- inference ----------
    def predict_like(self, skin_type: str | None, skin_tone: str | None = None) -> pd.Series:
        ids = self.products.index.to_numpy()
        X = self._design(pd.Series([skin_type] * len(ids)), ids, pd.Series([skin_tone] * len(ids)))
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=ids)

    # ------------------------------------------------------------------------------------------------
    # Recommendation pipeline.  Stage 0 hard constraints -> stage 1 candidate retrieval (cheap score, optional
    # cut-off) -> stage 2 ranking (calibrated model + taste + concern fit - watch penalty) -> stage 3 re-ranking
    # (brand diversity).
    # Scores are normalised over the WHOLE catalog, so cutting candidates never changes a candidate's score.
    # At ~300 products the cut-off is for structure, not speed: its recall cost is measured in evaluate.py.
    # ------------------------------------------------------------------------------------------------
    def _constraints(self, profile: UserProfile) -> np.ndarray:
        """Stage 0: products that are allowed at all (category, price, avoid list, sensitivity, exclusions, makeup)."""
        p = self.products
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
        if not profile.include_makeup and "is_makeup" in p:      # artifacts saved earlier lack the column
            mask &= ~p["is_makeup"].to_numpy()
        return mask

    def _components(self, profile: UserProfile) -> dict[str, np.ndarray]:
        """Per-product signals over the whole catalog: model probability, ingredient similarity to liked products,
        concern fit and watch flags."""
        p = self.products
        p_like = self.predict_like(profile.skin_type, profile.skin_tone).to_numpy()
        sim = np.zeros(len(p))
        liked = [self._pos[i] for i in profile.liked_product_ids if i in self._pos]
        if liked:
            sim = (self.unit @ self.unit[liked].T).mean(axis=1)
        concern_fit, watch = np.zeros(len(p)), np.zeros(len(p))
        if profile.concerns:
            concern_fit = np.mean([[min(len(helpful_hits(c, t)), 2) / 2 for t in p["Ingredients"]]
                                   for c in profile.concerns], axis=0)
            watch = np.max([[float(bool(watch_hits(c, t))) for t in p["Ingredients"]]
                            for c in profile.concerns], axis=0)
        return {"p_like": p_like, "sim": sim, "concern_fit": concern_fit, "watch": watch,
                "site": _z(self.pfeat["site_rating"].to_numpy())}

    def _retrieve(self, profile, mask, c, limit: int | None, w) -> np.ndarray:
        """Stage 1: eligible positions, cut to `limit` by a cheap score that does not call the model."""
        pos = np.flatnonzero(mask)
        if limit is None or len(pos) <= limit:
            return pos
        cheap = c["site"] + w["taste"] * _z(c["sim"]) + w["concern"] * _z(c["concern_fit"]) - w["watch"] * c["watch"]
        return pos[np.argsort(-cheap[pos], kind="stable")[:limit]]

    def _rank(self, c, w) -> np.ndarray:
        """Stage 2: the full score."""
        score = w["model"] * _z(c["p_like"])
        if c["sim"].any():
            score = score + w["taste"] * _z(c["sim"])
        if c["concern_fit"].any() or c["watch"].any():
            score = score + w["concern"] * _z(c["concern_fit"]) - w["watch"] * c["watch"]
        return score

    @staticmethod
    def _rerank(ranked: pd.DataFrame, k: int, max_per_brand: int | None) -> pd.DataFrame:
        """Stage 3: keep the order but allow at most `max_per_brand` products per brand in the top k
        (skipped products fill any remaining places so k results are still returned)."""
        if max_per_brand is None:
            return ranked.head(k)
        kept, skipped, seen = [], [], {}
        for idx, brand in zip(ranked.index, ranked["Brand"], strict=True):
            if seen.get(brand, 0) < max_per_brand:
                kept.append(idx)
                seen[brand] = seen.get(brand, 0) + 1
            else:
                skipped.append(idx)
            if len(kept) == k:
                break
        return ranked.loc[(kept + skipped)[:k]]

    def recommend(self, profile: UserProfile, k: int = 10, explain: bool = True, candidate_limit: int | None = None,
                  max_per_brand: int | None = None) -> pd.DataFrame:
        # merge with defaults so artifacts saved before a weight existed keep working
        p, w = self.products, {**DEFAULT_WEIGHTS, **self.weights}
        mask = self._constraints(profile)                                   # stage 0
        c = self._components(profile)
        pos = self._retrieve(profile, mask, c, candidate_limit, w)          # stage 1
        score = self._rank(c, w)                                            # stage 2
        frame = p.assign(score=score, p_like=c["p_like"], similarity=c["sim"], concern_fit=c["concern_fit"])
        ranked = frame.iloc[pos].sort_values("score", ascending=False, kind="stable")
        out = self._rerank(ranked, k, max_per_brand)                        # stage 3
        trace = {"eligible": int(mask.sum()), "candidates": int(len(pos)), "returned": int(len(out))}
        if explain:
            out = out.assign(why=[self._explain(r, profile) for r in out.itertuples()])
        out = out.assign(ingredient_flags=[sorted({f for c_ in profile.concerns for f in watch_hits(c_, ing)})
                                           for ing in out["Ingredients"]])
        out = out.drop(columns=["Ingredients"])
        out.attrs["trace"] = trace
        return out

    def _explain(self, row, profile):
        who = f"{profile.skin_type.lower()} skin" if profile.skin_type else "a typical user"
        parts = [f"model estimates a {row.p_like:.0%} chance of a 4-5 star experience for {who}"]
        ing = self.products.at[row.Index, "Ingredients"]
        for c in profile.concerns:
            if found := helpful_hits(c, ing):
                parts.append(f"contains {', '.join(found)} (often used for {c.replace('_', ' ')})")
        if profile.skin_type and getattr(row, profile.skin_type) == 1:
            parts.append(f"labelled suitable for {profile.skin_type.lower()} skin")
        if profile.liked_product_ids:
            parts.append(f"ingredient similarity to your liked products: {row.similarity:.2f}")
        return "; ".join(parts)
