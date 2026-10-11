"""Search and ingredient-similarity over the unified catalog (all sources)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _split(text: str) -> list[str]:
    return [t for t in text.split(" | ") if t]


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame):
        self.df = catalog.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(tokenizer=_split, token_pattern=None, lowercase=False)
        self.matrix = self.vectorizer.fit_transform(self.df["ingredients"])   # L2-normalised rows
        self._row = {uid: i for i, uid in enumerate(self.df["product_uid"])}

    def stats(self) -> dict:
        df = self.df
        return {
            "total": len(df),
            "sources": df.groupby("source").agg(products=("product_uid", "size"),
                                                with_ingredients=("has_ingredients", "sum"),
                                                with_price=("price", lambda s: int(s.notna().sum())),
                                                licence=("licence", "first")).reset_index().to_dict("records"),
            "categories": df["category"].value_counts().to_dict(),
        }

    def search(self, q: str = "", category: str | None = None, source: str | None = None,
               limit: int = 20) -> pd.DataFrame:
        m = np.ones(len(self.df), dtype=bool)
        if q:
            hay = (self.df["name"].fillna("") + " " + self.df["brand"].fillna("")).str.lower()
            m &= hay.str.contains(q.lower(), regex=False).to_numpy()
        if category:
            m &= (self.df["category"] == category).to_numpy()
        if source:
            m &= (self.df["source"] == source).to_numpy()
        return self.df[m].head(limit)

    def similar(self, product_uid: str, k: int = 3, same_category: bool = True,
                other_source: bool = True, source: str | None = None) -> pd.DataFrame:
        """Products with the most similar ingredient lists (cosine). Empty if the product has no ingredients."""
        i = self._row.get(product_uid)
        if i is None or not self.df.at[i, "has_ingredients"]:
            return self.df.iloc[0:0].assign(similarity=[])
        sims = (self.matrix @ self.matrix[i].T).toarray().ravel()
        ok = self.df["has_ingredients"].to_numpy().copy()
        ok[i] = False
        if same_category:
            ok &= (self.df["category"] == self.df.at[i, "category"]).to_numpy()
        if other_source:
            ok &= (self.df["source"] != self.df.at[i, "source"]).to_numpy()
        if source:
            ok &= (self.df["source"] == source).to_numpy()
        idx = np.where(ok)[0]
        top = idx[np.argsort(-sims[idx])[:k]]
        return self.df.iloc[top].assign(similarity=sims[top].round(3))
