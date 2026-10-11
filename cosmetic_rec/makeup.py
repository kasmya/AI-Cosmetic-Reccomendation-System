"""Foundation shade matching: k-nearest-neighbours in CIELAB colour space.

Lab distance approximates perceived colour difference, so the nearest swatches to a measured
skin colour are the closest-looking shades. Data: 621 liquid-foundation shades from 36 brands.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .data import DATA_DIR
from .vision import rgb_hex_to_lab

SHADES_CSV = DATA_DIR / "foundation_shades.csv"


class ShadeMatcher:
    def __init__(self, path=SHADES_CSV):
        self.shades = pd.read_csv(path)
        self.shades["hex"] = "#" + self.shades["hex"].str.lstrip("#")
        self.lab = np.stack([rgb_hex_to_lab(h) for h in self.shades["hex"]])
        self.knn = NearestNeighbors(metric="euclidean").fit(self.lab)

    @property
    def brands(self) -> list[str]:
        return sorted(self.shades["brand"].unique())

    def match(self, lab, k: int = 5, brand: str | None = None) -> pd.DataFrame:
        """Closest shade per product, best first. delta_e < ~2 is hard to tell apart by eye."""
        dist, idx = self.knn.kneighbors(np.asarray(lab, dtype=np.float32).reshape(1, -1), n_neighbors=len(self.shades))
        out = self.shades.iloc[idx[0]].assign(delta_e=dist[0].round(2))
        if brand:
            out = out[out["brand"].str.lower() == brand.lower()]
        out = out.drop_duplicates(["brand", "product"]).head(k)
        return out.reset_index(drop=True)
