"""Load the review CSV and collapse it into a one-row-per-product table."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "skindataall.csv"
SKIN_TYPES = ["Combination", "Dry", "Normal", "Oily"]
LABELS = SKIN_TYPES + ["Sensitive"]  # product-level "suitable for" flags in the dataset


def load_reviews(path=DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.replace("No data", np.nan)


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Product_id", "Product", "Brand", "Category", "Price", "Rating", "Ingredients"] + LABELS
    p = df.drop_duplicates("Product_id")[cols].set_index("Product_id").sort_index()
    p["Ingredients"] = p["Ingredients"].fillna("")
    return p
