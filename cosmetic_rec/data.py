"""Load the review CSV and collapse it into a one-row-per-product table."""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "skindataall.csv"
SKIN_TYPES = ["Combination", "Dry", "Normal", "Oily"]
# Product-level "suitable for" flags shipped with the dataset (constant per product).
SUITABILITY_FLAGS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]

REVIEW_COLS = ["User_id", "Product_id", "Rating_Stars", "Skin_Type", "Skin_Tone"]


def load_reviews(path=DEFAULT_CSV) -> pd.DataFrame:
    """Reviews with 'No data' converted to real missing values."""
    df = pd.read_csv(path)
    for col in ["Skin_Type", "Skin_Tone", "Eye_Color", "Hair_Color"]:
        df[col] = df[col].replace("No data", np.nan)
    return df


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product with static attributes (no review-derived stats)."""
    cols = ["Product_id", "Product", "Brand", "Category", "Price", "Ingredients"] + SUITABILITY_FLAGS
    products = df.drop_duplicates("Product_id")[cols].set_index("Product_id").sort_index()
    products["Ingredients"] = products["Ingredients"].fillna("")
    return products
