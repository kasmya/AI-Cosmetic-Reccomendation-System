"""Unified product catalog built from several sources, with provenance on every row.

Run: python -m cosmetic_rec.catalog     (prints a per-source report)

Columns: product_uid, source, source_id, name, brand, category, price, ingredients (' | ' joined, normalised),
has_ingredients, url, licence, n_reviews.
"""
import pandas as pd

from .data import DATA_DIR, build_products, load_reviews
from .ingredients import parse_ingredients

OBF_CSV = DATA_DIR / "obf_cosmetics.csv"
LICENCES = {
    "sephora_reviews": "dataset from the original project (see README)",
    "open_beauty_facts": "ODbL database, CC BY-SA 3.0 contents",
}
COLUMNS = ["product_uid", "source", "source_id", "name", "brand", "category", "price", "ingredients",
           "has_ingredients", "url", "licence", "n_reviews"]


def _sephora() -> pd.DataFrame:
    reviews = load_reviews()
    p = build_products(reviews).reset_index()
    counts = reviews.groupby("Product_id").size()
    urls = reviews.drop_duplicates("Product_id").set_index("Product_id")["Product_Url"]
    return pd.DataFrame({
        "source": "sephora_reviews", "source_id": p["Product_id"].astype(str), "name": p["Product"],
        "brand": p["Brand"], "category": p["Category"], "price": p["Price"].astype(float),
        "ingredients": p["Ingredients"], "url": p["Product_id"].map(urls), "n_reviews": p["Product_id"].map(counts)})


def _obf() -> pd.DataFrame:
    o = pd.read_csv(OBF_CSV, dtype=str)
    return pd.DataFrame({
        "source": "open_beauty_facts", "source_id": o["source_id"], "name": o["name"], "brand": o["brand"].fillna(""),
        "category": o["category"], "price": float("nan"), "ingredients": o["ingredients"].fillna(""),
        "url": o["url"], "n_reviews": 0})


def build_catalog() -> pd.DataFrame:
    df = pd.concat([_sephora(), _obf()], ignore_index=True)
    # Re-run every ingredient field through the shared cleaner so all sources use the same vocabulary.
    parsed = df["ingredients"].map(lambda t: parse_ingredients(t.replace(" | ", ", ") if isinstance(t, str) else t))
    df["ingredients"] = parsed.map(lambda xs: " | ".join(xs) if xs else "")
    df["has_ingredients"] = df["ingredients"] != ""
    df["product_uid"] = df["source"] + ":" + df["source_id"]
    df["licence"] = df["source"].map(LICENCES)
    return df[COLUMNS]


def report(df: pd.DataFrame) -> str:
    by_source = df.groupby("source").agg(products=("product_uid", "size"), with_ingredients=("has_ingredients", "sum"),
                                         with_price=("price", lambda s: int(s.notna().sum())))
    by_cat = df.pivot_table(index="category", columns="source", values="product_uid", aggfunc="size", fill_value=0)
    return f"{by_source.to_string()}\n\n{by_cat.to_string()}"


if __name__ == "__main__":
    print(report(build_catalog()))
