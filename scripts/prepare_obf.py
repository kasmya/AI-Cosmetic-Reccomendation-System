"""One-off: trim the Open Beauty Facts CSV export to the face skincare/makeup rows we use.

  curl -O https://static.openbeautyfacts.org/data/en.openbeautyfacts.org.products.csv.gz && gunzip it
  python scripts/prepare_obf.py path/to/en.openbeautyfacts.org.products.csv

Writes data/obf_cosmetics.csv. Source: Open Beauty Facts (https://world.openbeautyfacts.org), database under ODbL,
contents under CC BY-SA 3.0; attribution is in the README.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cosmetic_rec.categories import map_category  # noqa: E402
from cosmetic_rec.ingredients import parse_ingredients  # noqa: E402

COLS = ["code", "url", "product_name", "brands", "categories_en", "ingredients_text", "countries_en"]
OUT = Path(__file__).resolve().parent.parent / "data" / "obf_cosmetics.csv"


def main(src: str) -> None:
    raw = pd.read_csv(src, sep="\t", usecols=COLS, low_memory=False, on_bad_lines="skip", dtype=str)
    n_raw = len(raw)
    raw = raw.dropna(subset=["product_name", "ingredients_text"])
    raw["category"] = [map_category(f"{c} {n}") for c, n in zip(raw["categories_en"], raw["product_name"], strict=True)]
    raw = raw.dropna(subset=["category"])
    parsed = raw["ingredients_text"].map(parse_ingredients)
    raw = raw[parsed.notna()].assign(ingredients=parsed[parsed.notna()].map(" | ".join))
    raw["brands"] = raw["brands"].fillna("").str.split(",").str[0].str.strip()
    names = {"code": "source_id", "product_name": "name", "brands": "brand", "countries_en": "countries"}
    keep = ["source_id", "name", "brand", "category", "ingredients", "url", "countries"]
    out = raw.rename(columns=names)[keep].drop_duplicates("source_id")
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{n_raw} raw rows -> {len(out)} kept\n{out['category'].value_counts().to_string()}")


if __name__ == "__main__":
    main(sys.argv[1])
