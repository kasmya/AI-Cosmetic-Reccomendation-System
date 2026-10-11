"""Ingredient-list cleaning shared by every data source.

Different sources write the same ingredient differently ("Aqua/Water" vs "Water", "Parfum" vs "Fragrance")
and some "ingredient" fields are not lists at all ("Visit the brand boutique"). parse_ingredients() returns a
list of normalised names, or None when the field is not a usable ingredient list.
"""
import re

# INCI / marketing synonyms -> one canonical name. Extend as new sources are added.
SYNONYMS = {
    "aqua": "water", "eau": "water", "aqua/water": "water", "water/aqua": "water", "aqua/eau": "water",
    "parfum": "fragrance", "aroma": "fragrance", "perfume": "fragrance", "parfum/fragrance": "fragrance",
    "fragrance/parfum": "fragrance", "aroma/flavor": "fragrance",
    "vitamin c": "ascorbic acid", "vitamin e": "tocopherol", "vitamin b5": "panthenol",
    "alcohol denat.": "alcohol denat", "sd alcohol 40": "alcohol denat", "ethanol": "alcohol denat",
    "hyaluronic acid": "hyaluronic acid", "sodium hyaluronate": "hyaluronic acid",
    "tocopheryl acetate": "tocopherol", "retinyl palmitate": "retinol ester",
}
_JUNK = re.compile(r"https?://|www\.|visit |boutique|click |see (our|the) |ingredients? (not|are not)", re.I)
_PAREN = re.compile(r"\([^)]*\)")
_NOISE = re.compile(r"[*†‡•]|\d+(\.\d+)?\s*%")


def normalize_ingredient(token: str) -> str:
    t = _PAREN.sub("", token)
    t = _NOISE.sub("", t).lower().strip(" .;:-_/\\\"'")
    t = re.sub(r"\s+", " ", t)
    if "/" in t and t not in SYNONYMS:      # "aqua/water" style double names: keep the first known part
        first = t.split("/")[0].strip()
        t = SYNONYMS.get(first, first)
    return SYNONYMS.get(t, t)


def parse_ingredients(text) -> list[str] | None:
    """Normalised ingredient names, or None if the text is empty or not an ingredient list."""
    if not isinstance(text, str) or _JUNK.search(text):
        return None
    text = re.sub(r"(?i)^\s*(ingredients?|inci)\s*[:\-]\s*", "", text)
    tokens = [normalize_ingredient(t) for t in re.split(r"[,;]|\.\s+(?=[A-Za-z])", text)]
    tokens = [t for t in tokens if t]
    if not 3 <= len(tokens) <= 150:
        return None
    if sum(len(t) > 60 for t in tokens) > 0.2 * len(tokens):       # sentences, not names
        return None
    if sum(bool(re.search(r"[a-z]", t)) for t in tokens) < 0.8 * len(tokens):
        return None
    # de-duplicate, keep order
    return list(dict.fromkeys(tokens))
