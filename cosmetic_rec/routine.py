"""Routine builder: AM/PM slots, ingredient-clash rules, a budget-aware picker, sunscreen choice and cheaper swaps.

Everything here is hand-written cosmetic-routine convention (the kind found in dermatology-adjacent beauty guidance),
not clinical advice and not validated against outcomes. Clash rules are phrased as cautions, because evidence on
many combinations is mixed, and concentrations are unknown: an ingredient only counts as an active when it appears
near the top of the list (ingredients are listed in descending amount).
"""
import itertools
import re

import pandas as pd

# class -> (regex over a normalised ingredient name, how far down the list it still counts as an active)
ACTIVE_CLASSES: dict[str, tuple[re.Pattern, int]] = {
    "retinoid": (re.compile(r"\bretinol\b|\bretinal\b|retinaldehyde|retinoate|retinoic"), 25),
    "aha": (re.compile(r"glycolic acid|lactic acid|mandelic acid"), 15),
    "bha": (re.compile(r"salicylic acid"), 15),
    "benzoyl_peroxide": (re.compile(r"benzoyl peroxide"), 15),
    "vitamin_c": (re.compile(r"\bascorbic acid\b"), 15),
}
AM_UNSAFE = {"retinoid", "aha", "bha", "benzoyl_peroxide"}      # kept out of the morning treatment slot
MINERAL = re.compile(r"zinc oxide|titanium dioxide")
CHEMICAL_FILTERS = re.compile(r"avobenzone|butyl methoxydibenzoylmethane|octocrylene|oxybenzone|benzophenone|"
                              r"homosalate|octisalate|ethylhexyl salicylate|octinoxate|ethylhexyl methoxycinnamate")
CHILD_PRODUCT = re.compile(r"enfant|\bkids?\b|child|baby|b[eé]b[eé]|kinder|pediatric|ni[nñ]os", re.I)
_SPF = re.compile(r"(?:spf|fps|lsf|ip)\s*[:\-]?\s*(\d{2})|(\d{2})\s*\+?\s*(?:spf|fps|lsf)", re.I)

CLASH_RULES = [
    ({"retinoid", "aha"}, "caution",
     "A retinoid and an exfoliating acid (AHA) in the same routine raise irritation risk; "
     "many guides suggest using them on different nights."),
    ({"retinoid", "bha"}, "caution",
     "A retinoid and salicylic acid in the same routine raise irritation risk; "
     "many guides suggest alternating nights."),
    ({"retinoid", "benzoyl_peroxide"}, "caution",
     "Benzoyl peroxide can break down some retinoids and both can irritate; they are usually used at different times."),
    ({"aha", "bha"}, "info",
     "Layering several exfoliating acids can over-exfoliate; one at a time is the usual advice."),
    ({"vitamin_c", "aha"}, "info",
     "Vitamin C and exfoliating acids are both low-pH and can sting together; "
     "separating them (morning and evening) is common."),
    ({"vitamin_c", "bha"}, "info", "Vitamin C and salicylic acid can sting together; separating them is common."),
    ({"vitamin_c", "benzoyl_peroxide"}, "info", "Benzoyl peroxide can oxidise vitamin C, reducing its effect."),
]
RINSE_OFF_NOTE = " (one is a rinse-off cleanser, so the risk is lower)"
RETINOID_AM_NOTE = "Contains a retinoid: these are usually used at night and make daily sunscreen important."


def tokens(ingredients: str) -> list[str]:
    return [t.strip() for t in re.split(r"\s\|\s|,", ingredients or "") if t.strip()]


def active_classes(ingredients: str) -> set[str]:
    """Active classes whose ingredient appears within the position limit (a late mention is likely a pH adjuster)."""
    found = set()
    for pos, tok in enumerate(tokens(ingredients), start=1):
        for cls, (rx, limit) in ACTIVE_CLASSES.items():
            if pos <= limit and rx.search(tok):
                found.add(cls)
    return found


def spf_of(name: str) -> int | None:
    m = _SPF.search(name or "")
    return int(next(g for g in m.groups() if g)) if m else None


def is_mineral_only(ingredients: str) -> bool:
    return bool(MINERAL.search(ingredients)) and not CHEMICAL_FILTERS.search(ingredients)


# ---------------- clash detection ----------------
def find_clashes(slot: str, items: list[dict]) -> list[dict]:
    """Clashes among the products used in one slot. items: {name, category, ingredients}.
    Rinse-off cleansers are downgraded to 'info' (low contact time)."""
    out = []
    for a, b in itertools.combinations(items, 2):
        if a["name"] == b["name"]:
            continue
        ca, cb = active_classes(a["ingredients"]), active_classes(b["ingredients"])
        for pair, severity, message in CLASH_RULES:
            x, y = tuple(pair) if len(pair) == 2 else (None, None)
            hit = (x in ca and y in cb) or (y in ca and x in cb)
            if hit:
                rinse = "Cleanser" in (a["category"], b["category"])
                out.append({"severity": "info" if rinse else severity, "slot": slot,
                            "products": [a["name"], b["name"]], "message": message + (RINSE_OFF_NOTE if rinse else "")})
    for it in items:
        if slot == "am" and active_classes(it["ingredients"]) & {"retinoid"}:
            out.append({"severity": "caution", "slot": "am", "products": [it["name"]],
                        "message": RETINOID_AM_NOTE})
        elif slot == "am" and it["category"] != "Cleanser" and active_classes(it["ingredients"]) & {"aha", "bha"}:
            out.append({"severity": "info", "slot": "am", "products": [it["name"]],
                        "message": "Exfoliating acids can increase sun sensitivity: use sunscreen."})
    return out


# ---------------- budget-aware choice ----------------
def pareto(cands: list[dict], limit: int = 8) -> list[dict]:
    """Keep candidates that no other candidate beats on both score (higher) and price (lower), best score first.
    This lets a budget search look at a large pool but only combine the few that can matter."""
    kept = [c for c in cands if not any(o is not c and o["score"] >= c["score"] and o["price"] <= c["price"]
                                        and (o["score"] > c["score"] or o["price"] < c["price"]) for o in cands)]
    return sorted(kept, key=lambda c: (-c["score"], c["price"]))[:limit]



def choose(candidates: dict[str, list[dict]], budget: float | None) -> tuple[dict[str, dict | None], bool, float]:
    """Pick one candidate per slot maximising total score within `budget` (counting each distinct product once).
    candidates: slot -> [{pid, score, price, ...}] (best first). Returns (picks, within_budget, total_price).
    Without a feasible combination the cheapest one is returned with within_budget False."""
    slots = [s for s, c in candidates.items() if c]
    if not slots:
        return {s: None for s in candidates}, True, 0.0
    best = cheapest = None
    for combo in itertools.product(*(candidates[s] for s in slots)):
        total = sum({c["pid"]: c["price"] for c in combo}.values())
        value = sum(c["score"] for c in combo)
        if cheapest is None or total < cheapest[0]:
            cheapest = (total, combo)
        if (budget is None or total <= budget) and (best is None or (value, -total) > (best[0], -best[1])):
            best = (value, total, combo)
    picks = {s: None for s in candidates}
    if best:
        picks.update(dict(zip(slots, best[2], strict=True)))
        return picks, True, round(best[1], 2)
    picks.update(dict(zip(slots, cheapest[1], strict=True)))
    return picks, False, round(cheapest[0], 2)


# ---------------- sunscreen ----------------
def pick_sunscreen(catalog: pd.DataFrame, avoid: list[str], needs_mineral: bool, watch: list[str],
                   k: int = 3) -> tuple[list[dict], list[str]]:
    """Unscored sunscreen shortlist from the catalog (no prices or reviews exist for these products).
    Returns (picks best first, notes explaining any relaxed rule)."""
    pool = catalog[(catalog["category"] == "Sunscreen") & catalog["has_ingredients"]].copy()
    pool["spf"] = pool["name"].map(spf_of)
    pool = pool[(pool["spf"].fillna(0) >= 15) & ~pool["name"].str.contains(CHILD_PRODUCT)].copy()
    for a in avoid:
        pool = pool[~pool["ingredients"].str.contains(a, case=False, regex=False)]
    notes = []
    if needs_mineral:
        mineral = pool[pool["ingredients"].map(is_mineral_only)]
        if len(mineral):
            pool = mineral
        else:
            notes.append("No mineral-only sunscreen matched your other filters, so chemical filters are allowed.")
    if watch:
        clean = pool[~pool["ingredients"].map(lambda t: any(w in t for w in watch))]
        if len(clean):
            pool = clean
        else:
            notes.append("Every matching sunscreen contains an ingredient often reported as irritating "
                         "for your concerns.")
    pool = pool.assign(rank=((pool["spf"] >= 30).astype(int) * 2 + (pool["spf"] >= 50).astype(int)
                             + pool["name"].str.contains(r"face|visage|gesicht|rostro", case=False).astype(int)))
    pool = pool.sort_values(["rank", "name"], ascending=[False, True]).head(k)
    return [{"product_uid": r.product_uid, "name": r.name, "brand": r.brand or "", "source": r.source,
             "spf": int(r.spf), "url": r.url if isinstance(r.url, str) else None, "ingredients": r.ingredients,
             "mineral": is_mineral_only(r.ingredients)} for r in pool.itertuples()], notes


# ---------------- cheaper swaps ----------------
MIN_SAVING, MIN_SIMILARITY, MAX_SCORE_DROP = 3.0, 0.35, 5


def find_swaps(item: dict, similar: pd.DataFrame, scorer, taken: set[str], avoid: list[str], max_price: float | None,
               k: int = 2) -> list[dict]:
    """Cheaper products of the same category with a similar ingredient list that score nearly as well.
    item: {product_uid, price, score, category}; similar: CatalogIndex.similar() output (same source, same category);
    scorer(product_id) -> (score, verdict)."""
    out = []
    for r in similar.itertuples():
        if r.product_uid in taken or pd.isna(r.price) or r.similarity < MIN_SIMILARITY:
            continue
        saving = item["price"] - float(r.price)
        if saving < MIN_SAVING or (max_price is not None and r.price > max_price):
            continue
        if any(a.lower() in r.ingredients for a in avoid):
            continue
        pid = int(r.source_id)
        score, verdict = scorer(pid)
        if score < item["score"] - MAX_SCORE_DROP:
            continue
        out.append({"for_product_uid": item["product_uid"], "product_id": pid, "product_uid": r.product_uid,
                    "name": r.name, "brand": r.brand, "price": float(r.price), "saving": round(saving, 2),
                    "similarity": float(r.similarity), "aicosmetic_score": score, "verdict": verdict,
                    "note": "Similar ingredient list, not a guarantee of the same results."})
    out.sort(key=lambda s: (-s["similarity"], -s["saving"]))
    return out[:k]
