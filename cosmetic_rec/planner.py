"""Assembles the AM/PM routine and the shopping list from scored candidates (pure logic, no web framework).

Inputs are plain dicts so this module does not depend on the API schemas:
  candidate  {product_id, name, brand, category, price, score, verdict, source}
Output is a dict with `am`, `pm`, `clashes`, `notes` and `shopping` (see plan()).
"""
from dataclasses import dataclass, field

from . import concerns as cx
from . import routine as rt


@dataclass
class PlanInput:
    skin_type: str | None = None
    concerns: list[str] = field(default_factory=list)
    sensitive: bool = False
    avoid: list[str] = field(default_factory=list)
    budget: float | None = None
    max_price: float | None = None


def _step(label: str, cand: dict | None, note: str = "") -> dict:
    if cand is None:
        return {"step": label, "product_id": None, "name": None, "note": note}
    return {"step": label, "product_id": cand["product_id"], "name": cand["name"], "category": cand["category"],
            "product_uid": f"sephora_reviews:{cand['product_id']}", "price": cand["price"], "source": cand["source"],
            "note": note}


def _slot(cands: list[dict]) -> list[dict]:
    pool = [{"pid": c["product_id"], "score": c["score"], "price": c["price"], "cand": c} for c in cands]
    return rt.pareto(pool)


def _sunscreen_step(catalog_df, inp: PlanInput) -> tuple[dict, dict | None, list[str]]:
    needs_mineral = inp.sensitive or bool({"sensitivity", "redness"} & set(inp.concerns))
    picks, notes = rt.pick_sunscreen(catalog_df, inp.avoid, needs_mineral, cx.watch_substrings(inp.concerns))
    if not picks:
        return ({"step": "Protect", "product_id": None, "name": None, "category": "Sunscreen",
                 "note": "No sunscreen in the catalog matched your filters."}, None, notes)
    top = picks[0]
    note = "Unscored: no prices or reviews exist for this source."
    note += " Mineral filters." if top["mineral"] else ""
    return ({"step": "Protect", "product_id": None, "name": top["name"], "category": "Sunscreen",
             "product_uid": top["product_uid"], "source": top["source"], "spf": top["spf"], "note": note}, top, notes)


def _clashes(rec, steps_by_slot: dict[str, list[dict]], sunscreen: dict | None) -> list[dict]:
    out = []
    for slot, steps in steps_by_slot.items():
        items = []
        for st in steps:
            if st.get("product_id") is not None:
                items.append({"name": st["name"], "category": st["category"],
                              "ingredients": rec.products.at[st["product_id"], "Ingredients"]})
            elif sunscreen and st.get("product_uid") == sunscreen["product_uid"]:
                items.append({"name": sunscreen["name"], "category": "Sunscreen",
                              "ingredients": sunscreen["ingredients"]})
        out += rt.find_clashes(slot, items)
    return out


def _shopping_products(catalog_df, am: list[dict], pm: list[dict], by_pid: dict[int, dict]) -> list[dict]:
    """Each distinct product once, with the steps that use it."""
    used: dict[str, dict] = {}
    for when, steps in (("AM", am), ("PM", pm)):
        for st in steps:
            uid = st.get("product_uid")
            if uid:
                entry = used.setdefault(uid, {"step": st, "steps": []})
                entry["steps"].append(f"{when} {st['step'].lower()}")
    products = []
    for uid, entry in used.items():
        st = entry["step"]
        row = catalog_df[catalog_df["product_uid"] == uid].iloc[0]
        cand = by_pid.get(st.get("product_id"))
        products.append({
            "product_uid": uid, "product_id": st.get("product_id"), "name": st["name"], "brand": row["brand"] or "",
            "category": st["category"], "price": st.get("price"), "source": st.get("source"), "steps": entry["steps"],
            "aicosmetic_score": cand["score"] if cand else None, "verdict": cand["verdict"] if cand else None,
            "url": row["url"] if isinstance(row["url"], str) else None})
    return products


def _swaps(catalog, products: list[dict], inp: PlanInput, scorer) -> list[dict]:
    taken = {p["product_uid"] for p in products}
    out = []
    for p in products:
        if p["product_id"] is None or p["price"] is None:
            continue
        similar = catalog.similar(p["product_uid"], k=15, same_category=True, other_source=False,
                                  source="sephora_reviews")
        item = {"product_uid": p["product_uid"], "price": p["price"], "score": p["aicosmetic_score"],
                "category": p["category"]}
        out += rt.find_swaps(item, similar, scorer, taken, inp.avoid, inp.max_price)
    return out


def plan(inp: PlanInput, cands: dict[str, list[dict]], rec, catalog, scorer) -> dict:
    """cands: category -> scored candidates, best first. scorer(product_id) -> (score, verdict)."""
    ingredients = lambda c: rec.products.at[c["product_id"], "Ingredients"]  # noqa: E731
    morning_safe = [c for c in cands["Treatment"] if not rt.active_classes(ingredients(c)) & rt.AM_UNSAFE]
    slots = {"cleanser": _slot(cands["Cleanser"]), "treat_am": _slot(morning_safe),
             "treat_pm": _slot(cands["Treatment"]), "moisturizer": _slot(cands["Moisturizer"])}
    picks, within, total = rt.choose(slots, inp.budget)
    pick = {k: (v["cand"] if v else None) for k, v in picks.items()}

    sun_step, sunscreen, notes = _sunscreen_step(catalog.df, inp)
    pm_treat = pick["treat_pm"]
    actives = " Contains active ingredients: start 2-3 nights a week and build up." if (
        pm_treat and rt.active_classes(ingredients(pm_treat))) else ""
    no_am = "" if pick["treat_am"] else ("No morning-safe treatment matched "
                                          "(retinoids and exfoliating acids stay for night).")
    am = [_step("Cleanse", pick["cleanser"]), _step("Treat", pick["treat_am"], no_am),
          _step("Moisturise", pick["moisturizer"]), sun_step]
    pm = [_step("Cleanse", pick["cleanser"], "Double cleanse if you wore makeup or sunscreen."),
          _step("Treat", pm_treat, actives.strip()), _step("Moisturise", pick["moisturizer"])]
    clashes = _clashes(rec, {"am": am, "pm": pm}, sunscreen)

    by_pid = {c["product_id"]: c for v in cands.values() for c in v}
    products = _shopping_products(catalog.df, am, pm, by_pid)
    unpriced = sum(p["price"] is None for p in products)
    if unpriced:
        notes.append(f"{unpriced} product(s) have no price in their data source and are not in the total.")
    if inp.budget is not None and not within:
        notes.append(f"The cheapest matching routine costs ${total:.2f}, ${total - inp.budget:.2f} over your "
                     f"${inp.budget:.2f} budget.")
    swaps = _swaps(catalog, products, inp, scorer)
    if not swaps and any(p["product_id"] is not None for p in products):
        notes.append("No cheaper product with a similar ingredient list was found; close ingredient matches are "
                     "rare in this catalog.")
    return {"am": am, "pm": pm, "clashes": clashes, "notes": notes, "shopping": {
        "status": "live", "items": [p["product_id"] for p in products if p["product_id"] is not None],
        "total_price": round(sum(p["price"] for p in products if p["price"] is not None), 2),
        "budget": inp.budget, "within_budget": None if inp.budget is None else within,
        "products": products, "swaps": swaps, "notes": notes}}
