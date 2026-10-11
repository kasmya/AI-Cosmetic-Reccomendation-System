import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec import Recommender, UserProfile, load_reviews
from cosmetic_rec.api import app
from cosmetic_rec.catalog import build_catalog
from cosmetic_rec.categories import looks_like_makeup
from cosmetic_rec.routine import (
    AM_UNSAFE,
    active_classes,
    choose,
    find_clashes,
    find_swaps,
    is_mineral_only,
    pareto,
    pick_sunscreen,
    spf_of,
)
from cosmetic_rec.state import state


# ---------- ingredient classes ----------
def test_actives_count_only_near_the_top_of_the_list():
    assert active_classes("water | glycerin | retinol | niacinamide") == {"retinoid"}
    late = "water | " + " | ".join(f"x{i}" for i in range(20)) + " | lactic acid"
    assert active_classes(late) == set()             # a pH adjuster at the end is not an active
    assert active_classes("water | salicylic acid | glycolic acid") == {"bha", "aha"}


@pytest.mark.parametrize("name,spf", [("Sun Spray, SPF15", 15), ("Lait Enfant SPF 50+ La Roche", 50),
                                      ("Crème LSF 30", 30), ("30 SPF lotion", 30), ("sunscreen", None)])
def test_spf_is_parsed_from_names(name, spf):
    assert spf_of(name) == spf


def test_mineral_detection():
    assert is_mineral_only("water | zinc oxide | glycerin")
    assert not is_mineral_only("water | zinc oxide | avobenzone")
    assert not is_mineral_only("water | glycerin")


# ---------- clashes ----------
def item(name, ingredients, category="Treatment"):
    return {"name": name, "category": category, "ingredients": ingredients}


def test_retinoid_with_acid_is_a_caution_but_compatible_actives_are_not():
    r = item("Retinol serum", "water | retinol | glycerin")
    a = item("Glycolic toner", "water | glycolic acid | glycerin")
    n = item("Niacinamide serum", "water | niacinamide | glycerin")
    clashes = find_clashes("pm", [r, a])
    assert [c["severity"] for c in clashes] == ["caution"]
    assert set(clashes[0]["products"]) == {"Retinol serum", "Glycolic toner"}
    assert find_clashes("pm", [r, n]) == []


def test_rinse_off_cleanser_downgrades_the_clash():
    wash = item("BHA wash", "water | salicylic acid", "Cleanser")
    c = find_clashes("pm", [item("Retinol serum", "water | retinol"), wash])
    assert c and c[0]["severity"] == "info" and "rinse-off" in c[0]["message"]


def test_retinoid_in_the_morning_is_flagged_and_a_product_never_clashes_with_itself():
    r = item("Retinol serum", "water | retinol")
    assert any("night" in c["message"] for c in find_clashes("am", [r]))
    assert find_clashes("pm", [r, r]) == []


# ---------- optimiser ----------
def cand(pid, score, price):
    return {"pid": pid, "score": score, "price": price}


def test_pareto_drops_dominated_candidates():
    kept = pareto([cand(1, 90, 50), cand(2, 80, 60), cand(3, 85, 20), cand(4, 70, 10)])
    assert {c["pid"] for c in kept} == {1, 3, 4}            # 2 is worse and pricier than 3


def test_choose_maximises_score_within_budget():
    slots = {"a": [cand(1, 90, 50), cand(2, 80, 20)], "b": [cand(3, 90, 50), cand(4, 70, 10)]}
    assert choose(slots, None)[0]["a"]["pid"] == 1 and choose(slots, None)[2] == 100
    picks, within, total = choose(slots, 70)
    assert within and total <= 70 and {picks["a"]["pid"], picks["b"]["pid"]} == {2, 3}   # best total score under $70


def test_choose_reports_infeasible_budget_with_the_cheapest_option():
    slots = {"a": [cand(1, 90, 50), cand(2, 80, 20)], "b": [cand(3, 90, 50), cand(4, 70, 10)]}
    picks, within, total = choose(slots, 5)
    assert not within and total == 30 and {picks["a"]["pid"], picks["b"]["pid"]} == {2, 4}


def test_a_product_used_in_two_slots_is_paid_for_once():
    slots = {"am": [cand(7, 90, 40)], "pm": [cand(7, 90, 40)]}
    assert choose(slots, 40)[1:] == (True, 40)


# ---------- sunscreen ----------
@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def test_sunscreen_pick_respects_avoid_list_spf_minimum_and_adults(catalog):
    picks, _ = pick_sunscreen(catalog, ["fragrance"], False, [], k=10)
    assert picks and all(p["spf"] >= 15 and "fragrance" not in p["ingredients"] for p in picks)
    assert not any(w in p["name"].lower() for p in picks for w in ("enfant", "kids", "baby"))


def test_sensitive_skin_gets_mineral_only_sunscreen(catalog):
    picks, _ = pick_sunscreen(catalog, [], True, ["fragrance", "alcohol denat"], k=5)
    assert picks and all(p["mineral"] for p in picks)


def test_impossible_filters_return_nothing_not_an_error(catalog):
    # "|" separates ingredients, so every multi-ingredient list contains it: this avoids everything
    assert pick_sunscreen(catalog, ["|"], False, [])[0] == []


# ---------- swaps ----------
def similar_df(rows):
    cols = ["product_uid", "source_id", "name", "brand", "price", "similarity", "ingredients"]
    return pd.DataFrame(rows, columns=cols)


def test_swaps_need_saving_similarity_and_a_good_enough_score():
    it = {"product_uid": "s:1", "price": 40.0, "score": 80, "category": "Cleanser"}
    sim = similar_df([("s:2", "2", "Cheap close", "B", 20.0, 0.6, "water"),
                      ("s:3", "3", "Tiny saving", "B", 38.0, 0.9, "water"),
                      ("s:4", "4", "Too different", "B", 10.0, 0.2, "water"),
                      ("s:5", "5", "Much worse", "B", 15.0, 0.7, "water"),
                      ("s:6", "6", "Has avoided", "B", 15.0, 0.7, "water fragrance")])
    scores = {2: 79, 3: 80, 4: 80, 5: 60, 6: 80}
    out = find_swaps(it, sim, lambda pid: (scores[pid], "Okay"), set(), ["fragrance"], None)
    assert [s["name"] for s in out] == ["Cheap close"] and out[0]["saving"] == 20.0


# ---------- makeup kept out of skincare ----------
def test_makeup_looking_products_are_flagged_precisely():
    assert looks_like_makeup("Bye Bye Breakout Full-Coverage Concealer")
    assert looks_like_makeup("D-Bronzi Sunshine Drops")
    assert not looks_like_makeup("Clear Complexion Cleanser") and not looks_like_makeup("Skin Corrector Serum")


def test_recommender_skips_makeup_unless_asked():
    rec = Recommender().fit(load_reviews())
    flagged = set(rec.products.index[rec.products["is_makeup"]])
    assert flagged
    default = set(rec.recommend(UserProfile(), k=400, explain=False).index)
    assert not default & flagged
    assert flagged <= set(rec.recommend(UserProfile(include_makeup=True), k=400, explain=False).index)


# ---------- API ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def consult(client, **body):
    r = client.post("/api/v1/consult", json={"per_category": 1, **body})
    assert r.status_code == 200
    return r.json()


def ingredients_of(uid):
    df = state["catalog"].df
    return df[df["product_uid"] == uid].iloc[0]["ingredients"]


def test_routine_has_the_right_steps_and_a_sunscreen_only_in_the_morning(client):
    j = consult(client, skin_type="Oily")["routine"]
    assert [s["step"] for s in j["am"]] == ["Cleanse", "Treat", "Moisturise", "Protect"]
    assert [s["step"] for s in j["pm"]] == ["Cleanse", "Treat", "Moisturise"]
    protect = j["am"][3]
    assert protect["category"] == "Sunscreen" and protect["spf"] >= 15 and protect["price"] is None
    assert j["status"] == "live"


def test_morning_treatment_never_contains_a_retinoid_or_exfoliating_acid(client):
    for concerns in (["fine_lines"], ["acne"], ["dark_spots", "dullness"]):
        j = consult(client, skin_type="Normal", concerns=concerns)["routine"]
        step = j["am"][1]
        if step["product_id"] is not None:
            assert not active_classes(state["rec"].products.at[step["product_id"], "Ingredients"]) & AM_UNSAFE


@pytest.mark.parametrize("budget", [60, 90, 140])
def test_budget_is_respected_when_feasible(client, budget):
    s = consult(client, skin_type="Dry", total_budget=budget)["shopping_list"]
    if s["within_budget"]:
        assert s["total_price"] <= budget
    else:
        assert any("over your" in n for n in s["notes"])


def test_impossible_budget_is_reported_honestly(client):
    s = consult(client, skin_type="Dry", total_budget=5)["shopping_list"]
    assert s["within_budget"] is False and any("cheapest matching routine" in n for n in s["notes"])


def test_shopping_list_is_consistent_with_the_routine(client):
    j = consult(client, skin_type="Combination", total_budget=150)
    s, r = j["shopping_list"], j["routine"]
    uids = [p["product_uid"] for p in s["products"]]
    assert len(uids) == len(set(uids))
    used = {st["product_uid"] for st in r["am"] + r["pm"] if st.get("product_uid")}
    assert set(uids) == used
    assert s["total_price"] == round(sum(p["price"] for p in s["products"] if p["price"] is not None), 2)
    assert all(p["steps"] for p in s["products"])


def test_avoid_list_holds_for_every_product_including_the_sunscreen(client):
    j = consult(client, skin_type="Oily", avoid_ingredients=["fragrance", "alcohol denat"])
    for p in j["shopping_list"]["products"]:
        ing = (state["rec"].products.at[p["product_id"], "Ingredients"] if p["product_id"] is not None
               else ingredients_of(p["product_uid"]))
        assert "fragrance" not in ing and "alcohol denat" not in ing


def test_sensitive_user_gets_mineral_sunscreen(client):
    j = consult(client, skin_type="Dry", sensitive=True)["routine"]["am"][3]
    assert j["product_uid"] and is_mineral_only(ingredients_of(j["product_uid"]))


def test_no_makeup_in_the_routine(client):
    for st in ("Dry", "Oily"):
        for p in consult(client, skin_type=st, concerns=["acne"])["shopping_list"]["products"]:
            assert p["product_id"] is None or not looks_like_makeup(p["name"])


def test_routine_check_endpoint(client):
    df = state["catalog"].df
    treatments = df[(df["category"] == "Treatment") & df["has_ingredients"]]

    def with_active(name):
        return next(r.product_uid for r in treatments.itertuples() if name in active_classes(r.ingredients))

    retinol, acid = with_active("retinoid"), with_active("aha")
    ok = client.post("/api/v1/routine/check", json={"pm": [retinol, acid]}).json()
    assert any(c["severity"] == "caution" for c in ok["clashes"])
    assert client.post("/api/v1/routine/check", json={"pm": [retinol]}).json()["clashes"] == []
    assert client.post("/api/v1/routine/check", json={"pm": ["nope:1"]}).status_code == 404


def test_dupes_endpoint_shape_and_rules(client):
    pid = client.get("/skincare/products?limit=400").json()[5]["Product_id"]
    j = client.get(f"/api/v1/products/{pid}/dupes", params={"min_similarity": 0.1, "min_saving": 1}).json()
    assert j["product_id"] == pid and j["note"]
    price = state["rec"].products.at[pid, "Price"]
    assert all(s["price"] < price and s["saving"] >= 1 and s["similarity"] >= 0.1 for s in j["swaps"])
    assert client.get("/api/v1/products/999999/dupes").status_code == 404
