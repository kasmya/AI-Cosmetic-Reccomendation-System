import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app
from cosmetic_rec.catalog import COLUMNS, build_catalog
from cosmetic_rec.catalog_index import CatalogIndex
from cosmetic_rec.categories import ALL_CATEGORIES, map_category
from cosmetic_rec.ingredients import normalize_ingredient, parse_ingredients


def test_synonyms_and_noise_are_normalised():
    assert normalize_ingredient("AQUA/WATER") == "water"
    assert normalize_ingredient("Parfum") == "fragrance"
    assert normalize_ingredient("Niacinamide (2%)") == "niacinamide"
    assert parse_ingredients("Aqua, Parfum, Glycerin*") == ["water", "fragrance", "glycerin"]


@pytest.mark.parametrize("junk", ["Visit the Jack Black boutique", "", None, "water, glycerin", "http://x.com a, b, c"])
def test_non_lists_are_rejected(junk):
    assert parse_ingredients(junk) is None


def test_duplicates_removed_in_order():
    assert parse_ingredients("Water, Glycerin, Water, Niacinamide") == ["water", "glycerin", "niacinamide"]


def test_category_mapping_includes_and_excludes():
    assert map_category("Face Wash, Cleansers") == "Cleanser"
    assert map_category("Sun protection, Sunscreens") == "Sunscreen"
    assert map_category("Hair, Shampoos") is None
    assert map_category("Moisturizing Lip Balm, Lip care") is None
    assert map_category("Lipsticks") == "Lipstick"


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def test_catalog_schema_provenance_and_uniqueness(catalog):
    assert list(catalog.columns) == COLUMNS
    assert catalog["product_uid"].is_unique
    assert set(catalog["source"]) == {"sephora_reviews", "open_beauty_facts"}
    assert catalog["category"].isin(ALL_CATEGORIES).all()
    assert catalog["licence"].notna().all()
    assert (catalog.loc[~catalog["has_ingredients"], "ingredients"] == "").all()


def test_similar_returns_other_source_same_category(catalog):
    idx = CatalogIndex(catalog)
    seph = catalog[(catalog["source"] == "sephora_reviews") & catalog["has_ingredients"]].iloc[0]
    sim = idx.similar(seph["product_uid"], k=3)
    assert len(sim) == 3 and (sim["source"] != "sephora_reviews").all() and (sim["category"] == seph["category"]).all()
    assert sim["similarity"].is_monotonic_decreasing
    assert idx.similar("nope:1").empty


def test_catalog_endpoints_and_alternatives_in_consult():
    with TestClient(app) as c:
        stats = c.get("/api/v1/catalog/stats").json()
        assert stats["total"] > 1900
        assert {s["source"] for s in stats["sources"]} == {"sephora_reviews", "open_beauty_facts"}
        hits = c.get("/api/v1/catalog/search", params={"q": "sun", "source": "open_beauty_facts", "limit": 5}).json()
        assert 0 < len(hits) <= 5 and all(h["source"] == "open_beauty_facts" for h in hits)
        j = c.post("/api/v1/consult", json={"skin_type": "Dry", "per_category": 1}).json()
        alts = [a for items in j["skincare"].values() for i in items for a in i["alternatives"]]
        assert alts and all(a["source"] == "open_beauty_facts" and a["price"] is None for a in alts)
