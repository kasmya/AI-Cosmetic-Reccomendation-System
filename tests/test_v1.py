import cv2
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app

skimage_data = pytest.importorskip("skimage.data")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_options_drive_the_form(client):
    o = client.get("/api/v1/options").json()
    assert "Dry" in o["skin_types"] and "acne" in o["concerns"] and "Cleanser" in o["skincare_categories"]
    assert len(o["makeup_brands"]) > 20


def test_assess_maps_concerns_and_validates(client):
    r = client.post("/api/v1/skin/assess", json={"skin_type": "Oily", "concerns": ["acne"]})
    assert r.status_code == 200 and "salicylic acid" in r.json()["concerns"][0]["helpful_ingredients"]
    assert "medical" in r.json()["disclaimer"].lower()
    assert client.post("/api/v1/skin/assess", json={"concerns": ["bogus"]}).status_code == 422


def test_consult_full_shape_without_photo(client):
    r = client.post("/api/v1/consult", json={"skin_type": "Dry", "concerns": ["dryness"],
                                             "colour_preferences": ["warm"], "per_category": 2})
    assert r.status_code == 200
    j = r.json()
    assert set(j["skincare"]) == {"Cleanser", "Treatment", "Moisturizer"}
    item = j["skincare"]["Cleanser"][0]
    assert 0 <= item["aicosmetic_score"] <= 100 and item["verdict"] in ("Recommended", "Okay", "Caution")
    assert j["makeup"]["foundation"]["status"] == "planned" and j["makeup"]["lips"]["status"] == "planned"
    assert [x["step"] for x in j["routine"]["am"]] == ["Cleanse", "Treat", "Moisturise", "Protect"]
    assert [x["step"] for x in j["routine"]["pm"]] == ["Cleanse", "Treat", "Moisturise"]
    assert "medical" in j["disclaimer"].lower()


def test_consult_respects_budget_and_avoid_list(client):
    j = client.post("/api/v1/consult", json={"skin_type": "Oily", "max_price_per_item": 25,
                                             "avoid_ingredients": ["fragrance"]}).json()
    assert all(i["price"] <= 25 for items in j["skincare"].values() for i in items)


def test_consult_with_skin_hex_returns_live_foundation(client):
    j = client.post("/api/v1/consult", json={"skin_hex": "#c68642"}).json()
    assert j["makeup"]["foundation"]["status"] == "live" and len(j["makeup"]["foundation"]["items"]) == 5


def test_consult_rejects_bad_input(client):
    assert client.post("/api/v1/consult", json={"colour_preferences": ["neon"]}).status_code == 422
    assert client.post("/api/v1/consult", json={"skin_hex": "nope"}).status_code == 422


def test_face_analyze_then_consult_flow(client):
    ok, png = cv2.imencode(".png", cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR))
    face = client.post("/api/v1/face/analyze", content=png.tobytes()).json()
    j = client.post("/api/v1/consult", json={"skin_type": "Normal", "skin_hex": face["skin_hex"],
                                             "skin_tone": face["tone"]}).json()
    assert j["makeup"]["foundation"]["status"] == "live"


def test_product_detail_and_review_summary_contract(client):
    r = client.get("/api/v1/products/0")
    pid = client.get("/skincare/products?limit=1").json()[0]["Product_id"]
    r = client.get(f"/api/v1/products/{pid}")
    assert r.status_code == 200 and r.json()["review_summary"]["status"] == "live"
    assert client.get(f"/api/v1/products/{pid}/reviews/summary").status_code == 200
    assert client.get("/api/v1/products/999999").status_code == 404
