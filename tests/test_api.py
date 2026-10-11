import cv2
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app

skimage_data = pytest.importorskip("skimage.data")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def face_png():
    ok, png = cv2.imencode(".png", cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR))
    return png.tobytes()


def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert 0.5 < info["cv_auc"] < 1 and info["n_reviews"] > 8000
    assert "X-Process-Time-ms" in client.get("/health").headers


def test_skincare_recommend(client):
    r = client.post("/skincare/recommend", json={"skin_type": "Oily", "category": "Moisturizer", "k": 3})
    assert r.status_code == 200 and r.json()["count"] == 3
    assert client.post("/skincare/recommend", json={"skin_type": "Purple"}).status_code == 422
    assert client.post("/skincare/recommend", json={"skin_tone": "Purple"}).status_code == 422
    assert client.get("/skincare/products?q=cleanser&limit=5").status_code == 200


def test_face_analyze(client, face_png):
    r = client.post("/face/analyze", content=face_png)
    assert r.status_code == 200 and {"tone", "skin_hex", "lab"} <= r.json().keys()
    assert client.post("/face/analyze", content=b"not an image").status_code == 400
    assert client.post("/face/analyze", content=b"").status_code == 413


def test_makeup_match_photo_and_hex(client, face_png):
    r = client.post("/makeup/match-photo?k=3", content=face_png)
    assert r.status_code == 200 and len(r.json()["matches"]) == 3
    r = client.get("/makeup/match", params={"hex": "#c68642", "k": 4})
    assert r.status_code == 200 and len(r.json()["matches"]) == 4
    assert client.get("/makeup/match", params={"hex": "zzz"}).status_code == 422
    assert "MAC" in client.get("/makeup/brands").json()


def test_index_page(client):
    assert client.get("/").status_code == 200
