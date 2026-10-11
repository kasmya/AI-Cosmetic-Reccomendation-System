import collections

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app
from cosmetic_rec.catalog import build_catalog
from cosmetic_rec.makeup import ShadeMatcher
from cosmetic_rec.makeup_look import (
    PALETTES,
    SkinProfile,
    SkinProfiler,
    adapt_hex,
    depth_of,
    match_products,
    name_families,
    suggest,
)
from cosmetic_rec.state import state
from cosmetic_rec.tryon import apply_look
from cosmetic_rec.vision import NoFaceError, rgb_hex_to_lab
from cosmetic_rec.websearch import WebSearch

skimage_data = pytest.importorskip("skimage.data")


@pytest.fixture(scope="module")
def profiler():
    return SkinProfiler(ShadeMatcher().lab)


@pytest.fixture(scope="module")
def face_img():
    img = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)[20:320, 100:400]
    return cv2.resize(img, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)


# ---------- depth and undertone ----------
def test_depth_thresholds():
    assert [depth_of(L) for L in (80, 68, 60, 52, 40)] == ["light", "light", "medium", "medium", "deep"]


def test_undertone_is_balanced_within_each_depth_band(profiler):
    shades = ShadeMatcher().lab
    by = collections.defaultdict(collections.Counter)
    for lab in shades:
        p = profiler.profile(lab)
        by[p.depth][p.undertone] += 1
    for counts in by.values():
        total = sum(counts.values())
        assert all(0.25 < counts[u] / total < 0.42 for u in ("warm", "neutral", "cool"))


def test_same_hue_reads_warmer_on_deeper_skin_than_a_global_cutoff_would(profiler):
    """Deep shades are naturally lower in hue; the same hue angle must not mean the same undertone at every depth."""
    hue = 60.0
    lab = lambda L: np.array([L, 20 * np.cos(np.radians(hue)), 20 * np.sin(np.radians(hue))])  # noqa: E731
    assert profiler.profile(lab(40)).undertone == "warm" and profiler.profile(lab(78)).undertone == "cool"


def test_user_override_and_tone_label_fallback(profiler):
    assert profiler.profile(np.array([70.0, 12.0, 22.0]), "cool").undertone_basis == "you told us"
    p = profiler.from_tone_label("Deep")
    assert p.depth == "deep" and p.undertone_basis == "unknown" and profiler.from_tone_label("nope") is None


# ---------- palettes ----------
WARM = SkinProfile("medium", "warm", "measured")
COOL = SkinProfile("medium", "cool", "measured")


def top(kind, profile, prefs=None, k=3):
    return [s["name"] for s in suggest(kind, profile, prefs, k)]


def test_undertone_changes_the_ranking_in_the_expected_direction():
    assert {"Coral", "Peach", "Brick red", "Orange red", "Terracotta"} & set(top("lips", WARM))
    assert {"Berry", "Deep plum", "Classic blue-red", "Mauve nude"} & set(top("lips", COOL))
    assert set(top("lips", WARM)) != set(top("lips", COOL))


def test_preferences_move_matching_shades_up_and_explain_why():
    assert "Berry" in top("lips", WARM, ["berry"], k=3)
    out = suggest("lips", WARM, ["berry"], 3)
    assert any("preference for" in s["why"] for s in out)
    assert all(s["intensity"] == "bold" for s in suggest("lips", COOL, ["bold"], 3))
    assert all(s["intensity"] == "soft" for s in suggest("lips", COOL, ["natural", "nude"], 3))


def test_unknown_undertone_makes_no_undertone_claim():
    unknown = SkinProfile("medium", "neutral", "unknown")
    assert all("undertone" not in s["why"] for s in suggest("eyes", unknown, [], 4))


def test_every_palette_has_valid_hex_and_depth_adapts():
    for kind, items in PALETTES.items():
        for it in items:
            light, deep = adapt_hex(it["hex"], "light"), adapt_hex(it["hex"], "deep")
            assert light.startswith("#") and len(light) == 7
            assert rgb_hex_to_lab(light)[0] > rgb_hex_to_lab(deep)[0], (kind, it["name"])


# ---------- product matching ----------
def test_colour_words_are_found_in_several_languages():
    assert name_families("Maybelline Lipstick Color Sense Coral Rise 344") == {"coral"}   # "Color" is not "or" (gold)
    assert name_families("L'Oréal Paris Color Riche 108 Brun Cuivré") == {"brown", "bronze"}
    assert name_families("Herbal natural lipstick") == set()
    assert "pink" in name_families("L'Oreal True Match Rose Bonne Mine") and "red" in name_families("Rouge interdit")
    assert name_families("Lip Oil Intense Treatment Gloss") == set()


def test_product_matches_are_in_category_and_share_a_family():
    cat = build_catalog()
    pal = suggest("lips", WARM, ["coral"], 4)
    found = match_products("lips", pal, cat)
    assert found and all(m["matched_on"] for m in found)
    allowed = {f for p in pal for f in p["families"]}
    assert all(set(m["matched_on"]) <= allowed for m in found)
    assert set(cat[cat["product_uid"].isin([m["product_uid"] for m in found])]["category"]) == {"Lipstick"}


# ---------- try-on ----------
def test_tryon_changes_only_the_face_region(face_img):
    out, applied, (x, y, w, h) = apply_look(face_img, "#c68c6a", "#a23b5e", "#ee7f6a")
    assert out.shape == face_img.shape and applied == ["foundation", "lips", "blush"]
    assert np.array_equal(out[:40, :], face_img[:40, :])                       # background / hair top untouched
    assert np.array_equal(out[-120:, :], face_img[-120:, :])                   # clothing untouched
    assert np.abs(out.astype(int) - face_img.astype(int)).max() > 30


def test_foundation_moves_the_skin_colour_towards_the_target(face_img):
    target = "#9a6a4a"
    out, applied, (x, y, w, h) = apply_look(face_img, foundation_hex=target)
    cheek = (slice(y + int(0.5 * h), y + int(0.6 * h)), slice(x + int(0.2 * w), x + int(0.3 * w)))

    def mean_lab(img):
        return cv2.cvtColor(img[cheek].astype(np.float32) / 255, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)

    goal = rgb_hex_to_lab(target)
    assert np.linalg.norm(mean_lab(out) - goal) < np.linalg.norm(mean_lab(face_img) - goal) - 5
    assert applied == ["foundation"]


def test_strength_scales_the_effect_and_blush_only_leaves_lips_alone(face_img):
    soft, _, _ = apply_look(face_img, lips_hex="#a23b5e", strength=0.2)
    full, _, _ = apply_look(face_img, lips_hex="#a23b5e", strength=1.0)
    d = lambda o: np.abs(o.astype(int) - face_img.astype(int)).sum()  # noqa: E731
    assert 0 < d(soft) < d(full)
    blush, applied, (x, y, w, h) = apply_look(face_img, blush_hex="#ee7f6a")
    mouth = (slice(y + int(0.74 * h), y + int(0.82 * h)), slice(x + int(0.4 * w), x + int(0.6 * w)))
    assert applied == ["blush"] and np.array_equal(blush[mouth], face_img[mouth])


def test_tryon_without_a_face_raises():
    with pytest.raises(NoFaceError):
        apply_look(cv2.cvtColor(skimage_data.coffee(), cv2.COLOR_RGB2BGR), lips_hex="#a23b5e")


# ---------- API ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def png(face_img):
    return cv2.imencode(".png", face_img)[1].tobytes()


def test_face_analyze_reports_depth_and_undertone(client, png):
    j = client.post("/api/v1/face/analyze", content=png).json()
    assert j["depth"] in ("light", "medium", "deep") and j["undertone"] in ("warm", "neutral", "cool")
    assert j["undertone_basis"] == "measured"
    assert client.get("/api/v1/options").json()["undertones"] == ["warm", "neutral", "cool"]


def test_consult_makeup_sections_follow_preferences_and_overrides(client):
    body = {"skin_hex": "#c8a07c", "colour_preferences": ["berry", "bold"], "per_category": 1}
    j = client.post("/api/v1/consult", json=body).json()["makeup"]
    assert j["undertone_basis"] == "measured" and j["depth"]
    for kind in ("lips", "blush", "eyes"):
        assert j[kind]["status"] == "live" and len(j[kind]["palette"]) == 4
        assert all(p["hex"].startswith("#") for p in j[kind]["palette"])
    assert any("berry" in p["why"] for p in j["lips"]["palette"])
    forced = client.post("/api/v1/consult", json={**body, "undertone": "cool"}).json()["makeup"]
    assert forced["undertone"] == "cool" and forced["undertone_basis"] == "you told us"


def test_consult_with_only_a_tone_label_or_nothing(client):
    j = client.post("/api/v1/consult", json={"skin_tone": "Deep", "per_category": 1}).json()["makeup"]
    assert j["depth"] == "deep" and j["undertone_basis"] == "unknown" and j["lips"]["status"] == "live"
    none = client.post("/api/v1/consult", json={"per_category": 1}).json()["makeup"]
    assert none["lips"]["status"] == "planned" and none["depth"] is None
    assert client.post("/api/v1/consult", json={"undertone": "purple"}).status_code == 422


def test_tryon_endpoint(client, png):
    r = client.post("/api/v1/makeup/tryon", params={"lips_hex": "#a23b5e", "blush_hex": "#ee7f6a"}, content=png)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-applied"] == "lips,blush"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), 1).shape[:2] == (660, 660)
    assert client.post("/api/v1/makeup/tryon", content=png).status_code == 422
    assert client.post("/api/v1/makeup/tryon", params={"lips_hex": "nope"}, content=png).status_code == 422
    assert client.post("/api/v1/makeup/tryon", params={"lips_hex": "#a23b5e"}, content=b"junk").status_code == 400
    assert client.post("/api/v1/makeup/tryon", params={"lips_hex": "#a23b5e"}, content=b"").status_code == 413


def test_makeup_web_endpoint_disabled_live_and_validated(client):
    state["web"] = WebSearch(api_key="")
    j = client.get("/api/v1/makeup/web", params={"kind": "lips", "shade": "coral", "undertone": "warm"}).json()
    assert j["status"] == "disabled" and "coral lipstick" in j["query"] and "warm undertone" in j["query"]
    reply = {"results": [{"title": "T", "url": "https://example.com", "content": "c"}]}
    state["web"] = WebSearch(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    assert client.get("/api/v1/makeup/web", params={"kind": "blush"}).json()["status"] == "live"
    assert client.get("/api/v1/makeup/web", params={"kind": "hair"}).status_code == 422
    assert client.get("/api/v1/makeup/web", params={"kind": "lips", "shade": "a;drop"}).status_code == 422
