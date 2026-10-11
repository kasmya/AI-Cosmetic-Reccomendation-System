import cv2
import numpy as np
import pytest

from cosmetic_rec.makeup import ShadeMatcher
from cosmetic_rec.vision import NoFaceError, estimate_skin_tone, ita_to_tone, lab_to_hex, rgb_hex_to_lab, tone_index

skimage_data = pytest.importorskip("skimage.data")


def test_ita_mapping_is_monotonic():
    tones = [ita_to_tone(a) for a in (70, 55, 45, 30, 22, 15, 0, -20, -50)]
    assert [tone_index(t) for t in tones] == list(range(9))


def test_hex_lab_roundtrip():
    assert lab_to_hex(rgb_hex_to_lab("#c68642")) == "#c68642"


def test_exact_swatch_matches_itself():
    m = ShadeMatcher()
    row = m.shades.iloc[10]
    best = m.match(rgb_hex_to_lab(row["hex"]), k=1).iloc[0]
    assert best["delta_e"] < 0.5


def test_brand_filter_and_distinct_products():
    m = ShadeMatcher()
    out = m.match(rgb_hex_to_lab("#8d5524"), k=5, brand="mac")
    assert (out["brand"] == "MAC").all()
    assert not out.duplicated(["brand", "product"]).any()
    assert out["delta_e"].is_monotonic_increasing


def test_face_detected_and_skin_measured():
    face = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)
    out = estimate_skin_tone(face)
    assert out["tone"] in ("Porcelain", "Fair", "Light") and out["skin_pixels"] > 100
    assert out["skin_hex"].startswith("#") and len(out["lab"]) == 3


def test_no_face_raises():
    with pytest.raises(NoFaceError):
        estimate_skin_tone(cv2.cvtColor(skimage_data.coffee(), cv2.COLOR_RGB2BGR))


def test_brighter_photo_reads_lighter():
    face = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)
    dark = np.clip(255 * (face / 255.0) ** 1.5, 0, 255).astype(np.uint8)
    assert estimate_skin_tone(dark)["lab"][0] < estimate_skin_tone(face)["lab"][0]
