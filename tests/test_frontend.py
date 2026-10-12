"""The page (frontend/index.html) is checked against the real API, so a guessed URL, parameter or field name fails here
instead of silently showing nothing (or, worse, invented data) in the browser."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from cosmetic_rec.api import app

ROOT = Path(__file__).resolve().parent.parent
# FRONTEND_HTML=path/to/other.html runs these checks against a different file
HTML = Path(os.environ.get("FRONTEND_HTML") or ROOT / "frontend" / "index.html").read_text()
skimage_data = pytest.importorskip("skimage.data")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------- images ----------
def test_every_image_the_page_references_exists_and_is_served(client):
    refs = set(re.findall(r'src="assets/([\w\-.]+)"', HTML))
    assert len(refs) >= 8
    for name in refs:
        assert (ROOT / "frontend" / "assets" / name).is_file(), f"missing file frontend/assets/{name}"
        r = client.get(f"/assets/{name}")
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/"), name


def test_asset_paths_are_relative_so_the_page_also_works_when_opened_from_disk():
    assert 'src="/assets/' not in HTML and "public/assets" not in HTML


# ---------- URLs ----------
def api_paths_called_by_the_page() -> set[str]:
    found = set()
    for raw in re.findall(r"""['"`](/(?:api/v1|model-info|health|skincare)[^'"`?\s]*)""", HTML):
        found.add(re.sub(r"\$\{[^}]*\}", "X", raw).rstrip("/"))
    return found


def test_every_url_the_page_calls_exists_in_the_api(client):
    paths = [re.sub(r"\{[^}]+\}", "X", p) for p in client.get("/openapi.json").json()["paths"]]
    called = api_paths_called_by_the_page()
    assert len(called) >= 12, called
    for url in called:
        assert url in paths, f"the page calls {url} but the API has no such path"


def test_the_try_on_uses_the_real_endpoint_and_parameter_names(client):
    assert "/api/v1/makeup/tryon" in HTML and "/face/tryon" not in HTML
    op = client.get("/openapi.json").json()["paths"]["/api/v1/makeup/tryon"]["post"]
    params = {p["name"] for p in op["parameters"]}
    keys = re.findall(r"\['(\w+)', '[A-Z]\w+'\]", re.search(r"const T_ROWS = .*", HTML).group(0))
    assert set(keys) == {"foundation", "lips", "blush"}
    assert {f"{k}_hex" for k in keys} <= params and "strength" in params          # the page builds k + '_hex'
    assert "k + '_hex'" in HTML


def test_the_page_never_fabricates_results_when_the_api_fails():
    assert "data fallback layer" not in HTML and "window.fetch = async" not in HTML
    assert 'id="conn"' in HTML          # an honest connection banner instead


# ---------- request bodies ----------
def body_keys(after: str) -> set[str]:
    block = HTML[HTML.index(after):]
    block = block[block.index("{"):]
    depth, end = 0, 0
    for i, ch in enumerate(block):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            end = i
            break
    return set(re.findall(r"(?:^|[\s,{])(\w+):", re.sub(r"\([^()]*\)|\[[^\[\]]*\]", "()", block[:end])))


def test_consult_request_fields_are_all_accepted_by_the_api(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ConsultRequest"]["properties"]
    sent = body_keys("const body = {\n    skin_type: $('skin_type')")
    assert {"skin_type", "concerns", "answers", "photo_hints", "total_budget", "undertone", "skin_hex"} <= sent
    assert sent <= set(schema), sent - set(schema)


@pytest.mark.parametrize("marker,schema_name", [
    ("body: JSON.stringify({skin_type: $('skin_type').value || null, concerns: picked('concerns'), answers",
     "AssessRequest"),
    ("body: JSON.stringify({product_id: pid, helpful, request_id: rid})", "FeedbackRequest"),
    ("body: JSON.stringify({am: rc.am.map", "RoutineCheckRequest")])
def test_other_request_bodies_match_their_schemas(client, marker, schema_name):
    props = set(client.get("/openapi.json").json()["components"]["schemas"][schema_name]["properties"])
    seg = HTML[HTML.index(marker) + len("body: JSON.stringify("):]
    sent = set(re.findall(r"(?:^|[{,\s])(\w+)(?::|,|\})", seg[: seg.index("})")]))
    sent = {k for k in sent if k in props or k in {"am", "pm", "skin_type", "concerns", "answers", "photo_hints"}}
    assert sent and sent <= props, (schema_name, sent - props)


def test_options_the_page_hardcodes_are_values_the_api_accepts():
    # the API only knows warm / neutral / cool, and the page fills those in from /options
    assert 'value="olive"' not in HTML


# ---------- response fields the renderers read ----------
def dig(obj, path: str):
    for part in path.split("."):
        if part.endswith("[]"):
            part = part[:-2]
            obj = obj[part][0]
        else:
            obj = obj[part]
    return obj


CONSULT_FIELDS = [
    "request_id", "profile.skin_type", "profile.skin_type_source", "profile.concerns[].reasons",
    "profile.concerns[].level", "profile.concerns[].helpful_ingredients", "profile.concerns[].ingredients_to_watch",
    "profile.concerns[].tip", "profile.disclaimer", "skincare_status",
    "skincare.Cleanser[].breakdown.points", "skincare.Cleanser[].breakdown.weights",
    "skincare.Cleanser[].breakdown.confidence", "skincare.Cleanser[].breakdown.ingredient_flags",
    "skincare.Cleanser[].aicosmetic_score", "skincare.Cleanser[].verdict", "skincare.Cleanser[].why",
    "skincare.Cleanser[].alternatives", "routine.am[].step", "routine.am[].note", "routine.clashes", "routine.notes",
    "routine.status", "makeup.depth", "makeup.undertone", "makeup.undertone_basis",
    "makeup.foundation.items[].delta_e", "makeup.lips.palette[].hex", "makeup.lips.palette[].why",
    "makeup.lips.products", "makeup.eyes.palette", "shopping_list.products[].steps", "shopping_list.products[].price",
    "shopping_list.within_budget", "shopping_list.swaps", "shopping_list.notes", "shopping_list.total_price",
    "shopping_list.budget"]


def test_every_field_the_page_reads_exists_in_a_real_consultation(client):
    j = client.post("/api/v1/consult", json={"skin_type": "Dry", "concerns": ["dryness"], "total_budget": 150,
                                             "skin_hex": "#c68642", "per_category": 2}).json()
    for path in CONSULT_FIELDS:
        try:
            dig(j, path)
        except (KeyError, IndexError, TypeError) as e:
            pytest.fail(f"the consult response has no '{path}' ({e!r})")
    for path in CONSULT_FIELDS:                      # and the page really does use these names
        leaf = path.split(".")[-1].replace("[]", "")
        assert leaf in HTML, f"the page never reads '{leaf}', so '{path}' is not wired up"


def test_product_drawer_fields_exist_in_the_real_responses(client):
    pid = client.get("/skincare/products?limit=1").json()[0]["Product_id"]
    p = client.get(f"/api/v1/products/{pid}", params={"skin_type": "Dry"}).json()
    sc = client.get(f"/api/v1/products/{pid}/score", params={"skin_type": "Dry", "concerns": ["dryness"]}).json()
    for field in ("name", "brand", "category", "price", "source", "ingredients", "review_summary"):
        assert field in p
    for field in ("n_reviews", "sentiment", "pros", "cons", "themes", "quote_items", "skin_type_reviews",
                  "skin_type_sentiment", "note", "status"):
        assert field in p["review_summary"], field
    for field in ("aicosmetic_score", "verdict", "breakdown", "how_it_works"):
        assert field in sc
    assert {"points", "weights", "penalty", "review_basis", "confidence"} <= sc["breakdown"].keys()
    dupes = client.get(f"/api/v1/products/{pid}/dupes", params={"skin_type": "Dry"}).json()
    assert {"swaps", "note"} <= dupes.keys()
    web = client.get(f"/api/v1/products/{pid}/web").json()
    assert {"status", "note", "results"} <= web.keys()
    shop = client.get("/api/v1/makeup/web", params={"kind": "lips", "shade": "Peach"}).json()
    assert {"status", "query", "note", "results"} <= shop.keys()


def test_photo_flow_fields_exist(client):
    ok, png = cv2.imencode(".png", cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR))
    face = client.post("/api/v1/face/analyze", content=png.tobytes()).json()
    assert {"skin_hex", "tone", "depth", "undertone", "hints"} <= face.keys()
    assert {"redness", "shine", "blemishes", "note", "confidence"} <= face["hints"].keys()
    assert all("level" in face["hints"][k] for k in ("redness", "shine", "blemishes"))
    r = client.post("/api/v1/makeup/tryon", params={"lips_hex": "#a23b5e", "strength": 0.6}, content=png.tobytes())
    assert r.status_code == 200 and "x-applied" in r.headers
    opts = client.get("/api/v1/options").json()
    assert {"skin_types", "concerns", "colour_preferences", "undertones", "tones"} <= opts.keys()
    assert {"id", "text", "options"} <= client.get("/api/v1/skin/questionnaire").json()[0].keys()


# ---------- the page's JavaScript parses ----------
@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_inline_scripts_have_no_syntax_errors():
    scripts = re.findall(r"<script(?![^>]*src)[^>]*>(.*?)</script>", HTML, re.S)
    assert len(scripts) >= 2
    for i, code in enumerate(scripts):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
        assert r.returncode == 0, f"script block {i}: {r.stderr[:400]}"


def test_every_v1_endpoint_the_page_uses_for_the_new_studio_exists(client):
    paths = client.get("/openapi.json").json()["paths"]
    for url in ("/api/v1/makeup/suggest", "/api/v1/makeup/brands", "/api/v1/makeup/match", "/api/v1/system"):
        assert url in paths and url in HTML, url
    assert "liked_product_ids" in HTML and "per_category" in HTML
