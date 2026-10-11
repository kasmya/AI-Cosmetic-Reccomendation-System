"""FastAPI service.  Run: uvicorn cosmetic_rec.api:app --reload   (docs at /docs)"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import build_catalog
from .catalog_index import CatalogIndex
from .data import SKIN_TYPES
from .makeup import ShadeMatcher
from .recommender import UserProfile
from .state import state
from .train import load_all
from .v1 import router as v1_router
from .vision import TONES, NoFaceError, decode_image, estimate_skin_tone, rgb_hex_to_lab
from .websearch import WebSearch, load_dotenv

log = logging.getLogger("cosmetic_rec.api")
ROOT = Path(__file__).resolve().parent.parent
# The web UI lives in frontend/ (older checkouts used static/); serve whichever exists.
FRONTEND = next((d for d in (ROOT / "frontend", ROOT / "static") if (d / "index.html").exists()), ROOT / "frontend")
MAX_IMAGE_BYTES = 5_000_000


@asynccontextmanager
async def lifespan(app):
    blob = load_all()
    state["rec"], state["intel"], state["meta"] = blob["model"], blob["intel"], blob["meta"]
    state["shades"] = ShadeMatcher()
    state["catalog"] = CatalogIndex(build_catalog())
    load_dotenv()
    state["web"] = WebSearch()
    log.info("ready: %s", state["meta"])
    yield


app = FastAPI(title="Cosmetic Recommender", version="2.0", lifespan=lifespan,
              description="Skincare recommendations (ML) and foundation shade matching (OpenCV).")


app.include_router(v1_router)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


# ---------------- meta ----------------
@app.get("/health")
def health():
    return {"status": "ok", "products": len(state["rec"].products), "shades": len(state["shades"].shades)}


@app.get("/model-info")
def model_info():
    return state["meta"]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")


# ---------------- feature 1: skincare ----------------
class SkincareRequest(BaseModel):
    skin_type: str | None = Field(None, description=f"One of {SKIN_TYPES}")
    skin_tone: str | None = Field(None, description=f"One of {TONES}")
    sensitive: bool = False
    category: str | None = None
    max_price: float | None = Field(None, ge=0)
    avoid_ingredients: list[str] = []
    liked_product_ids: list[int] = []
    k: int = Field(10, ge=1, le=50)


@app.get("/skincare/products")
def skincare_products(q: str = "", limit: int = Query(20, ge=1, le=400)):
    p = state["rec"].products
    if q:
        hit = p["Product"].str.contains(q, case=False, regex=False)
        hit |= p["Brand"].str.contains(q, case=False, regex=False)
        p = p[hit]
    cols = ["Product_id", "Product", "Brand", "Category", "Price"]
    return p.sort_values("Product").head(limit).reset_index()[cols].to_dict("records")


@app.post("/skincare/recommend")
def skincare_recommend(req: SkincareRequest):
    if req.skin_type is not None and req.skin_type not in SKIN_TYPES:
        raise HTTPException(422, f"skin_type must be one of {SKIN_TYPES}")
    if req.skin_tone is not None and req.skin_tone not in TONES:
        raise HTTPException(422, f"skin_tone must be one of {TONES}")
    out = state["rec"].recommend(UserProfile(**req.model_dump(exclude={"k"})), k=req.k).reset_index()
    cols = ["Product_id", "Product", "Brand", "Category", "Price", "score", "p_like", "why"]
    return {"count": len(out), "results": out[cols].to_dict("records")}


# ---------------- feature 2: makeup (OpenCV + k-NN) ----------------
async def _read_image(request: Request):
    data = await request.body()
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Send an image under 5 MB as the request body")
    try:
        return decode_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/face/analyze")
async def face_analyze(request: Request):
    """Raw image bytes in the body. Processed in memory, never stored."""
    img = await _read_image(request)
    try:
        return estimate_skin_tone(img)
    except NoFaceError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/makeup/match-photo")
async def makeup_match_photo(request: Request, k: int = Query(5, ge=1, le=20), brand: str | None = None):
    """Photo in -> OpenCV skin colour -> closest real foundation shades."""
    img = await _read_image(request)
    try:
        face = estimate_skin_tone(img)
    except NoFaceError as e:
        raise HTTPException(422, str(e)) from e
    shades = state["shades"].match(face["lab"], k=k, brand=brand)
    return {"face": face, "matches": shades.to_dict("records")}


@app.get("/makeup/match")
def makeup_match(hex: str = Query(..., pattern=r"^#?[0-9a-fA-F]{6}$"), k: int = Query(5, ge=1, le=20),
                 brand: str | None = None):
    """No photo? Match from a skin-colour hex code instead."""
    shades = state["shades"].match(rgb_hex_to_lab(hex), k=k, brand=brand)
    return {"matches": shades.to_dict("records")}


@app.get("/makeup/brands")
def makeup_brands():
    return state["shades"].brands


# Images and other assets for the UI (frontend/public/assets/*), referenced from the page as /assets/<file>.
if (FRONTEND / "public" / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "public" / "assets"), name="assets")
