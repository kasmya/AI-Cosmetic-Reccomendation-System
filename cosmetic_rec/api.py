"""FastAPI service.  Run:  uvicorn cosmetic_rec.api:app --reload"""
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .data import SKIN_TYPES, load_reviews
from .recommender import Recommender, UserProfile
from .vision import TONES, NoFaceError, decode_image, estimate_skin_tone

state: dict = {}


@asynccontextmanager
async def lifespan(app):
    state["rec"] = Recommender().fit(load_reviews())
    yield


app = FastAPI(title="Cosmetic Recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    skin_type: str | None = Field(None, description=f"One of {SKIN_TYPES}")
    skin_tone: str | None = Field(None, description=f"One of {TONES}")
    sensitive: bool = False
    category: str | None = None
    max_price: float | None = Field(None, ge=0)
    avoid_ingredients: list[str] = []
    liked_product_ids: list[int] = []
    k: int = Field(10, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok", "products": len(state["rec"].products)}


@app.get("/products")
def products(q: str = "", limit: int = 20):
    p = state["rec"].products
    if q:
        p = p[p["Product"].str.contains(q, case=False, regex=False) | p["Brand"].str.contains(q, case=False, regex=False)]
    return p.sort_values("Product").head(limit).reset_index()[["Product_id", "Product", "Brand", "Category", "Price"]].to_dict("records")


@app.post("/analyze-photo")
async def analyze_photo(request: Request):
    """Raw image bytes in the body. Processed in memory, never stored."""
    data = await request.body()
    if not data or len(data) > 5_000_000:
        raise HTTPException(413, "Send an image under 5 MB")
    try:
        return estimate_skin_tone(decode_image(data))
    except NoFaceError as e:
        raise HTTPException(422, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.skin_type is not None and req.skin_type not in SKIN_TYPES:
        raise HTTPException(422, f"skin_type must be one of {SKIN_TYPES}")
    if req.skin_tone is not None and req.skin_tone not in TONES:
        raise HTTPException(422, f"skin_tone must be one of {TONES}")
    profile = UserProfile(**req.model_dump(exclude={"k"}))
    out = state["rec"].recommend(profile, k=req.k).reset_index()
    return {"count": len(out), "results": out[["Product_id", "Product", "Brand", "Category", "Price",
                                               "score", "p_like", "why"]].to_dict("records")}


STATIC = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
