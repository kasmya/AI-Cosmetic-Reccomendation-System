"""Versioned API (/api/v1): the contract the frontend builds against.

Every section of a response carries `status`:
  live         produced by real engine code
  provisional  real but simplified (will be replaced in a later phase, shape stays the same)
  planned      not built yet; empty or illustrative, shape is final
See docs/PLAN.md for the phase each planned section lands in.
"""
from typing import Literal

from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field

from .data import SKIN_TYPES
from .recommender import UserProfile
from .state import state
from .vision import TONES, NoFaceError, decode_image, estimate_skin_tone, rgb_hex_to_lab

router = APIRouter(prefix="/api/v1", tags=["v1"])
Status = Literal["live", "provisional", "planned"]

CONCERNS = ["acne", "dryness", "oiliness", "sensitivity", "dark_spots", "redness", "fine_lines", "dullness"]
COLOUR_PREFS = ["warm", "cool", "neutral", "nude", "pink", "coral", "berry", "bold", "natural"]
NOT_MEDICAL = "Informational only, not medical advice. See a dermatologist for persistent or severe skin problems."
# Concern -> ingredient guidance (small hand-written starter map; refined in Phase 3).
CONCERN_INGREDIENTS = {
    "acne": ["salicylic acid", "niacinamide", "benzoyl peroxide"],
    "dryness": ["ceramide", "hyaluronic acid", "glycerin"],
    "oiliness": ["niacinamide", "salicylic acid", "clay"],
    "sensitivity": ["centella", "panthenol", "allantoin"],
    "dark_spots": ["vitamin c", "niacinamide", "azelaic acid"],
    "redness": ["centella", "azelaic acid", "green tea"],
    "fine_lines": ["retinol", "peptide", "vitamin c"],
    "dullness": ["vitamin c", "glycolic acid", "lactic acid"],
}


# ---------------- schemas ----------------
class Options(BaseModel):
    skin_types: list[str]
    concerns: list[str]
    colour_preferences: list[str]
    tones: list[str]
    skincare_categories: list[str]
    makeup_brands: list[str]


class FaceResult(BaseModel):
    tone: str
    ita: float
    skin_hex: str
    lab: list[float]
    face: list[int]
    skin_pixels: int


class AssessRequest(BaseModel):
    skin_type: str | None = None
    concerns: list[str] = []


class ConcernAdvice(BaseModel):
    concern: str
    helpful_ingredients: list[str]


class AssessResponse(BaseModel):
    status: Status
    skin_type: str | None
    concerns: list[ConcernAdvice]
    disclaimer: str = NOT_MEDICAL


class ConsultRequest(BaseModel):
    skin_type: str | None = Field(None, description=f"One of {SKIN_TYPES}")
    concerns: list[str] = []
    colour_preferences: list[str] = []
    sensitive: bool = False
    max_price_per_item: float | None = Field(None, ge=0)
    avoid_ingredients: list[str] = []
    skin_hex: str | None = Field(None, pattern=r"^#?[0-9a-fA-F]{6}$", description="From /face/analyze")
    skin_tone: str | None = Field(None, description=f"One of {TONES}")
    per_category: int = Field(2, ge=1, le=5)


class ScoreBreakdown(BaseModel):
    model_like_probability: float
    review_sentiment: float | None = None   # Phase 4
    concern_fit: float | None = None        # Phase 3/4
    ingredient_flags: list[str] = []


class SkincareItem(BaseModel):
    product_id: int
    name: str
    brand: str
    category: str
    price: float
    source: str
    aicosmetic_score: int = Field(..., ge=0, le=100)
    verdict: Literal["Recommended", "Okay", "Caution"]
    score_status: Status
    breakdown: ScoreBreakdown
    why: str


class ShadeItem(BaseModel):
    brand: str
    product: str
    hex: str
    delta_e: float


class MakeupSection(BaseModel):
    status: Status
    items: list[ShadeItem] = []
    note: str = ""


class MakeupLook(BaseModel):
    skin_hex: str | None
    tone: str | None
    foundation: MakeupSection
    lips: MakeupSection
    blush: MakeupSection
    eyes: MakeupSection


class RoutineStep(BaseModel):
    step: str
    product_id: int | None
    name: str | None


class Routine(BaseModel):
    status: Status
    am: list[RoutineStep]
    pm: list[RoutineStep]
    warnings: list[str] = []


class ConsultResponse(BaseModel):
    profile: AssessResponse
    skincare: dict[str, list[SkincareItem]]
    skincare_status: Status
    makeup: MakeupLook
    routine: Routine
    shopping_list: dict
    disclaimer: str = NOT_MEDICAL


class ReviewSummary(BaseModel):
    status: Status
    product_id: int
    n_reviews: int
    sentiment: float | None
    themes: list[dict]
    quotes: list[str]


class ProductDetail(BaseModel):
    product_id: int
    name: str
    brand: str
    category: str
    price: float
    source: str
    ingredients: list[str]
    review_summary: ReviewSummary


# ---------------- helpers ----------------
def _verdict(score: int) -> str:
    return "Recommended" if score >= 80 else "Okay" if score >= 60 else "Caution"


def _check_inputs(skin_type, concerns, prefs=(), tone=None):
    if skin_type is not None and skin_type not in SKIN_TYPES:
        raise HTTPException(422, f"skin_type must be one of {SKIN_TYPES}")
    for c in concerns:
        if c not in CONCERNS:
            raise HTTPException(422, f"unknown concern '{c}', use {CONCERNS}")
    for p in prefs:
        if p not in COLOUR_PREFS:
            raise HTTPException(422, f"unknown colour preference '{p}', use {COLOUR_PREFS}")
    if tone is not None and tone not in TONES:
        raise HTTPException(422, f"skin_tone must be one of {TONES}")


def _assess(skin_type, concerns) -> AssessResponse:
    return AssessResponse(
        status="provisional", skin_type=skin_type,
        concerns=[ConcernAdvice(concern=c, helpful_ingredients=CONCERN_INGREDIENTS[c]) for c in concerns])


def _review_summary(product_id: int) -> ReviewSummary:
    n = int(state["rec"].products.loc[product_id, "n_reviews"]) if "n_reviews" in state["rec"].products else 0
    return ReviewSummary(status="planned", product_id=product_id, n_reviews=n, sentiment=None, themes=[], quotes=[])


# ---------------- endpoints ----------------
@router.get("/options", response_model=Options)
def options():
    """Everything the frontend needs to build its form controls."""
    return Options(skin_types=SKIN_TYPES, concerns=CONCERNS, colour_preferences=COLOUR_PREFS, tones=TONES,
                   skincare_categories=sorted(state["rec"].products["Category"].unique()),
                   makeup_brands=state["shades"].brands)


@router.post("/face/analyze", response_model=FaceResult)
async def face_analyze(request: Request):
    """Raw image bytes in the request body. Processed in memory, never stored."""
    data = await request.body()
    if not data or len(data) > 5_000_000:
        raise HTTPException(413, "Send an image under 5 MB as the request body")
    try:
        return estimate_skin_tone(decode_image(data))
    except NoFaceError as e:
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/skin/assess", response_model=AssessResponse)
def skin_assess(req: AssessRequest):
    """Maps selected concerns to helpful ingredients. Not a diagnosis."""
    _check_inputs(req.skin_type, req.concerns)
    return _assess(req.skin_type, req.concerns)


def _skincare_item(r) -> SkincareItem:
    """Provisional score: the model's P(like) as 0-100. Phase 4 replaces it with the full blended score."""
    score = int(round(100 * r.p_like))
    return SkincareItem(
        product_id=int(r.Product_id), name=r.Product, brand=r.Brand, category=r.Category, price=float(r.Price),
        source="sephora_reviews", aicosmetic_score=score, verdict=_verdict(score), score_status="provisional",
        breakdown=ScoreBreakdown(model_like_probability=round(float(r.p_like), 3)), why=r.why)


def _foundation(skin_hex: str | None) -> MakeupSection:
    if not skin_hex:
        return MakeupSection(status="planned", note="Upload a photo or pass skin_hex to get foundation matches.")
    matches = state["shades"].match(rgb_hex_to_lab(skin_hex), k=5)
    items = [ShadeItem(**row) for row in matches[["brand", "product", "hex", "delta_e"]].to_dict("records")]
    return MakeupSection(status="live", items=items)


def _routine(skincare: dict[str, list[SkincareItem]]) -> Routine:
    def step(label: str, category: str) -> RoutineStep:
        top = skincare[category][0] if skincare.get(category) else None
        return RoutineStep(step=label, product_id=top.product_id if top else None, name=top.name if top else None)

    steps = [step("Cleanse", "Cleanser"), step("Treat", "Treatment"), step("Moisturise", "Moisturizer")]
    return Routine(status="provisional", am=steps, pm=steps,
                   warnings=["SPF step and ingredient-clash checks arrive in Phase 6."])


@router.post("/consult", response_model=ConsultResponse)
def consult(req: ConsultRequest):
    """One call that returns the whole consultation. Sections not built yet say status='planned'."""
    _check_inputs(req.skin_type, req.concerns, req.colour_preferences, req.skin_tone)
    rec = state["rec"]

    skincare: dict[str, list[SkincareItem]] = {}
    for category in ("Cleanser", "Treatment", "Moisturizer"):
        profile = UserProfile(skin_type=req.skin_type, skin_tone=req.skin_tone, sensitive=req.sensitive,
                              category=category, max_price=req.max_price_per_item,
                              avoid_ingredients=req.avoid_ingredients)
        out = rec.recommend(profile, k=req.per_category).reset_index()
        skincare[category] = [_skincare_item(r) for r in out.itertuples()]

    planned = MakeupSection(status="planned", note="Colour-harmony palettes land in Phase 5.")
    makeup = MakeupLook(skin_hex=req.skin_hex, tone=req.skin_tone, foundation=_foundation(req.skin_hex),
                        lips=planned, blush=planned, eyes=planned)
    picks = [items[0] for items in skincare.values() if items]
    shopping = {"status": "provisional", "items": [i.product_id for i in picks],
                "total_price": round(sum(i.price for i in picks), 2)}
    return ConsultResponse(profile=_assess(req.skin_type, req.concerns), skincare=skincare,
                           skincare_status="provisional", makeup=makeup, routine=_routine(skincare),
                           shopping_list=shopping)


@router.get("/products/{product_id}", response_model=ProductDetail)
def product_detail(product_id: int = Path(..., ge=0)):
    products = state["rec"].products
    if product_id not in products.index:
        raise HTTPException(404, "Unknown product")
    r = products.loc[product_id]
    return ProductDetail(product_id=product_id, name=r["Product"], brand=r["Brand"], category=r["Category"],
                         price=float(r["Price"]), source="sephora_reviews",
                         ingredients=[i.strip() for i in r["Ingredients"].split(",") if i.strip()],
                         review_summary=_review_summary(product_id))


@router.get("/products/{product_id}/reviews/summary", response_model=ReviewSummary)
def review_summary(product_id: int = Path(..., ge=0)):
    """Phase 4 fills sentiment, themes and quotes. The shape is final."""
    if product_id not in state["rec"].products.index:
        raise HTTPException(404, "Unknown product")
    return _review_summary(product_id)
