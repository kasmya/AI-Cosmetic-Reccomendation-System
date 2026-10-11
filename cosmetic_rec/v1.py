"""Versioned API (/api/v1): the contract the frontend builds against.

Every section of a response carries `status`:
  live         produced by real engine code
  provisional  real but simplified (will be replaced in a later phase, shape stays the same)
  planned      not built yet; empty or illustrative, shape is final
See docs/PLAN.md for the phase each planned section lands in.
"""
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from . import concerns as cx
from .concerns import CONCERNS
from .data import SKIN_TYPES
from .recommender import UserProfile
from .score import combine, confidence, review_component
from .state import state
from .vision import TONES, NoFaceError, decode_image, estimate_skin_tone, rgb_hex_to_lab

router = APIRouter(prefix="/api/v1", tags=["v1"])
Status = Literal["live", "provisional", "planned"]

COLOUR_PREFS = ["warm", "cool", "neutral", "nude", "pink", "coral", "berry", "bold", "natural"]
NOT_MEDICAL = "Informational only, not medical advice. See a dermatologist for persistent or severe skin problems."


# ---------------- schemas ----------------
class Options(BaseModel):
    skin_types: list[str]
    concerns: list[str]
    colour_preferences: list[str]
    tones: list[str]
    skincare_categories: list[str]
    makeup_brands: list[str]


class FaceResult(BaseModel):
    hints: dict | None = None      # low-confidence photo hints (redness, shine, blemishes); forward to /skin/assess
    tone: str
    ita: float
    skin_hex: str
    lab: list[float]
    face: list[int]
    skin_pixels: int


class AssessRequest(BaseModel):
    skin_type: str | None = None
    concerns: list[str] = Field([], description="Concerns the user selected directly")
    answers: dict[str, str] = Field({}, description="Questionnaire answers: {question_id: option_value}")
    photo_hints: dict | None = Field(None, description="`hints` from /face/analyze (optional, low confidence)")


class ConcernAdvice(BaseModel):
    concern: str
    level: Literal["selected", "likely", "possible"] = "selected"
    score: float = 1.0
    reasons: list[str] = []
    helpful_ingredients: list[str]
    ingredients_to_watch: list[str] = []
    tip: str = ""


class AssessResponse(BaseModel):
    status: Status
    skin_type: str | None
    skin_type_source: str = ""
    concerns: list[ConcernAdvice]
    photo_hints_used: bool = False
    disclaimer: str = NOT_MEDICAL


class Question(BaseModel):
    id: str
    text: str
    options: list[dict]


class ConsultRequest(BaseModel):
    skin_type: str | None = Field(None, description=f"One of {SKIN_TYPES}")
    concerns: list[str] = []
    answers: dict[str, str] = Field({}, description="Questionnaire answers, see GET /skin/questionnaire")
    photo_hints: dict | None = None
    colour_preferences: list[str] = []
    sensitive: bool = False
    max_price_per_item: float | None = Field(None, ge=0)
    avoid_ingredients: list[str] = []
    skin_hex: str | None = Field(None, pattern=r"^#?[0-9a-fA-F]{6}$", description="From /face/analyze")
    skin_tone: str | None = Field(None, description=f"One of {TONES}")
    per_category: int = Field(2, ge=1, le=5)


class ScoreBreakdown(BaseModel):
    model_like_probability: float            # calibrated P(user rates it 4-5 stars)
    review_sentiment: float | None = None    # smoothed share of positive reviews (skin-matched when possible)
    review_basis: str = ""                   # which reviewers that share is based on
    concern_fit: float | None = None         # 0-1, only when the user has concerns
    ingredient_flags: list[str] = []         # ingredients often reported as aggravating for the user's concerns
    points: dict[str, float] = {}            # each component's contribution to the 0-100 score
    weights: dict[str, float] = {}
    penalty: int = 0
    confidence: str = ""                     # low / medium / high, from the number of reviews


class Alternative(BaseModel):
    """A similar product from a different data source (matched by ingredient list)."""
    product_uid: str
    name: str
    brand: str
    category: str
    source: str
    price: float | None
    similarity: float
    url: str | None


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
    alternatives: list[Alternative] = []


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


class ThemeItem(BaseModel):
    aspect: str
    mentions: int
    sentiment: float
    verdict: Literal["praised", "criticised", "typical"]   # relative to how all products score on this aspect


class QuoteItem(BaseModel):
    text: str
    sentiment: Literal["positive", "critical"]
    stars: int


class ReviewSummary(BaseModel):
    status: Status
    product_id: int
    n_reviews: int
    sentiment: float | None                  # raw share of positive (4-5 star) reviews
    skin_type_reviews: int = 0               # reviews from people with the requested skin type
    skin_type_sentiment: float | None = None
    themes: list[dict]
    pros: list[str] = []
    cons: list[str] = []
    quotes: list[str]
    quote_items: list[QuoteItem] = []
    confidence: str = ""
    note: str = ""


class CatalogProduct(BaseModel):
    product_uid: str
    source: str
    name: str
    brand: str
    category: str
    price: float | None
    url: str | None
    n_ingredients: int


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


def _assess(skin_type, selected, answers=None, photo_hints=None) -> AssessResponse:
    try:
        decided, source, found = cx.assess(answers, selected, photo_hints, skin_type)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    advice = [ConcernAdvice(concern=c.concern, level=c.level, score=c.score, reasons=c.reasons,
                            helpful_ingredients=cx.helpful_names(c.concern),
                            ingredients_to_watch=cx.watch_names(c.concern), tip=cx.TIPS.get(c.concern, ""))
              for c in found]
    return AssessResponse(status="live", skin_type=decided, skin_type_source=source, concerns=advice,
                          photo_hints_used=bool(photo_hints))


def _review_summary(product_id: int, skin_type: str | None = None) -> ReviewSummary:
    intel = state["intel"]
    c = intel.counts(product_id, skin_type)
    themes = intel.themes(product_id)
    quotes = intel.quotes(product_id)
    share = round(c["pos"] / c["n"], 3) if c["n"] else None
    skin_share = round(c["pos_skin"] / c["n_skin"], 3) if c["n_skin"] else None
    return ReviewSummary(
        status="live", product_id=product_id, n_reviews=c["n"], sentiment=share,
        skin_type_reviews=c["n_skin"], skin_type_sentiment=skin_share,
        themes=[{"aspect": t.aspect, "mentions": t.mentions, "sentiment": t.sentiment, "verdict": t.verdict}
                for t in themes],
        pros=[t.aspect for t in themes if t.verdict == "praised"],
        cons=[t.aspect for t in themes if t.verdict == "criticised"],
        quotes=[q["text"] for q in quotes], quote_items=[QuoteItem(**q) for q in quotes],
        confidence=confidence(c["n"]),
        note="Themes and quotes come from our dataset's user-written reviews. Polarity is a model estimate.")


def _score(product_id: int, p_like: float, skin_type: str | None,
           concerns: list[str]) -> tuple[int, str, ScoreBreakdown]:
    """The AICosmetic score for one product (formula and weights: see score.py)."""
    intel = state["intel"]
    ingredients = state["rec"].products.at[product_id, "Ingredients"]
    counts = intel.counts(product_id, skin_type)
    share, basis = review_component(counts, intel.global_share, skin_type)
    fit = cx.concern_fit(concerns, ingredients)
    flags = cx.watch_flags(concerns, ingredients)
    result = combine(p_like, share, fit, len(flags))
    return result["score"], result["verdict"], ScoreBreakdown(
        model_like_probability=round(p_like, 3), review_sentiment=round(share, 3), review_basis=basis,
        concern_fit=None if fit is None else round(fit, 2), ingredient_flags=flags, points=result["points"],
        weights=result["weights"], penalty=result["penalty"], confidence=confidence(counts["n"]))


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


@router.get("/skin/questionnaire", response_model=list[Question])
def skin_questionnaire():
    """The skin questionnaire. Send answers as {question_id: option_value} to /skin/assess or /consult."""
    return cx.public_questionnaire()


@router.post("/skin/assess", response_model=AssessResponse)
def skin_assess(req: AssessRequest):
    """Scores concerns from the questionnaire, selected concerns and (optionally) photo hints. Not a diagnosis."""
    _check_inputs(req.skin_type, req.concerns)
    return _assess(req.skin_type, req.concerns, req.answers, req.photo_hints)


def _alternatives(product_id: int) -> list[Alternative]:
    similar = state["catalog"].similar(f"sephora_reviews:{product_id}", k=2)
    return [Alternative(product_uid=r.product_uid, name=r.name, brand=r.brand, category=r.category, source=r.source,
                        price=None if pd.isna(r.price) else float(r.price), similarity=float(r.similarity),
                        url=r.url if isinstance(r.url, str) else None) for r in similar.itertuples()]


def _skincare_item(r, skin_type: str | None, concerns: list[str]) -> SkincareItem:
    score, verdict, breakdown = _score(int(r.Product_id), float(r.p_like), skin_type, concerns)
    return SkincareItem(
        product_id=int(r.Product_id), name=r.Product, brand=r.Brand, category=r.Category, price=float(r.Price),
        source="sephora_reviews", aicosmetic_score=score, verdict=verdict, score_status="live",
        breakdown=breakdown, why=r.why, alternatives=_alternatives(int(r.Product_id)))


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
    profile_result = _assess(req.skin_type, req.concerns, req.answers, req.photo_hints)
    skin_type = profile_result.skin_type
    active_concerns = [c.concern for c in profile_result.concerns]

    skincare: dict[str, list[SkincareItem]] = {}
    for category in ("Cleanser", "Treatment", "Moisturizer"):
        profile = UserProfile(skin_type=skin_type, skin_tone=req.skin_tone, sensitive=req.sensitive,
                              concerns=active_concerns,
                              category=category, max_price=req.max_price_per_item,
                              avoid_ingredients=req.avoid_ingredients)
        out = rec.recommend(profile, k=req.per_category).reset_index()
        skincare[category] = [_skincare_item(r, skin_type, active_concerns) for r in out.itertuples()]

    planned = MakeupSection(status="planned", note="Colour-harmony palettes land in Phase 5.")
    makeup = MakeupLook(skin_hex=req.skin_hex, tone=req.skin_tone, foundation=_foundation(req.skin_hex),
                        lips=planned, blush=planned, eyes=planned)
    picks = [items[0] for items in skincare.values() if items]
    shopping = {"status": "provisional", "items": [i.product_id for i in picks],
                "total_price": round(sum(i.price for i in picks), 2)}
    return ConsultResponse(profile=profile_result, skincare=skincare,
                           skincare_status="live", makeup=makeup, routine=_routine(skincare),
                           shopping_list=shopping)


@router.get("/products/{product_id}", response_model=ProductDetail)
def product_detail(product_id: int = Path(..., ge=0), skin_type: str | None = None):
    products = state["rec"].products
    if product_id not in products.index:
        raise HTTPException(404, "Unknown product")
    r = products.loc[product_id]
    return ProductDetail(product_id=product_id, name=r["Product"], brand=r["Brand"], category=r["Category"],
                         price=float(r["Price"]), source="sephora_reviews",
                         ingredients=[i.strip() for i in r["Ingredients"].split(",") if i.strip()],
                         review_summary=_review_summary(product_id, skin_type))


@router.get("/products/{product_id}/reviews/summary", response_model=ReviewSummary)
def review_summary(product_id: int = Path(..., ge=0), skin_type: str | None = None):
    """Themes (praised / criticised / typical), pros and cons, and representative quotes from the reviews."""
    if product_id not in state["rec"].products.index:
        raise HTTPException(404, "Unknown product")
    _check_inputs(skin_type, [])
    return _review_summary(product_id, skin_type)


class ProductScore(BaseModel):
    product_id: int
    aicosmetic_score: int = Field(..., ge=0, le=100)
    verdict: Literal["Recommended", "Okay", "Caution"]
    breakdown: ScoreBreakdown
    how_it_works: str


@router.get("/products/{product_id}/score", response_model=ProductScore)
def product_score(product_id: int = Path(..., ge=0), skin_type: str | None = None,
                  concerns: Annotated[list[str] | None, Query()] = None, skin_tone: str | None = None):
    """The AICosmetic score for any product, personalised by skin type (and optionally concerns)."""
    if product_id not in state["rec"].products.index:
        raise HTTPException(404, "Unknown product")
    concerns = concerns or []
    _check_inputs(skin_type, concerns, tone=skin_tone)
    p_like = float(state["rec"].predict_like(skin_type, skin_tone)[product_id])
    score, verdict, breakdown = _score(product_id, p_like, skin_type, concerns)
    return ProductScore(product_id=product_id, aicosmetic_score=score, verdict=verdict, breakdown=breakdown,
                        how_it_works="Weighted blend of the model's calibrated probability, the smoothed share of "
                                     "positive reviews and (with concerns) ingredient fit, minus a penalty for "
                                     "ingredients often reported as aggravating. Formula: cosmetic_rec/score.py.")


@router.get("/catalog/stats")
def catalog_stats():
    """Products per source and category, with licences. Shows where the data comes from."""
    return state["catalog"].stats()


@router.get("/catalog/search", response_model=list[CatalogProduct])
def catalog_search(q: str = "", category: str | None = None, source: str | None = None,
                   limit: int = Query(20, ge=1, le=100)):
    """Search every source at once."""
    rows = state["catalog"].search(q, category, source, limit)
    return [CatalogProduct(product_uid=r.product_uid, source=r.source, name=r.name, brand=r.brand, category=r.category,
                           price=None if pd.isna(r.price) else float(r.price),
                           url=r.url if isinstance(r.url, str) else None,
                           n_ingredients=len(r.ingredients.split(" | ")) if r.ingredients else 0)
            for r in rows.itertuples()]


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchResponse(BaseModel):
    status: Literal["live", "disabled", "error"]
    query: str
    results: list[WebResult] = []
    note: str = ""


@router.get("/products/{product_id}/web", response_model=WebSearchResponse)
def product_web(product_id: int = Path(..., ge=0)):
    """Live web results about a product (reviews, availability). Needs TAVILY_API_KEY; otherwise status='disabled'."""
    products = state["rec"].products
    if product_id not in products.index:
        raise HTTPException(404, "Unknown product")
    r = products.loc[product_id]
    query = f"{r['Brand']} {r['Product']} skincare product review"
    web = state["web"]
    if not web.enabled:
        return WebSearchResponse(status="disabled", query=query, note="Set TAVILY_API_KEY to enable live web search.")
    try:
        found = web.search(query)
    except RuntimeError as e:
        return WebSearchResponse(status="error", query=query, note=str(e))
    return WebSearchResponse(status="live", query=query, results=[WebResult(**x) for x in found],
                             note="Third-party web content, shown as-is. Not used by the models.")
