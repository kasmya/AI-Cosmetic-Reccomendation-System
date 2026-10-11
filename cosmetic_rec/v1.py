"""Versioned API (/api/v1): the contract the frontend builds against.

Every section of a response carries `status`:
  live         produced by real engine code
  provisional  real but simplified (will be replaced in a later phase, shape stays the same)
  planned      not built yet; empty or illustrative, shape is final
See docs/PLAN.md for the phase each planned section lands in.
"""
from typing import Annotated, Literal

import cv2
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, Field

from . import concerns as cx
from . import makeup_look as ml
from . import metrics, planner
from . import routine as rt
from .concerns import CONCERNS
from .data import SKIN_TYPES
from .drift import compute_drift
from .recommender import UserProfile
from .score import combine, confidence, review_component
from .security import require_ops_token
from .state import state
from .tryon import apply_look
from .vision import TONES, NoFaceError, decode_image, estimate_skin_tone, rgb_hex_to_lab

router = APIRouter(prefix="/api/v1", tags=["v1"])
Status = Literal["live", "provisional", "planned"]

COLOUR_PREFS = ["warm", "cool", "neutral", "nude", "pink", "coral", "berry", "bold", "natural"]
CANDIDATE_LIMIT, MAX_PER_BRAND = 100, 2     # chosen from the measurements in evaluate.py (sections E and F)
NOT_MEDICAL = "Informational only, not medical advice. See a dermatologist for persistent or severe skin problems."


# ---------------- schemas ----------------
class Options(BaseModel):
    skin_types: list[str]
    concerns: list[str]
    colour_preferences: list[str]
    tones: list[str]
    skincare_categories: list[str]
    makeup_brands: list[str]
    undertones: list[str] = []


class FaceResult(BaseModel):
    hints: dict | None = None      # low-confidence photo hints (redness, shine, blemishes); forward to /skin/assess
    depth: str | None = None       # light / medium / deep
    undertone: str | None = None   # warm / neutral / cool (estimate, judged against shades of similar depth)
    undertone_basis: str = ""
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
    undertone: str | None = Field(None, description="warm / neutral / cool; overrides the measured undertone")
    sensitive: bool = False
    max_price_per_item: float | None = Field(None, ge=0)
    total_budget: float | None = Field(None, ge=0, description="Cap on the routine's total (priced products)")
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


class PaletteItem(BaseModel):
    name: str
    hex: str
    score: float
    intensity: Literal["soft", "medium", "bold"]
    why: str


class ProductMatch(BaseModel):
    product_uid: str
    name: str
    brand: str
    source: str
    url: str | None
    matched_on: list[str]
    basis: str


class MakeupSection(BaseModel):
    status: Status
    items: list[ShadeItem] = []                 # foundation: closest real shades
    palette: list[PaletteItem] = []             # lips / blush / eyes: colour-harmony shades
    products: list[ProductMatch] = []           # lips / blush / eyes: catalog products matched by name colour word
    note: str = ""


class MakeupLook(BaseModel):
    depth: str | None = None
    undertone: str | None = None
    undertone_basis: str = ""
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
    category: str | None = None
    product_uid: str | None = None
    price: float | None = None               # None when the source has no price (e.g. Open Beauty Facts)
    source: str | None = None
    spf: int | None = None
    note: str = ""


class Clash(BaseModel):
    severity: Literal["info", "caution"]
    slot: Literal["am", "pm"]
    products: list[str]
    message: str


class Routine(BaseModel):
    status: Status
    am: list[RoutineStep]
    pm: list[RoutineStep]
    warnings: list[str] = []                 # the caution-level clash messages, as plain strings
    clashes: list[Clash] = []
    notes: list[str] = []


class ConsultResponse(BaseModel):
    request_id: str | None = None            # send back with /feedback so a vote is tied to this consultation
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


def _check_inputs(skin_type, concerns, prefs=(), tone=None, undertone=None):
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
    if undertone is not None and undertone not in ml.UNDERTONES:
        raise HTTPException(422, f"undertone must be one of {ml.UNDERTONES}")


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


def _score(product_id: int, p_like: float, skin_type: str | None, concerns: list[str],
           fit: float | None = None, flags: list[str] | None = None) -> tuple[int, str, ScoreBreakdown]:
    """The AICosmetic score for one product (formula and weights: see score.py). `fit` and `flags` may be passed when
    the ranking stage already computed them (same definitions), which avoids re-scanning the ingredient text."""
    intel = state["intel"]
    counts = intel.counts(product_id, skin_type)
    share, basis = review_component(counts, intel.global_share, skin_type)
    if fit is None and flags is None:
        ingredients = state["rec"].products.at[product_id, "Ingredients"]
        fit = cx.concern_fit(concerns, ingredients)
        flags = cx.watch_flags(concerns, ingredients)
    elif not concerns:
        fit = None
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
                   makeup_brands=state["shades"].brands, undertones=ml.UNDERTONES)


@router.post("/face/analyze", response_model=FaceResult)
async def face_analyze(request: Request):
    """Raw image bytes in the request body. Processed in memory, never stored."""
    data = await request.body()
    if not data or len(data) > 5_000_000:
        raise HTTPException(413, "Send an image under 5 MB as the request body")
    try:
        result = estimate_skin_tone(decode_image(data))
    except NoFaceError as e:
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    profile = state["profiler"].profile(result["lab"])
    return {**result, "depth": profile.depth, "undertone": profile.undertone,
            "undertone_basis": profile.undertone_basis}


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
    score, verdict, breakdown = _score(int(r.Product_id), float(r.p_like), skin_type, concerns,
                                       float(r.concern_fit), list(r.ingredient_flags))
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


def _palette_section(kind: str, profile: ml.SkinProfile, prefs: list[str]) -> MakeupSection:
    palette = ml.suggest(kind, profile, prefs)
    products = ml.match_products(kind, palette, state["catalog"].df)
    note = "Rule-based colour-harmony palette (no learned model). "
    note += ("Products are matched by a colour word in their name, so coverage is small."
             if products else "No catalog product has a matching colour word in its name.")
    return MakeupSection(status="live", palette=[PaletteItem(**{k: v for k, v in p.items() if k != "families"})
                                                 for p in palette],
                         products=[ProductMatch(**m) for m in products], note=note)


def _makeup(req) -> MakeupLook:
    profiler = state["profiler"]
    profile = None
    if req.skin_hex:
        profile = profiler.profile(rgb_hex_to_lab(req.skin_hex), req.undertone)
    elif req.skin_tone:
        profile = profiler.from_tone_label(req.skin_tone, req.undertone)
    waiting = MakeupSection(status="planned", note="Upload a photo or choose a skin tone to get colour suggestions.")
    sections = {k: _palette_section(k, profile, req.colour_preferences) if profile else waiting
                for k in ("lips", "blush", "eyes")}
    return MakeupLook(depth=profile.depth if profile else None, undertone=profile.undertone if profile else None,
                      undertone_basis=profile.undertone_basis if profile else "", skin_hex=req.skin_hex,
                      tone=req.skin_tone, foundation=_foundation(req.skin_hex), **sections)


def _candidate(r, skin_type, concerns: list[str]) -> dict:
    """Lightweight scored candidate for the planner (no alternatives lookup)."""
    score, verdict, _ = _score(int(r.Product_id), float(r.p_like), skin_type, concerns,
                               float(r.concern_fit), list(r.ingredient_flags))
    return {"product_id": int(r.Product_id), "name": r.Product, "brand": r.Brand, "category": r.Category,
            "price": float(r.Price), "score": score, "verdict": verdict, "source": "sephora_reviews"}


def _plan(req, plain: dict[str, list[dict]], skin_type, concerns: list[str]):
    """Routine and shopping list for the consultation (logic lives in planner.py)."""
    rec = state["rec"]
    p_like = rec.predict_like(skin_type, req.skin_tone)

    def scorer(pid: int):
        score, verdict, _ = _score(pid, float(p_like[pid]), skin_type, concerns)
        return score, verdict

    inp = planner.PlanInput(skin_type=skin_type, concerns=concerns, sensitive=req.sensitive,
                            avoid=req.avoid_ingredients, budget=req.total_budget, max_price=req.max_price_per_item)
    out = planner.plan(inp, plain, rec, state["catalog"], scorer)
    clashes = [Clash(**c) for c in out["clashes"]]
    routine = Routine(status="live", am=[RoutineStep(**x) for x in out["am"]], pm=[RoutineStep(**x) for x in out["pm"]],
                      warnings=[c.message for c in clashes if c.severity == "caution"], clashes=clashes,
                      notes=out["notes"])
    return routine, out["shopping"]


@router.post("/consult", response_model=ConsultResponse)
def consult(req: ConsultRequest):
    """One call that returns the whole consultation: profile, skincare, makeup, routine and shopping list."""
    _check_inputs(req.skin_type, req.concerns, req.colour_preferences, req.skin_tone, req.undertone)
    rec = state["rec"]
    profile_result = _assess(req.skin_type, req.concerns, req.answers, req.photo_hints)
    skin_type = profile_result.skin_type
    active_concerns = [c.concern for c in profile_result.concerns]

    skincare: dict[str, list[SkincareItem]] = {}
    plain: dict[str, list[dict]] = {}
    for category in ("Cleanser", "Treatment", "Moisturizer"):
        profile = UserProfile(skin_type=skin_type, skin_tone=req.skin_tone, sensitive=req.sensitive,
                              concerns=active_concerns, category=category, max_price=req.max_price_per_item,
                              avoid_ingredients=req.avoid_ingredients)
        # a wide pool so budgets can be met (candidate cut-off 100 loses nothing, see evaluate.py), and the picks
        # shown to the user get the brand-diversity re-rank
        pool = rec.recommend(profile, k=30, explain=False, candidate_limit=CANDIDATE_LIMIT)
        metrics.CANDIDATES.observe(pool.attrs["trace"]["candidates"])
        out = pool.reset_index()
        shown = rec.recommend(profile, k=req.per_category, candidate_limit=CANDIDATE_LIMIT,
                              max_per_brand=MAX_PER_BRAND).reset_index()
        skincare[category] = [_skincare_item(r, skin_type, active_concerns) for r in shown.itertuples()]
        plain[category] = [_candidate(r, skin_type, active_concerns) for r in out.itertuples()]
    routine, shopping = _plan(req, plain, skin_type, active_concerns)
    shown = [{"product_id": i.product_id, "score": i.aicosmetic_score,
              "p_like": i.breakdown.model_like_probability, "verdict": i.verdict}
             for items in skincare.values() for i in items]
    for v in (i.verdict for items in skincare.values() for i in items):
        metrics.VERDICTS.labels(v).inc()
    if shopping["within_budget"] is False:
        metrics.BUDGET_MISSES.inc()
    request_id = state["db"].log_event(state["meta"].get("version", "unversioned"), skin_type, active_concerns,
                                       req.total_budget, shown)
    return ConsultResponse(request_id=request_id, profile=profile_result, skincare=skincare, skincare_status="live",
                           makeup=_makeup(req), routine=routine, shopping_list=shopping)


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


HEX = r"^#?[0-9a-fA-F]{6}$"


@router.post("/makeup/tryon", responses={200: {"content": {"image/jpeg": {}}}})
async def makeup_tryon(request: Request,
                       foundation_hex: str | None = Query(None, pattern=HEX),
                       lips_hex: str | None = Query(None, pattern=HEX),
                       blush_hex: str | None = Query(None, pattern=HEX),
                       strength: float = Query(0.8, ge=0.2, le=1.0)):
    """Virtual try-on: raw image bytes in, a JPEG out with the chosen colours applied (never stored).
    `X-Applied` lists what could be applied (lips are skipped if no mouth is found). No eyeshadow."""
    if not (foundation_hex or lips_hex or blush_hex):
        raise HTTPException(422, "Give at least one of foundation_hex, lips_hex, blush_hex")
    data = await request.body()
    if not data or len(data) > 5_000_000:
        raise HTTPException(413, "Send an image under 5 MB as the request body")
    try:
        img, applied, _ = apply_look(decode_image(data), foundation_hex, lips_hex, blush_hex, strength)
    except NoFaceError as e:
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=jpg.tobytes(), media_type="image/jpeg", headers={"X-Applied": ",".join(applied)})


@router.get("/makeup/web", response_model=WebSearchResponse)
def makeup_web(kind: Literal["lips", "blush", "eyes", "foundation"],
               shade: str = Query("", max_length=30, pattern=r"^[A-Za-z ]*$"),
               undertone: str | None = None, tone: str | None = None):
    """Live web results for real products in a suggested shade (needs TAVILY_API_KEY)."""
    _check_inputs(None, [], tone=tone, undertone=undertone)
    noun = {"lips": "lipstick", "blush": "blush", "eyes": "eyeshadow", "foundation": "foundation"}[kind]
    query = " ".join(x for x in (f"best {shade.strip()} {noun}", f"for {undertone} undertone" if undertone else "",
                                 f"{tone} skin tone" if tone else "") if x)
    web = state["web"]
    if not web.enabled:
        return WebSearchResponse(status="disabled", query=query, note="Set TAVILY_API_KEY to enable live web search.")
    try:
        found = web.search(query)
    except RuntimeError as e:
        return WebSearchResponse(status="error", query=query, note=str(e))
    return WebSearchResponse(status="live", query=query, results=[WebResult(**x) for x in found],
                             note="Third-party web content, shown as-is. Not used by the models.")


class SwapItem(BaseModel):
    for_product_uid: str
    product_uid: str
    product_id: int
    name: str
    brand: str
    price: float
    saving: float
    similarity: float
    aicosmetic_score: int
    verdict: Literal["Recommended", "Okay", "Caution"]
    note: str


class DupeResponse(BaseModel):
    product_id: int
    swaps: list[SwapItem]
    note: str


@router.get("/products/{product_id}/dupes", response_model=DupeResponse)
def product_dupes(product_id: int = Path(..., ge=0), skin_type: str | None = None,
                  min_saving: float = Query(rt.MIN_SAVING, ge=0),
                  min_similarity: float = Query(rt.MIN_SIMILARITY, ge=0, le=1)):
    """Cheaper products of the same category with a similar ingredient list that score nearly as well."""
    products = state["rec"].products
    if product_id not in products.index:
        raise HTTPException(404, "Unknown product")
    _check_inputs(skin_type, [])
    p_like = state["rec"].predict_like(skin_type)

    def scorer(pid: int):
        score, verdict, _ = _score(pid, float(p_like[pid]), skin_type, [])
        return score, verdict

    uid = f"sephora_reviews:{product_id}"
    score, _ = scorer(product_id)
    similar = state["catalog"].similar(uid, k=20, same_category=True, other_source=False, source="sephora_reviews")
    similar = similar[similar["similarity"] >= min_similarity]
    item = {"product_uid": uid, "price": float(products.at[product_id, "Price"]), "score": score,
            "category": products.at[product_id, "Category"]}
    saving = max(min_saving, 0)
    found = [s for s in rt.find_swaps(item, similar, scorer, {uid}, [], None, k=5) if s["saving"] >= saving]
    note = ("Similar ingredient list is not a guarantee of the same results." if found else
            "No cheaper product with a similar ingredient list was found; close matches are rare in this catalog.")
    return DupeResponse(product_id=product_id, swaps=[SwapItem(**s) for s in found], note=note)


class RoutineCheckRequest(BaseModel):
    am: list[str] = Field([], description="product_uid values, e.g. sephora_reviews:12 or open_beauty_facts:3600...")
    pm: list[str] = []


class RoutineCheckResponse(BaseModel):
    clashes: list[Clash]
    notes: list[str]


@router.post("/routine/check", response_model=RoutineCheckResponse)
def routine_check(req: RoutineCheckRequest):
    """Check a routine you already use for ingredient clashes. Products come from /catalog/search (`product_uid`)."""
    df = state["catalog"].df
    clashes = []
    for slot, uids in (("am", req.am), ("pm", req.pm)):
        items = []
        for uid in uids:
            row = df[df["product_uid"] == uid]
            if row.empty:
                raise HTTPException(404, f"Unknown product '{uid}'")
            r = row.iloc[0]
            items.append({"name": r["name"], "category": r["category"], "ingredients": r["ingredients"]})
        clashes += [Clash(**c) for c in rt.find_clashes(slot, items)]
    notes = [] if clashes else ["No known clashes among these products' main active ingredients."]
    notes.append("Rule-based checks on listed ingredients; concentrations are unknown. Not medical advice.")
    return RoutineCheckResponse(clashes=clashes, notes=notes)


class FeedbackRequest(BaseModel):
    product_id: int = Field(..., ge=0)
    helpful: bool
    request_id: str | None = Field(None, max_length=36, description="`request_id` from /consult")


class FeedbackResponse(BaseModel):
    ok: bool
    updated: bool = False


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Thumbs up / down on a recommended product. Stored without any personal data; used to retrain."""
    if req.product_id not in state["rec"].products.index:
        raise HTTPException(404, "Unknown product")
    try:
        result = state["db"].add_feedback(req.product_id, req.helpful, req.request_id)
    except KeyError as e:
        raise HTTPException(404, "Unknown request_id (it may predate a database reset)") from e
    metrics.FEEDBACK.labels(str(req.helpful).lower()).inc()
    return FeedbackResponse(ok=True, updated=result["updated"])


# ---------------- operations (read-only aggregates; no personal data) ----------------
@router.get("/ops/stats", dependencies=[Depends(require_ops_token)])
def ops_stats():
    """Event and feedback counts, the helpful rate, and the model being served."""
    c = state["db"].counts()
    rate = round(c["helpful"] / c["feedback"], 3) if c["feedback"] else None
    return {**c, "helpful_rate": rate, "database": state["db"].dialect,
            "model_version": state["meta"].get("version", "unversioned")}


@router.get("/ops/drift", dependencies=[Depends(require_ops_token)])
def ops_drift():
    """PSI of recent vs older recommendation events (model probability and skin-type mix)."""
    result = compute_drift(state["db"].recent_events(2000))
    for name, f in result["features"].items():
        metrics.DRIFT_PSI.labels(name).set(f["psi"])
    return result
