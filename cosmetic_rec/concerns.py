"""Skin concern knowledge and questionnaire scoring (rule-based, transparent, not a diagnosis).

Ingredient guidance is general cosmetic-science folklore turned into a small lookup table. It is a ranking nudge,
not clinical advice, and it has not been validated against outcomes (this project has no outcome data).
"""
from dataclasses import dataclass, field

CONCERNS = ["acne", "dryness", "oiliness", "sensitivity", "dark_spots", "redness", "fine_lines", "dullness"]
SKIN_TYPES = ["Combination", "Dry", "Normal", "Oily"]

# concern -> [(display name, [substrings matched against the normalised ingredient list])]
HELPFUL: dict[str, list[tuple[str, list[str]]]] = {
    "acne": [("salicylic acid", ["salicylic acid"]), ("niacinamide", ["niacinamide"]),
             ("benzoyl peroxide", ["benzoyl peroxide"]), ("azelaic acid", ["azelaic acid"])],
    "dryness": [("ceramides", ["ceramide"]), ("hyaluronic acid", ["hyaluronic acid", "sodium hyaluronate"]),
                ("glycerin", ["glycerin"]), ("squalane", ["squalane"])],
    "oiliness": [("niacinamide", ["niacinamide"]), ("salicylic acid", ["salicylic acid"]),
                 ("clay", ["kaolin", "bentonite"]), ("zinc", ["zinc pca", "zinc oxide"])],
    "sensitivity": [("centella", ["centella", "madecassoside"]), ("panthenol", ["panthenol"]),
                    ("allantoin", ["allantoin"]), ("ceramides", ["ceramide"])],
    "dark_spots": [("vitamin C", ["ascorbic acid", "ascorbyl", "ascorbate"]), ("niacinamide", ["niacinamide"]),
                   ("azelaic acid", ["azelaic acid"]), ("tranexamic acid", ["tranexamic acid"])],
    "redness": [("centella", ["centella", "madecassoside"]), ("azelaic acid", ["azelaic acid"]),
                ("green tea", ["camellia sinensis"]), ("allantoin", ["allantoin"])],
    "fine_lines": [("retinol", ["retinol", "retinal", "retinyl"]), ("peptides", ["peptide"]),
                   ("vitamin C", ["ascorbic acid", "ascorbyl", "ascorbate"]), ("hyaluronic acid", ["hyaluronic acid"])],
    "dullness": [("vitamin C", ["ascorbic acid", "ascorbyl", "ascorbate"]),
                 ("glycolic / lactic acid", ["glycolic acid", "lactic acid"]), ("niacinamide", ["niacinamide"])],
}
# concern -> ingredients that are often reported as aggravating for that concern (evidence is mixed; shown as "watch")
WATCH: dict[str, list[tuple[str, list[str]]]] = {
    "sensitivity": [("fragrance", ["fragrance", "limonene", "linalool"]), ("denatured alcohol", ["alcohol denat"]),
                    ("essential oils", ["lavandula", "citrus", "mentha"])],
    "redness": [("fragrance", ["fragrance"]), ("denatured alcohol", ["alcohol denat"]), ("menthol", ["menthol"])],
    "dryness": [("denatured alcohol", ["alcohol denat"]), ("sulfate cleansers", ["sodium lauryl sulfate"])],
    "acne": [("coconut oil", ["cocos nucifera oil"]), ("isopropyl myristate", ["isopropyl myristate"])],
}
TIPS = {
    "acne": "Introduce one active at a time. See a dermatologist for painful, cystic or scarring acne.",
    "dark_spots": "Daily sunscreen matters as much as the active; marks darken with sun exposure.",
    "fine_lines": "Retinoids work slowly (months) and can irritate; start a few nights a week and use sunscreen.",
    "sensitivity": "Patch test new products on the jaw or inner arm for a few days.",
    "redness": "Persistent facial redness can have medical causes (such as rosacea); worth a professional opinion.",
    "dryness": "Apply moisturiser to damp skin and avoid very hot water.",
    "oiliness": "Over-cleansing can increase oil; two gentle washes a day is plenty.",
    "dullness": "Exfoliating acids help, but start slowly and always follow with sunscreen.",
}
REASONS = {c: c.replace("_", " ") for c in CONCERNS}

# ---------------- questionnaire ----------------
# Each option adds votes to skin types and scores (0-1) to concerns.
QUESTIONS: list[dict] = [
    {"id": "midday", "text": "By midday, with no product on, how does your skin feel?", "options": [
        {"value": "tight", "label": "Tight, rough or flaky", "types": {"Dry": 2}, "concerns": {"dryness": 0.7}},
        {"value": "oily_all", "label": "Shiny across the whole face", "types": {"Oily": 2},
         "concerns": {"oiliness": 0.7}},
        {"value": "oily_tzone", "label": "Shiny on forehead and nose only", "types": {"Combination": 2},
         "concerns": {"oiliness": 0.3}},
        {"value": "comfortable", "label": "Comfortable, neither tight nor shiny", "types": {"Normal": 2},
         "concerns": {}}]},
    {"id": "pores", "text": "How visible are your pores?", "options": [
        {"value": "large", "label": "Large and visible on most of my face", "types": {"Oily": 1},
         "concerns": {"oiliness": 0.3}},
        {"value": "nose", "label": "Visible on nose and cheeks only", "types": {"Combination": 1}, "concerns": {}},
        {"value": "small", "label": "Barely visible", "types": {"Dry": 0.5, "Normal": 0.5}, "concerns": {}}]},
    {"id": "breakouts", "text": "How often do you get breakouts?", "options": [
        {"value": "never", "label": "Rarely or never", "types": {}, "concerns": {}},
        {"value": "sometimes", "label": "A few times a month", "types": {}, "concerns": {"acne": 0.4}},
        {"value": "often", "label": "Most weeks", "types": {}, "concerns": {"acne": 0.7}},
        {"value": "constant", "label": "Constantly, or painful and deep", "types": {}, "concerns": {"acne": 1.0}}]},
    {"id": "reaction", "text": "Do new products make your skin sting, burn or turn red?", "options": [
        {"value": "never", "label": "Almost never", "types": {}, "concerns": {}},
        {"value": "sometimes", "label": "Sometimes", "types": {}, "concerns": {"sensitivity": 0.5}},
        {"value": "often", "label": "Often", "types": {}, "concerns": {"sensitivity": 0.9}}]},
    {"id": "redness", "text": "Do you have visible redness?", "options": [
        {"value": "none", "label": "No", "types": {}, "concerns": {}},
        {"value": "some", "label": "Some, on the cheeks or around the nose", "types": {}, "concerns": {"redness": 0.5}},
        {"value": "persistent", "label": "Persistent redness most days", "types": {}, "concerns": {"redness": 0.9}}]},
    {"id": "marks", "text": "Do you have dark marks from breakouts or sun?", "options": [
        {"value": "none", "label": "No", "types": {}, "concerns": {}},
        {"value": "some", "label": "A few", "types": {}, "concerns": {"dark_spots": 0.5}},
        {"value": "many", "label": "Many or large patches", "types": {}, "concerns": {"dark_spots": 0.9}}]},
    {"id": "lines", "text": "Do you notice fine lines or loss of firmness?", "options": [
        {"value": "none", "label": "No", "types": {}, "concerns": {}},
        {"value": "few", "label": "A few fine lines", "types": {}, "concerns": {"fine_lines": 0.5}},
        {"value": "noticeable", "label": "Noticeable lines", "types": {}, "concerns": {"fine_lines": 0.9}}]},
    {"id": "dull", "text": "Does your skin look dull or tired?", "options": [
        {"value": "no", "label": "No", "types": {}, "concerns": {}},
        {"value": "sometimes", "label": "Sometimes", "types": {}, "concerns": {"dullness": 0.5}},
        {"value": "often", "label": "Most of the time", "types": {}, "concerns": {"dullness": 0.9}}]},
]
_OPTIONS = {(q["id"], o["value"]): o for q in QUESTIONS for o in q["options"]}
LIKELY, POSSIBLE = 0.6, 0.3
PHOTO_CAP = 0.35          # photo hints alone can never push a concern past "possible"


@dataclass
class Concern:
    concern: str
    level: str                      # likely | possible | selected
    score: float
    reasons: list[str] = field(default_factory=list)


def public_questionnaire() -> list[dict]:
    """Questions as the frontend should show them (no scoring internals)."""
    return [{"id": q["id"], "text": q["text"],
             "options": [{"value": o["value"], "label": o["label"]} for o in q["options"]]} for q in QUESTIONS]


def validate_answers(answers: dict[str, str]) -> None:
    for qid, value in answers.items():
        if (qid, value) not in _OPTIONS:
            raise ValueError(f"unknown answer '{value}' for question '{qid}'")


def _photo_votes(hints: dict | None) -> dict[str, tuple[float, str]]:
    """concern -> (score contribution, reason). Hints are low-confidence and capped."""
    if not hints:
        return {}
    levels = {"low": 0.0, "moderate": 0.6, "high": 1.0}
    votes = {}
    for key, concern, text in (("redness", "redness", "photo shows more redness than usual for your skin tone"),
                               ("shine", "oiliness", "photo shows noticeable shine on the forehead and nose"),
                               ("blemishes", "acne", "photo shows several small blemish-like spots")):
        lvl = levels.get((hints.get(key) or {}).get("level", "low"), 0.0)
        if lvl > 0:
            votes[concern] = (round(PHOTO_CAP * lvl, 2), f"{text} (photo hint, low confidence)")
    return votes


def assess(answers: dict[str, str] | None = None, selected: list[str] | None = None, photo_hints: dict | None = None,
           skin_type: str | None = None) -> tuple[str | None, str, list[Concern]]:
    """Returns (skin_type, how it was decided, concerns sorted by score)."""
    answers, selected = answers or {}, selected or []
    validate_answers(answers)
    type_votes: dict[str, float] = {}
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for qid, value in answers.items():
        opt = _OPTIONS[(qid, value)]
        for t, v in opt["types"].items():
            type_votes[t] = type_votes.get(t, 0) + v
        for c, v in opt["concerns"].items():
            scores[c] = max(scores.get(c, 0), v)
            reasons.setdefault(c, []).append(f"you answered \"{opt['label']}\"")
    for c, (v, text) in _photo_votes(photo_hints).items():
        scores[c] = min(1.0, scores.get(c, 0) + v)
        reasons.setdefault(c, []).append(text)

    if skin_type in SKIN_TYPES:
        decided, source = skin_type, "you told us"
    elif type_votes and max(type_votes.values()) >= 2:
        decided, source = max(type_votes, key=type_votes.get), "questionnaire"
    else:
        decided, source = None, "not enough information"

    out = []
    for c in CONCERNS:
        s = scores.get(c, 0.0)
        if c in selected:
            out.append(Concern(c, "selected", max(s, 1.0), ["you selected this concern"] + reasons.get(c, [])))
        elif s >= LIKELY:
            out.append(Concern(c, "likely", round(s, 2), reasons[c]))
        elif s >= POSSIBLE:
            out.append(Concern(c, "possible", round(s, 2), reasons[c]))
    out.sort(key=lambda x: -x.score)
    return decided, source, out


def helpful_names(concern: str) -> list[str]:
    return [name for name, _ in HELPFUL.get(concern, [])]


def watch_names(concern: str) -> list[str]:
    return [name for name, _ in WATCH.get(concern, [])]


def helpful_hits(concern: str, ingredients: str) -> list[str]:
    """Names of helpful ingredients for `concern` found in a normalised ingredient string."""
    text = ingredients.lower()
    return [name for name, subs in HELPFUL.get(concern, []) if any(sub in text for sub in subs)]


def watch_hits(concern: str, ingredients: str) -> list[str]:
    text = ingredients.lower()
    return [name for name, subs in WATCH.get(concern, []) if any(sub in text for sub in subs)]


def concern_fit(concerns: list[str], ingredients: str) -> float | None:
    """Mean over the user's concerns of how well the ingredient list addresses each (0-1). None without concerns."""
    if not concerns:
        return None
    return sum(min(len(helpful_hits(c, ingredients)), 2) / 2 for c in concerns) / len(concerns)


def watch_flags(concerns: list[str], ingredients: str) -> list[str]:
    return sorted({f for c in concerns for f in watch_hits(c, ingredients)})


def watch_substrings(concerns: list[str]) -> list[str]:
    """Every substring that marks an ingredient as 'often aggravating' for any of the concerns."""
    return sorted({sub for c in concerns for _, subs in WATCH.get(c, []) for sub in subs})
