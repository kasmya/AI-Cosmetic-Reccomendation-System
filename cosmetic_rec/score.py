"""The AICosmetic score: a published, transparent 0-100 rating with a visible breakdown.

    components   model    calibrated P(user rates it 4-5 stars) from the ML model (personalised to skin type / tone)
                 reviews  Bayesian-smoothed share of positive (4-5 star) reviews, using reviewers who share the
                          user's skin type when there are at least MIN_SKIN_REVIEWS of them
                 concern  share of the user's concerns the product's ingredients address (only when concerns given)
    weights      without concerns: model 0.5, reviews 0.5      with concerns: model 0.4, reviews 0.4, concern 0.2
    penalty      4 points per distinct "watch" ingredient for the user's concerns (at most 2)
    score        100 * (weighted sum) - penalty, clipped to 0..100
    verdict      Recommended >= 80, Okay >= 70, otherwise Caution. Scores cluster around 75 because three quarters
                 of all reviews are positive, so the bands are relative to this catalog (roughly the top 20% /
                 middle 60% / bottom 20%).
    confidence   from the number of reviews behind the product: low < 15, medium < 30, high otherwise.
"""
WEIGHTS_BASE = {"model": 0.5, "reviews": 0.5}
WEIGHTS_CONCERN = {"model": 0.4, "reviews": 0.4, "concern": 0.2}
PENALTY_PER_FLAG, MAX_FLAGS = 4, 2
PRIOR_N, SKIN_PRIOR_N, MIN_SKIN_REVIEWS = 10, 5, 3
RECOMMENDED, OKAY = 80, 70


def smoothed_share(pos: int, n: int, prior: float, prior_n: float) -> float:
    return (pos + prior * prior_n) / (n + prior_n)


def review_component(counts: dict, global_share: float, skin_type: str | None = None) -> tuple[float, str]:
    """(smoothed positive share, which reviews it is based on)."""
    overall = smoothed_share(counts["pos"], counts["n"], global_share, PRIOR_N)
    if skin_type and counts["n_skin"] >= MIN_SKIN_REVIEWS:
        share = smoothed_share(counts["pos_skin"], counts["n_skin"], overall, SKIN_PRIOR_N)
        return share, f"{skin_type.lower()}-skin reviewers"
    return overall, "all reviewers"


def confidence(n_reviews: int) -> str:
    return "low" if n_reviews < 15 else "medium" if n_reviews < 30 else "high"


def verdict(score: int) -> str:
    return "Recommended" if score >= RECOMMENDED else "Okay" if score >= OKAY else "Caution"


def combine(p_like: float, review_share: float, concern_fit: float | None = None, n_flags: int = 0) -> dict:
    """The score and the exact contribution of each part (in points)."""
    weights = WEIGHTS_BASE if concern_fit is None else WEIGHTS_CONCERN
    parts = {"model": p_like, "reviews": review_share}
    if concern_fit is not None:
        parts["concern"] = concern_fit
    points = {k: round(100 * weights[k] * v, 1) for k, v in parts.items()}
    penalty = PENALTY_PER_FLAG * min(n_flags, MAX_FLAGS)
    score = int(round(min(100, max(0, sum(points.values()) - penalty))))
    return {"score": score, "verdict": verdict(score), "points": points, "weights": weights, "penalty": penalty}
