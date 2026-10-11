"""Drift monitoring from the recommendation event log.

Two views, both using the Population Stability Index (PSI):
  prediction drift   model probabilities of the products shown: newest window vs the older reference window
  input drift        skin-type mix of requests: newest window vs the older reference window
PSI = sum((cur% - ref%) * ln(cur% / ref%)) over bins.
Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant.
Limits: it compares the log with itself (there is no labelled production traffic to compare with), it needs enough
events, and a real shift in who uses the site is also "drift" even when the model is fine.
"""
import numpy as np

MIN_PER_WINDOW = 30
EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI between two aligned count vectors (or share vectors); empty bins are smoothed."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "significant"


def _continuous_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                    # reference has (almost) no spread
        return 0.0 if np.allclose(ref.mean(), cur.mean(), atol=0.02) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(np.histogram(ref, edges)[0], np.histogram(cur, edges)[0])


def _categorical_psi(ref: list, cur: list) -> float:
    cats = sorted({*ref, *cur}, key=str)
    return psi(np.array([ref.count(c) for c in cats]), np.array([cur.count(c) for c in cats]))


def compute_drift(events: list[dict], current_share: float = 0.5) -> dict:
    """events: newest first (as Database.recent_events returns them), each {skin_type, shown: [{p_like, ...}]}."""
    n = len(events)
    n_cur = int(n * current_share)
    cur, ref = events[:n_cur], events[n_cur:]
    if len(cur) < MIN_PER_WINDOW or len(ref) < MIN_PER_WINDOW:
        return {"status": "insufficient_data", "events": n, "needed_per_window": MIN_PER_WINDOW, "features": {}}

    def p_likes(evs):
        return np.array([s["p_like"] for e in evs for s in e["shown"]], float)

    features = {
        "model_probability": _continuous_psi(p_likes(ref), p_likes(cur)),
        "skin_type": _categorical_psi([e["skin_type"] or "unknown" for e in ref],
                                      [e["skin_type"] or "unknown" for e in cur]),
    }
    out = {name: {"psi": round(v, 4), "status": status(v)} for name, v in features.items()}
    worst = max(features.values())
    return {"status": status(worst), "events": n, "reference_events": len(ref), "current_events": len(cur),
            "features": out}
