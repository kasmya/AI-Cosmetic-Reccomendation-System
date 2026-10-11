"""Makeup look engine: depth and undertone from skin colour, colour-harmony palettes, name-based product matches.

depth       light / medium / deep from CIELAB lightness (fixed round thresholds).
undertone   warm / neutral / cool from the Lab hue angle, judged against shades of similar depth: lighter skin is
            naturally more yellow (hue and lightness correlate at 0.55 in the 621-shade dataset), so cut-offs are the
            lower and upper thirds of the hue distribution *within each depth band*. It is an estimate from one
            photo's pixel colour; lighting moves it. Users can override it.
palettes    hand-written colour-theory rules (no learned model, no ground truth): each shade has a fit score per
            undertone, tags matching the user's colour preferences, and an intensity. Hex values are adapted to depth.
products    lipsticks/blushes/eyeshadows from Open Beauty Facts whose NAME contains a colour word that matches the
            palette (multilingual keywords). Approximate and small: those products have no shade colours.
"""
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .vision import lab_to_hex, rgb_hex_to_lab

LIGHT_MIN, MEDIUM_MIN = 68.0, 52.0
UNDERTONES = ["warm", "neutral", "cool"]
DEPTHS = ["light", "medium", "deep"]
TONE_TO_DEPTH = {"Porcelain": "light", "Fair": "light", "Light": "light", "Medium": "medium", "Olive": "medium",
                 "Tan": "medium", "Deep": "deep", "Dark": "deep", "Ebony": "deep"}


def depth_of(lightness: float) -> str:
    return "light" if lightness >= LIGHT_MIN else "medium" if lightness >= MEDIUM_MIN else "deep"


def hue_angle(lab) -> float:
    return math.degrees(math.atan2(float(lab[2]), float(lab[1])))


@dataclass
class SkinProfile:
    depth: str
    undertone: str
    undertone_basis: str     # "measured" | "you told us" | "unknown"
    hue: float | None = None


class SkinProfiler:
    """Fits the within-depth hue cut-offs from the foundation shade dataset."""

    def __init__(self, shade_labs: np.ndarray):
        L, hue = shade_labs[:, 0], np.degrees(np.arctan2(shade_labs[:, 2], shade_labs[:, 1]))
        self.cutoffs = {}
        for name, lo, hi in (("light", LIGHT_MIN, 200), ("medium", MEDIUM_MIN, LIGHT_MIN), ("deep", -1, MEDIUM_MIN)):
            band = hue[(L >= lo) & (L < hi)]
            self.cutoffs[name] = (float(np.percentile(band, 33.3)), float(np.percentile(band, 66.7)))

    def profile(self, lab, undertone: str | None = None) -> SkinProfile:
        depth = depth_of(float(lab[0]))
        h = hue_angle(lab)
        if undertone in UNDERTONES:
            return SkinProfile(depth, undertone, "you told us", round(h, 1))
        cool_max, warm_min = self.cutoffs[depth]
        tone = "cool" if h <= cool_max else "warm" if h >= warm_min else "neutral"
        return SkinProfile(depth, tone, "measured", round(h, 1))

    def from_tone_label(self, tone: str | None, undertone: str | None = None) -> SkinProfile | None:
        """Depth only (no measured colour): used when the user picked a tone instead of uploading a photo."""
        if tone not in TONE_TO_DEPTH:
            return None
        u = undertone if undertone in UNDERTONES else "neutral"
        return SkinProfile(TONE_TO_DEPTH[tone], u, "you told us" if undertone in UNDERTONES else "unknown")


# ---------------- palettes ----------------
def _P(name: str, hx: str, fit: tuple, tags: str, intensity: int, families: str) -> dict:
    """One palette shade: base hex (medium depth), fit for (warm, neutral, cool), preference tags,
    intensity 0 soft / 1 medium / 2 bold, and colour-family words used for name matching."""
    return dict(name=name, hex=hx, fit=dict(zip(UNDERTONES, fit, strict=True)), tags=set(tags.split()),
                intensity=intensity, families=set(families.split()))



PALETTES: dict[str, list[dict]] = {
    "lips": [
        _P("Rosy nude", "#C98F86", (0.6, 1.0, 0.7), "nude natural pink neutral", 0, "nude rose pink"),
        _P("Warm caramel nude", "#B9806A", (1.0, 0.7, 0.3), "nude natural warm neutral", 0, "nude brown beige caramel"),
        _P("Mauve nude", "#A8737B", (0.3, 0.8, 0.9), "nude cool pink neutral", 0, "mauve nude rose"),
        _P("Soft pink", "#E49AA8", (0.5, 0.8, 0.9), "pink natural cool", 1, "pink rose"),
        _P("Peach", "#F0937A", (1.0, 0.7, 0.3), "coral warm natural", 0, "peach coral orange"),
        _P("Coral", "#E8735E", (1.0, 0.7, 0.2), "coral warm", 1, "coral orange"),
        _P("Terracotta", "#B5603F", (1.0, 0.6, 0.2), "warm nude bold", 1, "brown terracotta orange"),
        _P("Berry", "#A23B5E", (0.4, 0.8, 1.0), "berry cool bold pink", 2, "berry plum raspberry"),
        _P("Deep plum", "#6E2D4F", (0.3, 0.6, 1.0), "berry cool bold", 2, "plum berry burgundy"),
        _P("Brick red", "#A8412F", (1.0, 0.6, 0.2), "warm bold red", 2, "red brick"),
        _P("Classic blue-red", "#C0223B", (0.3, 0.8, 1.0), "cool bold red", 2, "red cherry"),
        _P("Orange red", "#D9452B", (1.0, 0.6, 0.2), "warm bold coral red", 2, "red orange coral"),
    ],
    "blush": [
        _P("Peach", "#F2A48C", (1.0, 0.8, 0.3), "peach coral warm natural", 0, "peach coral"),
        _P("Soft pink", "#E8A0B0", (0.4, 0.9, 0.9), "pink cool natural", 0, "pink rose"),
        _P("Rose", "#D2768A", (0.5, 1.0, 0.8), "pink berry neutral", 1, "rose pink"),
        _P("Coral", "#EE7F6A", (1.0, 0.7, 0.2), "coral warm", 1, "coral orange"),
        _P("Mauve", "#B77C94", (0.3, 0.8, 0.9), "cool nude pink", 1, "mauve plum"),
        _P("Bronze glow", "#C08060", (1.0, 0.7, 0.2), "warm natural nude bold", 1, "bronze brown gold"),
        _P("Berry rose", "#B04A6C", (0.3, 0.7, 1.0), "berry cool bold", 2, "berry plum raspberry"),
    ],
    "eyes": [
        _P("Champagne gold", "#D8B676", (0.9, 0.8, 0.4), "warm natural nude", 0, "gold champagne beige"),
        _P("Pink champagne", "#D9B8B0", (0.4, 0.9, 0.9), "cool nude pink natural", 0, "pink champagne rose"),
        _P("Soft brown", "#7B5B49", (0.8, 1.0, 0.8), "neutral natural nude", 0, "brown taupe"),
        _P("Taupe", "#8C7467", (0.7, 1.0, 0.7), "neutral natural nude", 0, "taupe brown"),
        _P("Warm bronze", "#A9764A", (1.0, 0.7, 0.3), "warm natural bronze", 1, "bronze brown gold"),
        _P("Mauve plum", "#8A5A78", (0.3, 0.7, 1.0), "cool berry pink", 1, "mauve plum purple"),
        _P("Copper", "#B5673A", (1.0, 0.6, 0.2), "warm bold coral", 2, "copper orange bronze"),
        _P("Olive green", "#707A4A", (1.0, 0.7, 0.3), "warm bold", 2, "green olive"),
        _P("Slate", "#4A4F5C", (0.4, 0.8, 1.0), "cool bold", 2, "grey black blue"),
        _P("Navy", "#2F3E66", (0.2, 0.6, 1.0), "cool bold", 2, "blue navy"),
        _P("Deep plum", "#5E3350", (0.3, 0.6, 1.0), "berry cool bold", 2, "plum purple berry"),
    ],
}
PREFERENCE_TAGS = {"warm", "cool", "neutral", "nude", "pink", "coral", "berry", "bold", "natural"}


def adapt_hex(hex_code: str, depth: str) -> str:
    """Shift a medium-depth swatch for lighter or deeper skin: lighter skin softer and brighter, deeper richer."""
    L, a, b = rgb_hex_to_lab(hex_code)
    if depth == "light":
        L, a, b = L + 6, a * 0.92, b * 0.92
    elif depth == "deep":
        L, a, b = L - 8, a * 1.12, b * 1.12
    return lab_to_hex([float(np.clip(L, 5, 95)), float(a), float(b)])


def suggest(kind: str, profile: SkinProfile, prefs: list[str] | None = None, k: int = 4) -> list[dict]:
    """Top `k` palette shades for lips / blush / eyes, best first, each with the reason it was chosen."""
    prefs = [p for p in (prefs or []) if p in PREFERENCE_TAGS]
    scored = []
    for item in PALETTES[kind]:
        fit = item["fit"][profile.undertone] if profile.undertone_basis != "unknown" else 0.7
        hits = sorted(item["tags"] & set(prefs))
        bonus = 0.6 * len(hits)
        if "bold" in prefs and item["intensity"] == 2:
            bonus += 0.4
        if ({"natural", "nude"} & set(prefs)) and item["intensity"] == 0:
            bonus += 0.4
        reasons = []
        if profile.undertone_basis != "unknown":
            reasons.append(f"flatters {profile.undertone} undertones" if fit >= 0.9 else
                           f"works with {profile.undertone} undertones" if fit >= 0.7 else
                           f"less ideal for {profile.undertone} undertones")
        if hits:
            reasons.append("matches your preference for " + ", ".join(hits))
        scored.append((fit + bonus, item, reasons))
    scored.sort(key=lambda t: (-t[0], t[1]["name"]))
    return [{"name": it["name"], "hex": adapt_hex(it["hex"], profile.depth), "score": round(sc, 2),
             "why": "; ".join(r) or "a versatile shade", "intensity": ["soft", "medium", "bold"][it["intensity"]],
             "families": sorted(it["families"])} for sc, it, r in scored[:k]]


# ---------------- name-based product matching ----------------
FAMILY_WORDS = {
    "pink": r"pink|\brosa\b|\brose|rosé|rosy",
    "nude": r"\bnude",
    "beige": r"beige|\bsand\b|\bsable\b|ivory|ivoire",
    "brown": r"brown|brun|marron|chocolate|mocha|caramel|cognac|terracotta|chestnut|cannelle",
    "coral": r"coral|corail",
    "peach": r"peach|p[eê]che|pesca|apricot|abricot",
    "orange": r"\borange",
    "red": r"\bred\b|rouge|rojo|\brot\b|cherry|cerise|brick",
    "berry": r"berry|baies|framboise|raspberry|cassis",
    "plum": r"plum|\bprune\b|burgundy|bordeaux|\bwine\b",
    "mauve": r"mauve|lilac|lilas",
    "purple": r"purple|violet|lavender",
    "bronze": r"bronze|copper|cuivr",
    "gold": r"\bgold|dor[eé]\b|champagne|\bor\b",
    "grey": r"\bgrey|\bgray|\bgris\b|silver|argent|taupe|charcoal",
    "black": r"black|noir|schwarz",
    "blue": r"\bblue|\bbleu|navy",
    "green": r"\bgreen|\bvert\b|olive|khaki",
    "champagne": r"champagne",
}
_FAMILY_RE = {f: re.compile(p, re.I) for f, p in FAMILY_WORDS.items()}
KIND_CATEGORIES = {"lips": ["Lipstick"], "blush": ["Blush"], "eyes": ["Eyeshadow"]}


def name_families(name: str) -> set[str]:
    return {f for f, rx in _FAMILY_RE.items() if rx.search(name or "")}


def match_products(kind: str, palette: list[dict], catalog: pd.DataFrame, k: int = 4) -> list[dict]:
    """Catalog products of the right category whose NAME shares a colour family with the palette (best first)."""
    wanted: dict[str, float] = {}
    for rank, shade in enumerate(palette):
        for fam in shade["families"]:
            wanted[fam] = max(wanted.get(fam, 0), 1.0 - 0.15 * rank)
    pool = catalog[catalog["category"].isin(KIND_CATEGORIES[kind])]
    out = []
    for r in pool.itertuples():
        fams = name_families(r.name) & wanted.keys()
        if fams:
            out.append((sum(wanted[f] for f in fams), r, sorted(fams)))
    out.sort(key=lambda t: (-t[0], t[1].name))
    return [{"product_uid": r.product_uid, "name": r.name, "brand": r.brand or "", "source": r.source,
             "url": r.url if isinstance(r.url, str) else None, "matched_on": fams,
             "basis": "colour word in product name"}
            for _, r, fams in out[:k]]
