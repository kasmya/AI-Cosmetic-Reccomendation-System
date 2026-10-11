"""Review intelligence: a sentiment model, aspect themes and representative quotes per product.

Sentiment: TF-IDF (1-2 grams) + logistic regression predicting "rated 4-5 stars" from the review text.
Themes: a small keyword lexicon of skincare aspects; polarity of an aspect = mean model sentiment of the
sentences that mention it, judged relative to all products (about 75% of reviews are positive, so absolute
thresholds would call everything "praised"). Quotes: the most clearly positive / critical short sentences.
The sentiment model is trained on review-level labels and applied to single sentences, which is a domain
shift: aspect polarities are estimates, not ground truth.
"""
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ASPECTS: dict[str, str] = {
    "hydration": r"hydrat|moistur|dewy|plump|dries|drying|dried|dry skin|dryness|tight|flak",
    "breakouts": r"break ?outs?|acne|pimple|zit|blemish|clog|cystic|whitehead|blackhead",
    "irritation": r"irritat|burn|sting|redness|rash|itch|reaction|allerg|inflam",
    "scent": r"scent|smell|fragran|perfum|odou?r",
    "texture": r"sticky|greasy|oily|thick|lightweight|light weight|texture|silky|smooth|creamy|tacky|heavy|"
               r"residue|film",
    "results": r"results?|worked|works|improv|cleared|clearer|glow|brighten|firmer|wrinkle|fine lines|dark spots|scar",
    "gentleness": r"gentle|soothing|calming|mild|non-?irritating|sensitive skin",
    "value": r"price|expensive|worth|pricey|afford|cheap|value|tiny|small amount",
    "packaging": r"pump|bottle|packag|container|leak|jar|tube|dispens",
}
_ASPECT_RE = {a: re.compile(p, re.I) for a, p in ASPECTS.items()}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MARGIN = 0.07     # an aspect is praised/criticised if it scores this far above/below how ALL products score on it


def clean_text(text: str) -> str:
    text = str(text).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(clean_text(text)) if len(s.strip()) >= 25]


def dedupe_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Drop repeat rows of the same user's identical review of the same product."""
    return reviews.drop_duplicates(["User_id", "Product_id", "Review"])


def _prep(text: str) -> str:
    """Named (not a lambda) so the fitted pipeline can be pickled."""
    return clean_text(text).lower()


def make_sentiment_pipeline(seed: int = 0):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, preprocessor=_prep),
        LogisticRegression(C=1.0, max_iter=1000, random_state=seed))


@dataclass
class Theme:
    aspect: str
    mentions: int
    sentiment: float
    verdict: str


class ReviewStats:
    """Counts of positive (4-5 star) reviews per product, overall and per reviewer skin type. Cheap to fit."""

    def fit(self, reviews: pd.DataFrame):
        self.reviews = dedupe_reviews(reviews)[["User_id", "Product_id", "Rating_Stars", "Skin_Type"]]
        self.global_share = float((self.reviews["Rating_Stars"] >= 4).mean())
        self._by_product = {pid: g for pid, g in self.reviews.groupby("Product_id")}
        return self

    def _table(self) -> dict:
        """{product_id: {"n", "pos", "skin": {skin_type: (n, pos)}}}, built once on first use (not persisted)."""
        table = self.__dict__.get("_count_table")
        if table is None:
            table = {}
            r = self.reviews
            pos = (r["Rating_Stars"] >= 4).to_numpy()
            for pid, skin, is_pos in zip(r["Product_id"], r["Skin_Type"], pos, strict=True):
                e = table.setdefault(int(pid), {"n": 0, "pos": 0, "skin": {}})
                e["n"] += 1
                e["pos"] += int(is_pos)
                if isinstance(skin, str):
                    n, p = e["skin"].get(skin, (0, 0))
                    e["skin"][skin] = (n + 1, p + int(is_pos))
            self.__dict__["_count_table"] = table
        return table

    def counts(self, product_id: int, skin_type: str | None = None) -> dict:
        e = self._table().get(int(product_id), {"n": 0, "pos": 0, "skin": {}})
        n_skin, pos_skin = e["skin"].get(skin_type, (0, 0)) if skin_type else (0, 0)
        return {"n": e["n"], "pos": e["pos"], "n_skin": n_skin, "pos_skin": pos_skin}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_count_table", None)
        return state


class ReviewIntel(ReviewStats):
    """ReviewStats plus a sentiment model, aspect themes and quotes."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, reviews: pd.DataFrame):
        ReviewStats.fit(self, reviews)
        r = dedupe_reviews(reviews).reset_index(drop=True)
        self.sentiment = make_sentiment_pipeline(self.seed).fit(r["Review"], (r["Rating_Stars"] >= 4).astype(int))
        rows = []
        for i, (text, stars) in enumerate(zip(r["Review"], r["Rating_Stars"], strict=True)):
            for s in sentences(text):
                rows.append((r.at[i, "Product_id"], stars, s))
        self.sent = pd.DataFrame(rows, columns=["Product_id", "stars", "text"])
        self.sent["p_pos"] = self.sentiment.predict_proba(self.sent["text"])[:, 1]
        for a, rx in _ASPECT_RE.items():
            self.sent[a] = self.sent["text"].map(lambda t, rx=rx: bool(rx.search(t)))
        self.aspect_mean = {a: float(self.sent.loc[self.sent[a], "p_pos"].mean()) for a in ASPECTS}
        return self

    # ---------- per-product ----------

    def themes(self, product_id: int, min_mentions: int = 3, top: int = 5) -> list[Theme]:
        s = self.sent[self.sent["Product_id"] == product_id]
        out = []
        for a in ASPECTS:
            m = s[s[a]]
            if len(m) >= min_mentions:
                mean = float(m["p_pos"].mean())
                delta = mean - self.aspect_mean[a]
                verdict = "praised" if delta >= MARGIN else "criticised" if delta <= -MARGIN else "typical"
                out.append(Theme(a, len(m), round(mean, 2), verdict))
        return sorted(out, key=lambda t: -t.mentions)[:top]

    def quotes(self, product_id: int, n_each: int = 1) -> list[dict]:
        """Most clearly positive / critical sentences that talk about a concrete aspect (generic ones are skipped)."""
        s = self.sent[(self.sent["Product_id"] == product_id) & self.sent["text"].str.len().between(40, 170)]
        s = s[s[list(ASPECTS)].any(axis=1)]
        pos = s[s["stars"] >= 4].nlargest(n_each, "p_pos")
        neg = s[s["stars"] <= 3].nsmallest(n_each, "p_pos")
        return ([{"text": t, "sentiment": "positive", "stars": int(st)}
                 for t, st in zip(pos["text"], pos["stars"], strict=True)]
                + [{"text": t, "sentiment": "critical", "stars": int(st)}
                   for t, st in zip(neg["text"], neg["stars"], strict=True)])
