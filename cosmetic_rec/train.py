"""Train a challenger model, register it, and let the registry gate decide whether it becomes the champion.

    python -m cosmetic_rec.train                  train on the original reviews, promote if the gate passes
    python -m cosmetic_rec.train --no-promote     register only
    python -m cosmetic_rec.train --with-feedback  also learn from thumbs up/down stored in the database

The gold holdout (see registry.py) is never trained on. Reviews text/statistics for the review model use all reviews:
they are descriptive and are not what the gate measures.
"""
import argparse
import hashlib
import json
import logging
import time
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import registry
from .data import build_products, load_reviews
from .recommender import Recommender
from .reviews_nlp import ReviewIntel

log = logging.getLogger("cosmetic_rec.train")
CONFIG = {"hgb": Recommender.DEFAULT_HGB, "svd_dim": 20, "calibration": "sigmoid", "seed": 42}


def feedback_to_reviews(rows: list[dict], products: pd.DataFrame) -> pd.DataFrame:
    """Turn thumbs up/down into review-shaped rows (5 stars / 1 star). Each consultation is one pseudo-user with a
    negative id, so it can never collide with a real reviewer and calibration folds keep its votes together.
    Limitation: people only vote on what was shown, so this data carries position/selection bias."""
    out = []
    for i, r in enumerate(rows):
        if r["product_id"] not in products.index:
            continue
        key = r.get("event_id") or f"row-{i}"
        uid = -(int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16) + 1)
        out.append({"User_id": uid, "Product_id": r["product_id"], "Rating_Stars": 5 if r["helpful"] else 1,
                    "Skin_Type": r.get("skin_type"), "Skin_Tone": np.nan})
    return pd.DataFrame(out, columns=["User_id", "Product_id", "Rating_Stars", "Skin_Type", "Skin_Tone"])


def cv_auc(train: pd.DataFrame, config: dict, folds: int = 5) -> tuple[float, float]:
    """User-grouped cross-validated AUC on the training data (the holdout is evaluated separately by the gate)."""
    base = Recommender(seed=config["seed"], svd_dim=config["svd_dim"], hgb_params=config["hgb"]).fit(train)
    X = base._design(train["Skin_Type"], train["Product_id"].to_numpy(), train["Skin_Tone"])
    y = (train["Rating_Stars"] >= 4).astype(int).to_numpy()
    aucs = []
    for tr, te in GroupKFold(folds).split(X, y, train["User_id"]):
        m = HistGradientBoostingClassifier(**config["hgb"], random_state=config["seed"]).fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def train(feedback_rows: list[dict] | None = None, config: dict | None = None, promote: bool = True,
          root=registry.REGISTRY_DIR) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    config = {**CONFIG, **(config or {})}
    reviews = load_reviews()
    products = build_products(reviews)
    train_part, gold = registry.split_gold(reviews)
    extra = feedback_to_reviews(feedback_rows or [], products)
    fit_data = pd.concat([train_part, extra], ignore_index=True) if len(extra) else train_part

    model = Recommender(seed=config["seed"], svd_dim=config["svd_dim"], hgb_params=config["hgb"])
    model.fit(fit_data, products)
    intel = ReviewIntel(seed=config["seed"]).fit(reviews)
    auc, std = cv_auc(fit_data, config)
    from .evaluate import sentiment_cv
    sent = sentiment_cv(reviews, config["seed"])
    y = (gold["Rating_Stars"] >= 4).astype(int).to_numpy()
    holdout = registry.metrics_for(y, registry.predict_holdout(model, gold))

    version = registry.new_version_id(f"{len(fit_data)}-{len(extra)}-{json.dumps(config, sort_keys=True)}")
    meta = {"model": "HistGradientBoosting (Platt-calibrated) + TF-IDF/SVD ingredient embeddings",
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "n_reviews": len(reviews),
            "n_train_reviews": len(fit_data), "n_feedback_rows": int(len(extra)), "n_products": len(model.products),
            "cv_auc": round(auc, 3), "cv_auc_std": round(std, 3), "holdout": holdout,
            "sentiment_model": "TF-IDF (1-2 grams) + logistic regression",
            "sentiment_auc": round(sent["AUC"], 3), "sentiment_accuracy": round(sent["accuracy"], 3),
            "sklearn": sklearn.__version__}
    registry.register({"model": model, "intel": intel, "meta": meta}, holdout, config, version, root)
    decision = registry.evaluate_and_promote(version, gold, root=root) if promote else None
    log.info("trained %s in %.1fs, holdout AUC %.3f", version, time.time() - t0, holdout["auc"])
    return {"version": version, "holdout": holdout, "decision": decision, "feedback_rows_used": int(len(extra))}


def load_all(root=registry.REGISTRY_DIR) -> dict:
    """The champion's artifact. If no model has been registered yet, train and promote one first."""
    if registry.champion(root) is None:
        log.info("no champion in %s, training the first model", root)
        train(root=root)
    return registry.load_version(registry.champion(root), root)


def load(root=registry.REGISTRY_DIR) -> tuple[Recommender, dict]:
    blob = load_all(root)
    return blob["model"], blob["meta"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-promote", action="store_true", help="register the model but do not run the promotion gate")
    ap.add_argument("--with-feedback", action="store_true", help="include feedback rows from the database")
    args = ap.parse_args()
    rows = None
    if args.with_feedback:
        from .db import Database
        rows = Database().feedback_rows()
        print(f"using {len(rows)} feedback rows")
    out = train(rows, promote=not args.no_promote)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
