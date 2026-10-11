"""Train the skincare model and save a versioned artifact.  Run: python -m cosmetic_rec.train"""
import json
import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import load_reviews
from .recommender import Recommender

log = logging.getLogger("cosmetic_rec.train")
ARTIFACT = Path(__file__).resolve().parent.parent / "models" / "skincare_model.joblib"
SEED = 42


def cv_auc(reviews, folds=5) -> tuple[float, float]:
    """Grouped-by-user cross-validated ROC-AUC for 'rated >= 4 stars' (no user in both train and test)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    base = Recommender(seed=SEED).fit(reviews)
    X = base._design(reviews["Skin_Type"], reviews["Product_id"].to_numpy(), reviews["Skin_Tone"])
    y = (reviews["Rating_Stars"] >= 4).astype(int).to_numpy()
    aucs = []
    for tr, te in GroupKFold(folds).split(X, y, reviews["User_id"]):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)
        aucs.append(roc_auc_score(y[te], m.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def train(path: Path = ARTIFACT) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    reviews = load_reviews()
    auc, std = cv_auc(reviews)
    model = Recommender(seed=SEED).fit(reviews)
    meta = {"model": "HistGradientBoosting + TF-IDF/SVD ingredient embeddings",
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_reviews": len(reviews), "n_products": len(model.products),
            "cv_auc": round(auc, 3), "cv_auc_std": round(std, 3), "sklearn": sklearn.__version__}
    path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)
    log.info("trained in %.1fs, saved %s", time.time() - t0, path)
    return meta


def load(path: Path = ARTIFACT) -> tuple[Recommender, dict]:
    """Load the saved artifact, or train one if it is missing."""
    if not path.exists():
        log.info("no artifact at %s, training", path)
        train(path)
    blob = joblib.load(path)
    return blob["model"], blob["meta"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(train(), indent=2))
