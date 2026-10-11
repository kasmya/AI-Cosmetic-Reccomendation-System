"""Model registry with champion / challenger promotion.

Layout (git-ignored, under models/registry/):
    versions/<version>/artifact.joblib   {"model", "intel", "meta"}
    versions/<version>/metrics.json      holdout metrics + training config
    champion.json                        the version the API serves
    history.jsonl                        every registration, promotion, rejection and rollback

The gold holdout is a fixed, hash-chosen 20% of the original review users that NO version is ever trained on, so
every version is judged on the same unseen people. The deployed champion is evaluated as it is (frozen), not retrained.
Promotion gate (all must hold, challenger vs champion on the gold holdout):
    AUC non-inferior      lower end of the 95% paired-bootstrap interval of (challenger - champion) AUC >= -0.01
    Brier not worse       challenger Brier <= champion Brier + 0.002
    calibration holds     challenger ECE <= champion ECE + 0.01
A challenger trained on newer data therefore replaces the champion unless it is measurably worse. `strictly_better` is
recorded when the whole interval is above zero. With a signal as weak as this one (AUC ~0.6) most differences are noise,
and the gate is written to be honest about that.
"""
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = Path(os.environ.get("COSMETIC_REGISTRY_DIR") or ROOT / "models" / "registry")
HOLDOUT_MOD, HOLDOUT_REMAINDER = 5, 0          # users with hash(User_id) % 5 == 0 form the gold holdout
AUC_TOLERANCE, BRIER_TOLERANCE, ECE_TOLERANCE = -0.01, 0.002, 0.01


# ---------------- gold holdout ----------------
def _bucket(user_id) -> int:
    return int(hashlib.sha256(str(int(user_id)).encode()).hexdigest(), 16) % HOLDOUT_MOD


def split_gold(reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training reviews, gold holdout reviews). Deterministic: the same users are held out every time."""
    is_gold = reviews["User_id"].map(_bucket) == HOLDOUT_REMAINDER
    return reviews[~is_gold], reviews[is_gold]


# ---------------- metrics ----------------
def ece(y, p, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def predict_holdout(model, holdout: pd.DataFrame) -> np.ndarray:
    X = model._design(holdout["Skin_Type"], holdout["Product_id"].to_numpy(), holdout["Skin_Tone"])
    return model.clf.predict_proba(X)[:, 1]


def metrics_for(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4),
            "ece": round(ece(y, p), 4), "n_reviews": int(len(y))}


def paired_auc_interval(y, p_new, p_old, users, n_boot: int = 400, seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap over users of AUC(new) - AUC(old): (mean difference, 2.5th, 97.5th percentile)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    by_user = {u: np.flatnonzero(users == u) for u in uniq}
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([by_user[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_old[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


# ---------------- storage ----------------
def _vdir(version: str, root: Path = REGISTRY_DIR) -> Path:
    return root / "versions" / version


def _log(event: dict, root: Path = REGISTRY_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    event = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    with open(root / "history.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


def history(root: Path = REGISTRY_DIR) -> list[dict]:
    f = root / "history.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def new_version_id(fingerprint: str) -> str:
    return f"v{datetime.now(UTC):%Y%m%d-%H%M%S}-{hashlib.sha1(fingerprint.encode()).hexdigest()[:6]}"


def register(blob: dict, metrics: dict, config: dict, version: str, root: Path = REGISTRY_DIR) -> str:
    d = _vdir(version, root)
    d.mkdir(parents=True, exist_ok=True)
    blob["meta"]["version"] = version
    joblib.dump(blob, d / "artifact.joblib")
    (d / "metrics.json").write_text(json.dumps({"version": version, "holdout": metrics, "config": config}, indent=2))
    _log({"event": "registered", "version": version, "holdout": metrics}, root)
    return version


def versions(root: Path = REGISTRY_DIR) -> list[str]:
    d = root / "versions"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def champion(root: Path = REGISTRY_DIR) -> str | None:
    f = root / "champion.json"
    return json.loads(f.read_text())["version"] if f.exists() else None


def load_version(version: str, root: Path = REGISTRY_DIR) -> dict:
    return joblib.load(_vdir(version, root) / "artifact.joblib")


def version_metrics(version: str, root: Path = REGISTRY_DIR) -> dict:
    return json.loads((_vdir(version, root) / "metrics.json").read_text())


def _set_champion(version: str, root: Path) -> None:
    (root / "champion.json").write_text(json.dumps({"version": version,
                                                    "promoted_at": datetime.now(UTC).isoformat(timespec="seconds")}))


# ---------------- gate ----------------
def gate(challenger: dict, champion_m: dict, diff: tuple[float, float, float]) -> dict:
    """Decision with the reasons spelled out. `diff` = (mean, low, high) of challenger - champion AUC."""
    mean, lo, hi = diff
    checks = {
        "auc_non_inferior": lo >= AUC_TOLERANCE,
        "brier_not_worse": challenger["brier"] <= champion_m["brier"] + BRIER_TOLERANCE,
        "calibration_holds": challenger["ece"] <= champion_m["ece"] + ECE_TOLERANCE,
    }
    return {"promote": all(checks.values()), "checks": checks, "strictly_better": lo > 0,
            "auc_diff": {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]},
            "challenger": challenger, "champion": champion_m}


def evaluate_and_promote(version: str, holdout: pd.DataFrame, force: bool = False, root: Path = REGISTRY_DIR) -> dict:
    """Compare `version` (challenger) with the current champion on the gold holdout; promote or reject."""
    y = (holdout["Rating_Stars"] >= 4).astype(int).to_numpy()
    users = holdout["User_id"].to_numpy()
    p_new = predict_holdout(load_version(version, root)["model"], holdout)
    new_m = metrics_for(y, p_new)
    current = champion(root)
    if current is None or current == version:
        _set_champion(version, root)
        decision = {"promote": True, "reason": "first model" if current is None else "already champion",
                    "challenger": new_m}
    else:
        p_old = predict_holdout(load_version(current, root)["model"], holdout)   # the champion, frozen as deployed
        decision = gate(new_m, metrics_for(y, p_old), paired_auc_interval(y, p_new, p_old, users))
        if decision["promote"] or force:
            _set_champion(version, root)
            decision["promote"] = True
            decision["forced"] = force and not all(decision["checks"].values())
    _log({"event": "promoted" if decision["promote"] else "rejected", "version": version,
          "previous_champion": current, "decision": decision}, root)
    return decision


def rollback(root: Path = REGISTRY_DIR) -> str:
    """Make the previous champion the champion again."""
    promoted = [e["version"] for e in history(root) if e["event"] == "promoted"]
    current = champion(root)
    earlier = [v for v in dict.fromkeys(reversed(promoted)) if v != current and _vdir(v, root).exists()]
    if not earlier:
        raise RuntimeError("no earlier champion to roll back to")
    _set_champion(earlier[0], root)
    _log({"event": "rollback", "version": earlier[0], "from": current}, root)
    return earlier[0]


def prune(keep: int = 5, root: Path = REGISTRY_DIR) -> list[str]:
    """Delete old versions, always keeping the champion and the newest `keep`."""
    vs, current = versions(root), champion(root)
    drop = [v for v in vs[:-keep] if v != current]
    for v in drop:
        shutil.rmtree(_vdir(v, root))
    return drop


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect and manage the model registry")
    ap.add_argument("command", choices=["list", "history", "promote", "rollback", "prune"])
    ap.add_argument("version", nargs="?", help="for promote: the version to evaluate and promote")
    ap.add_argument("--force", action="store_true", help="promote even if the gate rejects it (recorded as forced)")
    args = ap.parse_args()
    if args.command == "list":
        current = champion()
        for v in versions():
            m = version_metrics(v)["holdout"]
            print(f"{'*' if v == current else ' '} {v}  AUC {m['auc']}  Brier {m['brier']}  ECE {m['ece']}")
    elif args.command == "history":
        for e in history():
            print(e["at"], e["event"], e.get("version", ""))
    elif args.command == "promote":
        if not args.version or args.version not in versions():
            ap.error(f"give an existing version; known: {versions()}")
        from .data import load_reviews
        decision = evaluate_and_promote(args.version, split_gold(load_reviews())[1], force=args.force)
        print("PROMOTED" if decision["promote"] else "REJECTED", json.dumps(decision.get("checks", {})),
              "(forced)" if decision.get("forced") else "")
    elif args.command == "rollback":
        print("champion is now", rollback())
    else:
        print("removed", prune())


if __name__ == "__main__":
    main()
