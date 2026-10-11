"""Fill the feedback table with SYNTHETIC votes so the retraining loop can be demonstrated end to end.

    python scripts/simulate_feedback.py --n 400          add synthetic consultations and thumbs up/down
    python scripts/simulate_feedback.py --clear          remove them again

What this is: each vote replays a real training-set review (a 4-5 star review becomes a thumbs up, with 10% of votes
flipped as noise). The gold-holdout users are never used. It adds NO new information about anyone: it exists to exercise
the plumbing (database -> retrain -> registry gate). Synthetic rows are tagged model_version='synthetic-demo'.
Real feedback is thumbs up/down people give in the app, and arrives slowly.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cosmetic_rec import registry  # noqa: E402
from cosmetic_rec.data import load_reviews  # noqa: E402
from cosmetic_rec.db import Database, Feedback, RecommendationEvent  # noqa: E402

TAG = "synthetic-demo"


def clear(db: Database) -> int:
    with db.session() as s:
        ids = [e.id for e in s.query(RecommendationEvent).filter(RecommendationEvent.model_version == TAG)]
        s.query(Feedback).filter(Feedback.event_id.in_(ids)).delete(synchronize_session=False)
        s.query(RecommendationEvent).filter(RecommendationEvent.id.in_(ids)).delete(synchronize_session=False)
    return len(ids)


def simulate(db: Database, n: int, seed: int = 0, noise: float = 0.10) -> int:
    rng = random.Random(seed)
    train, _ = registry.split_gold(load_reviews())
    sample = train.sample(n, random_state=seed)
    for r in sample.itertuples():
        skin = r.Skin_Type if isinstance(r.Skin_Type, str) else None
        shown = [{"product_id": int(r.Product_id), "score": 0, "p_like": 0.0, "verdict": "Okay"}]
        event_id = db.log_event(TAG, skin, [], None, shown)
        helpful = r.Rating_Stars >= 4
        if rng.random() < noise:
            helpful = not helpful
        db.add_feedback(int(r.Product_id), helpful, event_id)
    return len(sample)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()
    database = Database()
    if args.clear:
        print("removed", clear(database), "synthetic consultations")
    else:
        print("added", simulate(database, args.n), "synthetic votes;", database.counts())
