# AI Cosmetic Recommendation System

Skincare recommendations from 8,649 real reviews of 315 products. Python + scikit-learn backend, plain HTML/CSS/JS frontend.

## How it works
1. **Filters:** category, max price, sensitive skin, ingredients to avoid.
2. **Gradient boosting classifier** predicts the probability that a user with a given skin type rates a product 4+ stars (features: skin type, category, brand, price, site rating, suitability labels, ingredient embedding).
3. **Ingredient embeddings:** TF-IDF followed by truncated SVD (LSA). Cosine similarity finds products similar to ones the user already liked.
4. Final score is a blend of (2) and (3). Each result shows its reason.

## Run
```bash
pip install -r requirements.txt
uvicorn cosmetic_rec.api:app --reload      # UI at http://localhost:8000, API docs at /docs
```
Docker: `docker build -t cosmetic-rec . && docker run -p 8000:8000 cosmetic-rec`

## Evaluation
`python -m cosmetic_rec.evaluate` (the numbers below come from this script).

**A. Will a user rate it 4+ stars?** 5-fold CV grouped by user, ROC-AUC:

| Model | AUC |
|---|---|
| Baseline: site rating | 0.591 |
| Logistic regression | 0.592 |
| Neural net (MLP) | 0.578 |
| **Gradient boosting (used)** | **0.598** |

**B. Find the hidden product.** One liked product per user is hidden (417 users, ~315 candidates), mean of 5 seeds:

| Model | HitRate@10 |
|---|---|
| Random | 3.2% |
| ML P(like) only | 3.5% |
| Ingredient similarity only | 8.3% |
| **Hybrid (used)** | **8.7%** |

**Honest reading:** the dataset is small and most users wrote one review, so predicting satisfaction is hard (AUC ~0.6, barely above using the site rating; a neural net did not help). Ingredient similarity is the component that clearly beats chance (about 2.7x random). Skin-type signal is weak.

## Why a rewrite
The original notebook reported 100% accuracy because its label `Good_Stuff` is exactly `Rating_Stars == 5` and `Rating_Stars` was a model input, which is target leakage. The same dataset also contains repeat user/product reviews that can leak a held-out item into training; the evaluation here removes them.

## Layout
```
cosmetic_rec/   data.py  recommender.py  evaluate.py  api.py
static/         index.html (HTML/CSS/JS)
tests/          pytest suite
skindataall.csv dataset
```
Tests: `pip install -r requirements-dev.txt && pytest`
