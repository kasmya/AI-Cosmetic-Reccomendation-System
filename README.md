# AI Cosmetic Recommendation System

Ranks skincare products from 8,649 real reviews of 315 products, with a plain-English reason for every result.

## How it works
Hard filters (category, budget, sensitive skin, ingredients to avoid), then a score combining:

1. **Quality**: Bayesian-smoothed mean star rating, so 3 reviews at 5 stars cannot beat 200 reviews at 4.6.
2. **Skin fit**: the same rating computed only over reviewers with the user's skin type.
3. **Label fit**: the product's own "suitable for X skin" label.
4. **Taste**: TF-IDF ingredient similarity to products the user already liked.

## Run
```bash
pip install -r requirements.txt
uvicorn cosmetic_rec.api:app --reload     # UI at http://localhost:8000, docs at /docs
```
Or with Docker: `docker build -t cosmetic-rec . && docker run -p 8000:8000 cosmetic-rec`

```bash
curl -X POST localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"skin_type":"Dry","category":"Moisturizer","max_price":50,"avoid_ingredients":["fragrance"],"k":3}'
```

## Evaluation
```bash
python -m cosmetic_rec.evaluate
```
One liked product per user is hidden (1,130 users); the model ranks ~315 candidates.

| Model | HitRate@10 |
|---|---|
| Random | 1.3% |
| Popularity | 0.0% |
| Quality + skin fit | 1.2% |
| **Full model** | **3.0%** |

Honest caveats: the lift over random is modest. The dataset has ~24 reviews per product, so popularity carries
no signal, and most users wrote a single review. Only ingredient similarity measurably beats chance; the rating
components cannot be validated by this protocol, which measures what users reviewed, not what they would rate highly.

## Why a rewrite
The original notebook reported 100% accuracy because its label `Good_Stuff` is exactly `Rating_Stars == 5`
and `Rating_Stars` was a model input. That was target leakage. `python -m cosmetic_rec.evaluate` prints the check.
The old code is in git history.

## Tests
```bash
pip install -r requirements-dev.txt && pytest
```
