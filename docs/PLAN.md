# Build plan

Goal: a user enters their skin type and concerns and colour preferences, uploads a photo, and gets a complete
shopping list: skincare for their skin, makeup that suits their colouring, a review summary and a transparent
"AICosmetic score" per product, and an AM/PM routine.

Status key: DONE / NEXT / PLANNED. The API contract (Phase 1) is fixed up front so frontend and backend can move in parallel.

## Honest scope decisions
| Request | What we will build | What we will not claim |
|---|---|---|
| "Diagnose skin problems" | A guided concern assessment (questionnaire + optional photo hints for redness/shine/blemish density with OpenCV) that maps to concern categories and ingredient suggestions | A medical diagnosis. Everything is labelled "not medical advice, see a dermatologist for persistent or severe issues" |
| "Products from varied sources" | Merge several **open or user-approved datasets** into one product table with provenance per row | Live scraping of retailer sites (terms-of-service and legal risk, and brittle) |
| "Summarize reviews" | NLP over the review text we have: sentiment model + top themes (dryness, breakouts, fragrance, sticky...) + 2-3 representative quotes | An LLM-written summary (optional later, adds a key and cost) |
| "AICosmetic approval rating" | A published formula, 0-100, with a visible breakdown | A "scientific" safety rating |
| "Makeup that suits them" | Foundation by measured skin colour (live), plus colour-harmony rules for lips/blush/eyes from tone and undertone and the user's colour preferences, and a virtual shade try-on | Lip/blush product matches until we have a dataset with shade hex values |

## Phase 0: Foundations (DONE)
Skincare ML recommender (P(like) model + ingredient embeddings), foundation shade matcher (OpenCV + k-NN), FastAPI, tests, Docker, CI.

## Phase 1: Contract and skeleton (DONE in this commit)
- `/api/v1/*` endpoints with Pydantic schemas (visible at `/docs`). Live where the engine exists, stubbed and flagged `"status": "planned"` elsewhere.
- Barebones `static/index.html`: a step-by-step wizard wired to those endpoints, with one clearly marked section per feature, ready to restyle.
- Done when: frontend runs end to end against the API, tests cover the contract.

## Phase 2: Data sources and ingestion (DONE)
- Inventory of sources, each with licence, fields and what it adds:
  1. Sephora skincare reviews (have): products, ingredients, reviews, skin type.
  2. The Pudding shades (have): 621 foundation shades.
  3. Open Beauty Facts (open licence): extra products and ingredient lists, including makeup. Needs your OK to download.
  4. A lip/blush/eyeshadow dataset with shade hex values (needed for makeup beyond foundation). Candidates: Makeup API when it is back, or a Kaggle dataset you download.
- Unified `products` table: id, source, category, brand, name, price, ingredients, shades, url, licence.
- Ingredient normalisation (INCI names) so lists from different sources compare.
- Done when: one loader builds the table with provenance, row counts per source are reported, tests check schema.
- Risk: sources disagree on categories and price currency. Mitigation: mapping tables plus validation tests.

## Phase 3: Skin profile and concern assessment (DONE)
- Questionnaire to concern categories (acne, dryness, oiliness, sensitivity, dark spots, redness, ageing) with weights.
- Optional photo hints with OpenCV (redness index in Lab a*, shine from highlight pixels, blemish density by blob detection). Reported as hints with low confidence, never a diagnosis.
- Concern to ingredient map (for example acne: salicylic acid, niacinamide; dryness: ceramides, hyaluronic acid) used to boost and filter skincare.
- Done when: `/skin/assess` returns concerns with reasons and ingredient guidance, unit-tested on hand-made cases.
- Risk: photo hints are noisy. Mitigation: show confidence, let the user override.

## Phase 4: Review intelligence and AICosmetic score (DONE)
- Sentiment classifier (TF-IDF + logistic regression) on review text, with held-out accuracy reported.
- Theme extraction per product (keyword and n-gram scoring) plus representative quotes.
- AICosmetic score 0-100 = weighted blend of: personalised P(like) (ML model), sentiment of reviews from similar skin types, ingredient flags (user's avoid list, concern fit), review volume confidence. Weights published in the README and returned in the response.
- Verdict bands: Recommended / Okay / Caution, each with the reasons.
- Done when: every product returns a score with breakdown, and a test shows the score changes sensibly with inputs.

## Phase 5: Makeup engine
- Foundation: live (OpenCV skin colour + k-NN). Add undertone (warm/cool/neutral) from Lab hue.
- Colour harmony for lips, blush, eyes from tone + undertone + the user's colour preferences (for example warm, nude, berry, bold) as rule-based palettes with hex swatches.
- Product matches for those categories once Phase 2 supplies shade data.
- Virtual try-on: blend the chosen shade on the photo (OpenCV masking).
- Done when: user preferences visibly change the palette and results, with tests on the rules.

## Phase 6: Complete list and routine
- AM/PM routine builder (cleanser, treatment, moisturizer, SPF) within budget, with ingredient clash warnings (for example retinol with exfoliating acids).
- Shopping list: total price, swaps for cheaper similar products (dupe finder using ingredient similarity).
- Done when: `/consult` returns a routine and shopping list that respects budget and avoid-lists.

## Phase 7: Architecture and operations
- Two-stage recommender (candidate generation, ranking, re-ranking), model registry with champion/challenger promotion, Postgres, feedback loop, Prometheus/Grafana, drift check, load test, docker-compose.

## Phase 8: Frontend polish and deploy
- Restyle (terracotta editorial design), accessibility, mobile, deploy to a public URL.

## Decisions needed from you
1. OK to download Open Beauty Facts data (open licence)?
2. Do you have, or want me to look for, a lip/blush/eyeshadow dataset with shade colours?
3. Confirm the "not medical advice" framing for concern assessment.

## Phase 2 results
- Sources in the unified catalog (`python -m cosmetic_rec.catalog`): Sephora reviews (315 products, with prices and reviews) and Open Beauty Facts (1,681 face skincare and makeup products with ingredients, no prices or reviews). Foundation shades (621) stay a separate shade table.
- Shared ingredient cleaning (`ingredients.py`): synonym map (Aqua to water, Parfum to fragrance), noise removal, junk detection. 41 of 315 Sephora ingredient fields (13%) were not ingredient lists and are now treated as missing.
- Shared category vocabulary (`categories.py`) with 14 categories. New for skincare: Sunscreen (464 products, used for the SPF step in Phase 6) and Toner. New for makeup: Foundation, Concealer, Blush, Lipstick, Mascara, Eyeshadow, Eyeliner (Open Beauty Facts has product names but no shade colours, so lip/blush matching by colour still needs a shade dataset).
- Cross-source features: `/api/v1/catalog/search`, `/api/v1/catalog/stats`, and "also consider" alternatives (ingredient similarity across sources) inside `/consult`.
- Optional live web search (Tavily) behind `/api/v1/products/{id}/web`, off unless `TAVILY_API_KEY` is set.
- Effect on the skincare model: none measurable (AUC 0.599 to 0.595, hybrid hit rate 8.6% to 8.2%, both within noise).
- Known limits: Open Beauty Facts is mostly European, product names are in many languages, and category assignment is keyword based, so some products are mislabelled. Brand is missing for some rows.

## Phase 3 results
- **Questionnaire** (8 questions, `GET /api/v1/skin/questionnaire`): answers infer skin type and score 8 concerns with reasons (levels: selected / likely / possible). `POST /api/v1/skin/assess` is now live; `/consult` runs it internally, so a single call can use `answers` and `photo_hints`.
- **Concern knowledge base** (`concerns.py`): per concern, helpful ingredients, ingredients often reported as aggravating ("watch"), and a short tip. It is a hand-written lookup, not clinical evidence, and nothing here has been validated against outcomes (the datasets have no outcome data).
- **Ranking effect:** concerns now change skincare rankings. Ingredient fit adds a z-scored boost; watch ingredients take a flat penalty. Measured on this catalog: with the acne concern the top-10 contains 2.0 helpful ingredients per product on average (vs 0.3 without); with sensitivity, products containing fragrance and similar fall from 6 of the top 10 to 1 (the penalty had to be 2.5, since 59% of products contain one and a gentler penalty changed nothing). The model's own P(like) for the picks drops a little (0.89 to 0.80), the expected cost.
- **Photo hints (OpenCV)** in `/face/analyze` (`hints`): redness (share of cheek pixels redder than the person's own median), shine (small specular highlights relative to surrounding skin) and blemishes (small spots redder than surroundings). Labelled low confidence. A photo alone can never push a concern past "possible".
- **Checked on synthetic patches only:** clean skin scores low on all three; patches with added spots, red areas or highlights score high. There is no labelled face dataset behind these thresholds, so treat the hints as a weak nudge. Two issues found and fixed on the way: the skin mask excluded very red pixels (hiding the redness it measures), and an "absolute brightness" shine test fired on a brightly lit photo.
- Contract changes are additive (new optional fields and endpoints). 57 tests.

## Phase 4 results
- **Sentiment model** (`reviews_nlp.py`): TF-IDF (1-2 grams) + logistic regression, "is this a 4-5 star review?". Held-out (users never shared between train and test, 7,498 de-duplicated reviews): ROC-AUC 0.956, accuracy 87.5% vs 74.7% for always guessing positive, F1 on negative reviews 0.69. Note this is a sentiment task (the text and the rating come from the same person), not a recommendation task.
- **Review summaries** (`GET /api/v1/products/{id}/reviews/summary`): counts, positive share overall and for the user's skin type, aspect themes (hydration, breakouts, irritation, scent, texture, results, gentleness, value, packaging) judged relative to how all products score on that aspect (praised / criticised / typical), pros and cons, and representative quotes. About 18% of product-aspect pairs come out praised, 17% criticised, 65% typical. Aspect polarity applies a review-level model to single sentences, so it is an estimate, not ground truth.
- **Calibration:** the classifier is Platt-scaled with user-grouped folds. Expected calibration error 0.026 to 0.021; the top predicted probability fell from 0.96 to 0.88. The earlier "96% chance" values were mostly a winner's-curse effect (picking the highest of many noisy estimates).
- **AICosmetic score** (`score.py`, formula in the module docstring and in `/products/{id}/score`): 0-100 from the calibrated model probability, the smoothed positive-review share (skin-matched when at least 3 reviewers share the skin type) and, with concerns, ingredient fit, minus a penalty for aggravating ingredients. Verdict bands Recommended 80+, Okay 70+, otherwise Caution, which split this catalog into roughly the top 20%, middle 60% and bottom 20%. Every response carries the points each component contributed and a confidence level from the review count.
- **Does the score predict held-out reviews?** ROC-AUC 0.599, essentially equal to its parts (model 0.600, reviews 0.595) and to the site rating (0.591). Predicting one person's rating from the product alone looks capped near 0.6 in this data. The score's value is that it is calibrated, personalised and explainable, not that it is more predictive. Skin-matched review shares did not beat overall shares (0.592 vs 0.595, within noise).
- Training now also fits and saves the review model (about 11 s). 73 tests.
