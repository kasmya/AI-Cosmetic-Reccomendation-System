# Frontend

One self-contained page: [frontend/index.html](../frontend/index.html) (HTML, CSS and JavaScript inline; fonts and Lenis smooth
scroll load from a CDN). Images live in `frontend/assets/` and are referenced as `assets/<file>`, so they work both from the
server (`/assets/<file>`) and when the file is opened from disk.

Run it: `make run`, then open <http://localhost:8000>. Opened as a plain file the page cannot reach the API and says so in a
banner. Live API reference: `/docs`.

## What a visitor can do, all on this one page

| Section | What it does | Endpoints |
|---|---|---|
| Consultation, 6 steps | skin type and sensitivity; **skin quiz** (8 questions, optional); concerns and colour preferences; total budget, max price and avoid list; photo (optional) with undertone override; review before generating | `GET /api/v1/options`, `GET /api/v1/skin/questionnaire`, `POST /api/v1/skin/assess`, `POST /api/v1/face/analyze`, `POST /api/v1/consult` |
| Your beauty edit | profile and concerns with reasons; skincare with AICosmetic score, per-component points and confidence; AM/PM routine with SPF, clashes and notes; makeup (foundation shades with ΔE, lip/blush/eye palettes, catalog matches); shopping list with total, budget status and cheaper swaps | `POST /api/v1/consult` |
| Product drawer | score breakdown, review summary (themes, praised and criticised, quotes, skin-type stats), ingredients, cheaper alternatives, live web results, thumbs up/down | `GET /api/v1/products/{id}`, `/products/{id}/score`, `/products/{id}/dupes`, `/products/{id}/web`, `POST /api/v1/feedback` |
| Colour studio | compare shades per category, add to a comparison, "Shop this shade online" | `GET /api/v1/makeup/web` |
| Try the shades on | foundation, lips and blush applied to the visitor's photo, strength slider, before/after | `POST /api/v1/makeup/tryon?foundation_hex=&lips_hex=&blush_hex=&strength=` (photo bytes in, JPEG out, `X-Applied` header) |
| Product index | search all sources, open details, add products to the routine checker | `GET /api/v1/catalog/search`, `/catalog/stats` |
| Routine check | add products to morning and evening, or load the generated routine, and check for ingredient clashes | `POST /api/v1/routine/check` |
| How this works | the real model numbers | `GET /model-info` |
| Live demo | three sample faces with **hardcoded** sample data, clearly labelled; makes no API calls | none |

## Rules the page follows

- **It never invents results.** If the server is unreachable it says so (a banner on load, an error where a result would be). The earlier
  "data fallback layer" that fabricated products, scores, skin analysis and try-ons when an API call failed was removed.
- Every string from the API is escaped before it is inserted (`esc`), and links only open if they are `http(s)` (`safeUrl`).
- `request_id` from `/consult` is kept and sent with each thumbs up/down so a vote is tied to that consultation.
- Statuses (`live`, `provisional`, `planned`) show as badges; set `LAUNCH = true` in the script to hide `planned` ones.
- Colour, depth, undertone and photo hints are labelled as estimates with low confidence.

## Keeping it linked to the backend

`tests/test_frontend.py` parses this page and checks it against the real API: every URL it calls exists, every request field is accepted
by the schema, every response field it reads is present in a real response, every image is served, no value the API rejects is
hardcoded, and the JavaScript parses (`node --check`). Run `FRONTEND_HTML=other.html pytest tests/test_frontend.py` to check another
version of the page. After changing an endpoint, run `pytest tests/test_frontend.py`.

## Editing

Keep the element ids: the script binds to them. The reveal animations use `data-r` attributes; elements revealed with
`clip-*` are observed through their parent (a fully clipped element is reported by the browser as never visible, which used to
leave the hero and studio images blank).

## Feature coverage (what the page really does)

Every row below is reachable from the single page and was exercised against a live server.

| Feature | Where in the page | Endpoint |
|---|---|---|
| Skin-type quiz, concerns, avoid list, budget | Consultation form | `POST /api/v1/consult` |
| Products you already like (similarity boost, never recommended back) | Liked picker | `consult.liked_product_ids` |
| Results per category (1 to 6) | `#per_cat` | `consult.per_category` |
| Photo analysis (face, skin hex, tone, depth, undertone) | Photo upload and sample faces | `POST /api/v1/face/analyze` |
| Manual tone or hex when there is no photo | Tone select and hex field | `consult.skin_tone`, `skin_hex` |
| AICosmetic score, verdict, breakdown | Result cards and drawer | `consult` |
| Review themes and sentiment | Drawer | `consult` |
| Routine (AM/PM), clash warnings, sunscreen, cheaper swaps, budget | Routine band and shopping list | `consult` |
| Foundation match with brand filter and k | Colour studio | `GET /api/v1/makeup/match`, `/makeup/brands` |
| Shade suggestions for a photo | Try-on studio | `POST /api/v1/makeup/suggest` |
| Virtual try-on (foundation, lips, blush, strength) | Try-on studio, before/after, save | `POST /api/v1/makeup/tryon` |
| Save or print my edit | Edit actions | client-side |
| Model, registry, drift and feedback status | "Under the hood" band | `GET /api/v1/system` |

Limits, stated plainly: tone, depth and undertone are estimates from one photo and lighting; the try-on is a colour-space preview, not a product swatch; sunscreens are unscored because the catalog has no prices or reviews for them.
