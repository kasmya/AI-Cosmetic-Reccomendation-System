# Frontend guide

`frontend/index.html` is a barebones wizard. Restyle it freely; the script binds to element ids, so keep those.
Live, typed API reference: run the server and open `/docs`.

| UI section (id) | Endpoint | Status |
|---|---|---|
| `step-profile` controls | `GET /api/v1/options` | live |
| `step-quiz` | `GET /api/v1/skin/questionnaire`, `POST /api/v1/skin/assess` (`answers`, `concerns`, `skin_type`, `photo_hints`) | live |
| `step-photo` | `POST /api/v1/face/analyze` (raw image bytes as body) | live |
| `res-profile` | `POST /api/v1/consult` -> `profile` (skin type, concerns with level, reasons, helpful and watch ingredients, tip) | live |
| `res-skincare` | `consult` -> `skincare` (by category, with AICosmetic score, verdict, per-component points, confidence) | live |
| `res-makeup` | `consult` -> `makeup`: `depth`, `undertone`, `foundation.items` (real shades), and for `lips` / `blush` / `eyes`: `palette` (name, hex, intensity, why) and `products` (name-matched catalog items) | live |
| makeup try-on (new) | `POST /api/v1/makeup/tryon?foundation_hex=&lips_hex=&blush_hex=&strength=` (raw image bytes in, `image/jpeg` out, header `X-Applied`) | live |
| shade shopping links (new) | `GET /api/v1/makeup/web?kind=lips&shade=coral&undertone=warm&tone=Light` | live when `TAVILY_API_KEY` is set |
| `res-routine` | `consult` -> `routine`: `am` (cleanse, treat, moisturise, protect) and `pm` steps with `name`, `price`, `spf`, `note`; `clashes` (`severity` info/caution, `slot`, `products`, `message`); `warnings`; `notes` | live |
| `res-shopping` | `consult` -> `shopping_list`: `products` (each once, with `steps`), `total_price`, `budget`, `within_budget`, `swaps`, `notes`; `items` is the list of Sephora product ids (kept for compatibility) | live |
| thumbs up/down (new) | `POST /api/v1/feedback` `{product_id, helpful, request_id}` | live |
| budget input (new) | send `total_budget` in `/consult` | live |
| find cheaper alternatives (new) | `GET /api/v1/products/{id}/dupes?skin_type=` | live |
| check my own routine (new) | `POST /api/v1/routine/check` with `{am: [product_uid], pm: [product_uid]}` (uids from `/catalog/search`) | live |
| `res-skincare` alternatives | `consult` -> `skincare[cat][i].alternatives` (similar products from other sources) | live |
| catalog browsing (not in the barebones UI yet) | `GET /api/v1/catalog/search`, `GET /api/v1/catalog/stats` | live |
| `res-product` web panel | `GET /api/v1/products/{id}/web` (`status`: live / disabled / error) | live when `TAVILY_API_KEY` is set |
| `res-product` | `GET /api/v1/products/{id}?skin_type=`, `/products/{id}/reviews/summary`, `/products/{id}/score?skin_type=&concerns=` | live |

## Conventions
- Every section has `status`: `live`, `provisional` (real but simplified) or `planned` (empty, shape final). Show a badge, or hide planned sections, your call.
- Response shapes will not change as backend phases land; planned fields just start getting filled.
- Score: `breakdown.points` are the points each component contributed (they sum, minus `penalty`, to `aicosmetic_score`); `breakdown.weights` and `review_basis` explain how. Scores cluster around 75, so show the `verdict` and the breakdown rather than only the number.
- Review summary: themes are `praised` / `criticised` / `typical` relative to other products; quotes are user-written review sentences (escape them).
- Makeup: `/face/analyze` returns `depth` and `undertone` (an estimate; let the user override and send `undertone` to `/consult`). Palette `hex` values are ready-to-use swatches. `products` are matched by a colour word in the product name, so the list is often short or empty; say that in the UI. Try-on: send the same photo bytes, show the returned JPEG; if `X-Applied` lacks `lips` the mouth was not found.
- Routine: the sunscreen step has `price: null` and a note, because its source has no prices. If `within_budget` is false, show `notes`; the routine is the cheapest one found. `swaps` is often empty and `notes` says so; do not treat that as an error.
- Feedback: keep the `request_id` from `/consult`. A thumbs up/down button on each recommended product calls `POST /api/v1/feedback` with `{product_id, helpful, request_id}`. Voting again on the same product replaces the vote. If `request_id` is null the database was unavailable; votes without it are still accepted. `/ops/*` and `/metrics` are for operators, not the UI.
- Always escape API strings before inserting HTML (the helper `esc` does this).
- Photo flow: call `/face/analyze` first, then send `skin_hex` and `skin_tone` in `/consult`. The photo itself is never sent to `/consult`.
- Photo hints: `/face/analyze` returns `hints` (low confidence). Forward them as `photo_hints` to `/skin/assess` or `/consult`; they never raise a concern above `possible` on their own.
- Quiz: render `/skin/questionnaire`, send answers as `{question_id: option_value}`. `/consult` accepts them directly and infers skin type if `skin_type` is empty.
- Errors come back as `{"detail": "message"}` with 400/413/422.
