# Frontend guide

`static/index.html` is a barebones wizard. Restyle it freely; the script binds to element ids, so keep those.
Live, typed API reference: run the server and open `/docs`.

| UI section (id) | Endpoint | Status |
|---|---|---|
| `step-profile` controls | `GET /api/v1/options` | live |
| `step-quiz` | `GET /api/v1/skin/questionnaire`, `POST /api/v1/skin/assess` (`answers`, `concerns`, `skin_type`, `photo_hints`) | live |
| `step-photo` | `POST /api/v1/face/analyze` (raw image bytes as body) | live |
| `res-profile` | `POST /api/v1/consult` -> `profile` (skin type, concerns with level, reasons, helpful and watch ingredients, tip) | live |
| `res-skincare` | `consult` -> `skincare` (by category, with AICosmetic score, verdict, per-component points, confidence) | live |
| `res-makeup` | `consult` -> `makeup` (foundation live; lips, blush, eyes planned) | partly live |
| `res-routine` | `consult` -> `routine` | provisional |
| `res-shopping` | `consult` -> `shopping_list` | provisional |
| `res-skincare` alternatives | `consult` -> `skincare[cat][i].alternatives` (similar products from other sources) | live |
| catalog browsing (not in the barebones UI yet) | `GET /api/v1/catalog/search`, `GET /api/v1/catalog/stats` | live |
| `res-product` web panel | `GET /api/v1/products/{id}/web` (`status`: live / disabled / error) | live when `TAVILY_API_KEY` is set |
| `res-product` | `GET /api/v1/products/{id}?skin_type=`, `/products/{id}/reviews/summary`, `/products/{id}/score?skin_type=&concerns=` | live |

## Conventions
- Every section has `status`: `live`, `provisional` (real but simplified) or `planned` (empty, shape final). Show a badge, or hide planned sections, your call.
- Response shapes will not change as backend phases land; planned fields just start getting filled.
- Score: `breakdown.points` are the points each component contributed (they sum, minus `penalty`, to `aicosmetic_score`); `breakdown.weights` and `review_basis` explain how. Scores cluster around 75, so show the `verdict` and the breakdown rather than only the number.
- Review summary: themes are `praised` / `criticised` / `typical` relative to other products; quotes are user-written review sentences (escape them).
- Always escape API strings before inserting HTML (the helper `esc` does this).
- Photo flow: call `/face/analyze` first, then send `skin_hex` and `skin_tone` in `/consult`. The photo itself is never sent to `/consult`.
- Photo hints: `/face/analyze` returns `hints` (low confidence). Forward them as `photo_hints` to `/skin/assess` or `/consult`; they never raise a concern above `possible` on their own.
- Quiz: render `/skin/questionnaire`, send answers as `{question_id: option_value}`. `/consult` accepts them directly and infers skin type if `skin_type` is empty.
- Errors come back as `{"detail": "message"}` with 400/413/422.
