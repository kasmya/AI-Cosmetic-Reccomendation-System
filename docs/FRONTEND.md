# Frontend guide

`static/index.html` is a barebones wizard. Restyle it freely; the script binds to element ids, so keep those.
Live, typed API reference: run the server and open `/docs`.

| UI section (id) | Endpoint | Status |
|---|---|---|
| `step-profile` controls | `GET /api/v1/options` | live |
| `step-photo` | `POST /api/v1/face/analyze` (raw image bytes as body) | live |
| `res-profile` | `POST /api/v1/consult` -> `profile` (concern to ingredient guidance) | provisional |
| `res-skincare` | `consult` -> `skincare` (by category, with score, verdict, breakdown) | live products, provisional score |
| `res-makeup` | `consult` -> `makeup` (foundation live; lips, blush, eyes planned) | partly live |
| `res-routine` | `consult` -> `routine` | provisional |
| `res-shopping` | `consult` -> `shopping_list` | provisional |
| `res-product` | `GET /api/v1/products/{id}` and `/reviews/summary` | product live, review summary planned |

## Conventions
- Every section has `status`: `live`, `provisional` (real but simplified) or `planned` (empty, shape final). Show a badge, or hide planned sections, your call.
- Response shapes will not change as backend phases land; planned fields just start getting filled.
- Always escape API strings before inserting HTML (the helper `esc` does this).
- Photo flow: call `/face/analyze` first, then send `skin_hex` and `skin_tone` in `/consult`. The photo itself is never sent to `/consult`.
- Errors come back as `{"detail": "message"}` with 400/413/422.
