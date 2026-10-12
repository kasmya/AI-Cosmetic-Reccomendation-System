# Deploy: Cloudflare Pages + Render (Docker) + Neon (Postgres)

The browser only talks to the Pages domain. `frontend/_worker.js` serves the static site and forwards API calls to the
backend (`BACKEND_URL`); `/api/v1/ops/*` is never forwarded.

## 1. Database (Neon, free)
Create a project at neon.tech and copy the connection string (`postgresql://...?sslmode=require`).

## 2. Backend (Render, free)
New > Blueprint > pick this repo (it reads `render.yaml`). Set `DATABASE_URL` to the Neon string; `TAVILY_API_KEY` is optional.
The trained model registry is committed (`models/registry`) and copied into the image, because training on Render's free builder ran over an hour; retrain locally and commit to update it. A service created through the API (not the blueprint) needs `OPS_TOKEN` set by hand. Check `https://<service>.onrender.com/health`.
Free services sleep after 15 minutes idle; the first request afterwards takes about a minute.

## 3. Frontend (Cloudflare Pages)
```bash
npx wrangler login
npx wrangler pages project create colour-atelier --production-branch main
npx wrangler pages deploy frontend --project-name colour-atelier
npx wrangler pages secret put BACKEND_URL --project-name colour-atelier   # https://<service>.onrender.com
npx wrangler pages deploy frontend --project-name colour-atelier           # redeploy so the secret applies
```
Open `https://colour-atelier.pages.dev`, run a consultation, and check the "Under the hood" band shows live status.

Limits: free tiers are for hobby use, the Render disk is not persistent (Postgres holds feedback), and the model is
trained locally and committed.
