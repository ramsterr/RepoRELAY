# Deploy

## What changed (since last deploy)

**Backend (Render):**
- `apps/mvp_api/src/reporelay_mvp_api/main.py` — in-memory recommendation cache (5 min TTL, 1000 max entries). Repeated requests for the same repo are instant.
- `Dockerfile.api` — embedding model is now baked into the image. Cold starts no longer pay the 5-9s HuggingFace download.
- `packages/mvp/src/reporelay_mvp/github.py` — GitHub search timeout cut from 15s → 5s.
- `packages/mvp/src/reporelay_mvp/recommend.py` — disk-backed search cache (24h TTL) + skip GitHub search when DB pool already has ≥200 candidates.

**Frontend (Vercel):**
- **`astro.config.mjs`** — switched from `output: "server"` → `output: "static"`. Pages are now pre-rendered at build time and served from Vercel's CDN — zero cold starts, zero serverless function lag.
- **`vercel.json` (new)** — rewrite rule maps `/repo/owner/name` → `/repo.html` so dynamic repo URLs work without SSR.
- **`apps/site/src/pages/repo.astro` (new)** — replaces the old `repo/[owner]/[name].astro` SSR page. Reads the repo name from `window.location.pathname` client-side.
- Increased client-side fetch timeout to 15s and added 1 automatic retry after 2s.

**Keepalive:**
- `.github/workflows/ping.yml` — now also hits `/recommend?repo=fastapi/fastapi&limit=1&tags=python` so the embedding model is loaded into RAM between pings, not just on first real user request.

## Architecture

```
Browser                          Vercel CDN (static)          Render (Docker)
──────                          ──────────────────          ────────────────
GET /              ────►  index.html  (<1ms, static)        
GET /explore       ────►  explore.html (<1ms, static)       
GET /repo/f/b      ────►  vercel.json rewrite ─► repo.html  
                            │                                  GET /recommend?repo=f/b
                            │    fetch("/recommend?...") ────►  (cached if warm)
                            │    ◄──── JSON ────────────────
                            │
                          All pages render instantly.
                          No serverless functions.
                          No cold starts.
```

## Deploy steps

### 1. Render (API)

Render watches your repo and auto-rebuilds on push to the connected branch.

```bash
git add apps/mvp_api/ .github/workflows/
git commit -m "perf: add recommendation cache + static frontend"
git push
```

Then in the Render dashboard:
1. Wait for the build to start (or trigger **Manual Deploy → Clear build cache & deploy** if it doesn't pick up the push).
2. First build will be slower (~30s extra) because it downloads the 130MB model.
3. Watch the logs for `embedding model preloaded and ready` — that's how you know the model is in the image.

**Verify it works:**
```bash
curl -w "\n%{http_code} in %{time_total}s\n" \
  "https://reporelay-mvp-api-0w1k.onrender.com/recommend?repo=fastapi/fastapi&limit=3"
```
Should return 200 in **under 1 second** on a warm container, **under 5 seconds** on a cold one (was 15-25s before).

### 2. Vercel (site)

Vercel also auto-rebuilds on push. The key env var is `PUBLIC_API_URL` — **must be set** to the Render URL in Vercel's project settings.

```bash
# same push as above — Vercel picks it up automatically
```

**Important**: After the deploy, visit the Vercel project dashboard → Settings → Environment Variables and verify `PUBLIC_API_URL` is set. If it's missing, the frontend will try to fetch from `localhost:8001` and fail.

**Verify in browser:**
1. Open the Vercel URL.
2. The page should render immediately (<100ms from CDN). No more blank white screen.
3. Recommendations should appear within a few seconds. If you see "still searching…" for more than 3 seconds, the API is likely cold — wait a moment and try again (or check that UptimeRobot is pinging the API).
4. Try the URL directly: `https://reporelay-site.vercel.app/repo/facebook/react` — should show the page shell instantly, then load recommendations.

### 3. UptimeRobot (prevent cold starts entirely)

You said you have UptimeRobot set up for the Vercel site. **Add the Render API too** — this is the single biggest win because it stops Render from ever spinning down:

1. UptimeRobot → Add New Monitor
2. Monitor Type: **HTTP(s)**
3. URL: `https://reporelay-mvp-api-0w1k.onrender.com/health`
4. Monitoring Interval: **5 minutes** (Render's free plan spins down at 15min)
5. Save

With UptimeRobot pinging every 5min, the Render container **never goes cold** and the GitHub Actions keepalive becomes belt-and-suspenders.

### 4. Verify the keepalive is doing its job

After deploy, wait 25 minutes (longer than Render's spin-down window) without hitting the site yourself. Then:

```bash
curl -w "\n%{http_code} in %{time_total}s\n" \
  "https://reporelay-mvp-api-0w1k.onrender.com/recommend?repo=fastapi/fastapi&limit=3"
```

If the keepalive + UptimeRobot are working, this should still be **under 1 second** (the cache is warm from the keepalive). If it's 5+ seconds, the model got unloaded — check the GitHub Actions tab to make sure the cron is firing.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Vercel site shows blank / 404 for repo pages | `vercel.json` not deployed. Verify it's in `apps/site/` and committed. |
| Frontend shows "API is unreachable" | `PUBLIC_API_URL` not set in Vercel env vars, or Render is down. |
| Render build fails on `sentence_transformers` import | The `RUN python -c ...` step needs the package installed first. Make sure `uv sync` runs before the model download in the Dockerfile. |
| Model still downloads on cold start | `HF_HOME` not set in the running container. Check the deployed env vars in Render. |
| Vercel site shows "failed to load API" | `PUBLIC_API_URL` got unset. Vercel → Settings → Environment Variables. |
| `/recommend` still takes 5+ seconds | Render cold start + model not pre-baked. Check Render build logs for the model download step. |
| GitHub Actions ping failing | Repo → Settings → Actions → Workflow permissions → "Read and write permissions" (needed for some setups). |

## Cost note

Render free plan: 750 hours/month, sleeps after 15min idle. With UptimeRobot pinging every 5min, you stay well within free tier and never sleep. No change needed.
