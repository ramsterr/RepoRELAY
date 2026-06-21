# GitHub Actions Setup (do once after first deploy)

The cron workflows in `.github/workflows/` need two secrets. Set them at:
**GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret | Value | Where to get it |
|---|---|---|
| `DATABASE_URL` | Your Neon connection string | neon.tech → project → Connection Details |
| `GITHUB_TOKEN` | A GitHub PAT with `public_repo` scope | github.com/settings/tokens |

Then the three workflows will run on schedule:
- `seed.yml` — every 3h on the hour (`:00`)
- `embed.yml` — every hour at `:30`
- `trending.yml` — every 30 min (`:00` and `:30`)

You can also trigger any of them manually:
**GitHub repo → Actions → select workflow → Run workflow**

## Webhook setup (optional, after first deploy)

1. Pick a random secret string (e.g. `openssl rand -hex 32`)
2. Set it as `GITHUB_WEBHOOK_SECRET` env var in your Render dashboard
3. Render auto-redeploys
4. Run once locally to subscribe to top repos:
   ```bash
   uv run --package reporelay-mvp reporelay-mvp register-webhooks \
     --callback-url https://reporelay-mvp-api-0w1k.onrender.com \
     --secret '<the same secret>' \
     --min-stars 1000
   ```
5. Push to a watched repo's main branch → check Render logs for the webhook hit
