# Deployment Issues — RepoRelay

Notes from the version-0 deploy to Vercel + Render + Neon. Each entry
is something that broke and how it was fixed — so the next person (or
future-me) doesn't have to rediscover it.

---

## 1. Hardcoded `http://localhost:8001` everywhere

**Symptom:** Site builds fine, but the frontend can't reach the API in production.

**Files affected:**
- `apps/site/src/lib/api.ts:1`
- `apps/site/src/pages/index.astro:13`
- `apps/site/src/pages/explore.astro:22` (inside `<script is:inline>`)

**Why:** Astro/JS code doesn't auto-read env vars. You have to wire them in.

**Fix:**
- For `.ts` / `.astro` frontmatter: `import.meta.env.PUBLIC_API_URL`
- For inline `<script>` blocks: pass via `define:vars={{ API: ... }}` in the frontmatter

**Lesson:** Pick the env-var convention (`PUBLIC_*` for Vite, `NEXT_PUBLIC_*` for Next, etc.) on day one and grep for hardcoded URLs before any deploy.

---

## 2. `output: "server"` needs an adapter, not just a build

**Symptom:** Build fails with `[NoAdapterInstalled] Cannot use server-rendered pages without an adapter.`

**Why:** Astro 5 with `output: "server"` (SSR) needs a host adapter — `@astrojs/vercel`, `@astrojs/node`, etc. Static-only output (`"static"`) doesn't.

**Fix:**
```bash
pnpm add @astrojs/vercel   # Astro 5 needs v8.x, not v10
```
```js
// astro.config.mjs
import vercel from "@astrojs/vercel";
export default defineConfig({ output: "server", adapter: vercel() });
```

**Version trap:** Latest `@astrojs/vercel@10` requires Astro 6. Astro 5 needs v8. Always check the peer-dep range before `pnpm add`.

---

## 3. Render doesn't have `uv` pre-installed

**Symptom:** Build fails with `uv: command not found` on Render's native Python runtime.

**Why:** Render's Python runtime expects `pip` + `requirements.txt`. Our monorepo uses `uv` for workspace management.

**Fix:** Two options:
- (A) `buildCommand: pip install uv && uv sync --frozen` — works on native Python runtime
- (B) Switch to Docker runtime with a `Dockerfile.api` — slower build but more control

We went with (B). The Dockerfile installs uv, copies the workspace, runs `uv sync --frozen --no-dev`, and starts uvicorn.

**Lesson:** `runtime: docker` is more reliable than the native Python runtime for monorepos. Tradeoff: slower build (~2-3 min vs ~30s).

---

## 4. `postgresql://` vs `postgresql+psycopg://` driver prefix

**Symptom:** `ModuleNotFoundError: No module named 'psycopg2'` even though `psycopg` (v3) is installed.

**Why:** SQLAlchemy defaults to `psycopg2` when it sees `postgresql://`. Neon gives you `postgresql://...` (no driver prefix). We have `psycopg[binary]`, not `psycopg2`.

**Fix:** Normalize the URL in `settings.py` AND `migrations/env.py`:
```python
if v.startswith("postgresql://"):
    return "postgresql+psycopg://" + v[len("postgresql://"):]
```

**Lesson:** Hosts give you `postgresql://` URLs. Your code needs `postgresql+psycopg://` (or `+asyncpg`, `+psycopg2`, etc.). Pick one and normalize everywhere.

---

## 5. Alembic ignored `DATABASE_URL` env var

**Symptom:** `alembic upgrade head` runs against the local Docker Postgres, not Neon — even with `export DATABASE_URL=...`

**Why:** `migrations/env.py` was reading `sqlalchemy.url` from `alembic.ini` (hardcoded to localhost), not from the env var.

**Fix:** Make `env.py` read `DATABASE_URL` first, fall back to the ini value:
```python
def _resolve_url() -> str:
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    return config.get_main_option("sqlalchemy.url") or ""
```

**Lesson:** Every entry point that connects to the DB needs to read `DATABASE_URL`. Don't assume the alembic.ini URL will be overridden.

---

## 6. Migration state mismatch (corrupted `alembic_version`)

**Symptom:** `relation "mvp_repos" does not exist` even though `alembic current` says `mvp_002 (head)`.

**Why:** Earlier debugging left the version table at `mvp_002` but the table was never created. Or: `alembic stamp base` was run after the migrations, leaving the version table empty while the table existed.

**Fix:** Re-stamp to skip the existing migrations, then run upgrade:
```bash
alembic stamp mvp_002    # tell alembic "everything up to mvp_002 is already applied"
alembic upgrade head     # apply only the new ones (mvp_002 -> mvp_003)
```

**Lesson:** If you ever `stamp base` manually, remember to `stamp <revision>` next time to skip past existing schema.

---

## 7. Render running stale code

**Symptom:** API returns 500 with a traceback that references a line number that doesn't match the local file. (E.g. local has `r.trending_score` on line 135, but the traceback says line 134.)

**Why:** Render didn't pick up the latest commit, or the cache held an older image.

**Fix:** Manual Deploy → "Clear build cache & deploy" (top right of the service page).

**Lesson:** Always check the deployed commit hash against the latest on `main`/`version-0` when something looks off.

---

## 8. Two Neon DBs that "look the same"

**Symptom:** Local queries return 5,183 repos; Render returns "table does not exist" or empty. The URLs *look* identical.

**Why:** Neon free tier lets you create multiple projects. Accidentally created two. Both have similar-looking URLs.

**Fix:** Identify the canonical one (the one with the data) and use its URL everywhere. **Delete the other** to avoid future confusion.

**Lesson:** One Neon project per repo. Don't make a second one "just to test" — you'll forget which is which. Name them clearly: `reporelay-prod`, `reporelay-staging`, etc.

---

## 9. Pydantic model + raw SQL column drift

**Symptom:** `Could not locate column in row for column 'trending_score'` after adding a new column.

**Why:** `PopularRepo(trending_score=...)` requires the column to be in the SELECT. Updated the model but only updated the "no topic" branch of `/popular`. The "with topic" branch still had the old SELECT.

**Fix:** Update every SQL query that builds the model. Grep for the column name to find them.

**Lesson:** When adding a column, search the codebase for that column name AND for the model it's part of. Any SELECT that maps to the model needs the new column.

---

## 10. Vercel Authentication blocks public access

**Symptom:** `401 Authentication Required` when visiting the deployed URL in a private/incognito window.

**Why:** Vercel protects new deployments with auth by default (Vercel Authentication / Deployment Protection).

**Fix:** Vercel → project → **Settings** → **Deployment Protection** → set to **"Public"**.

**Lesson:** This is on by default. Either turn it off, or document that the URL requires Vercel login.

---

## 11. `noglob` + single quotes for env vars with `?`

**Symptom:** `zsh: no matches found` when running `DATABASE_URL='postgresql://...?sslmode=require' uv run ...`

**Why:** zsh treats `?` as a glob wildcard. The `?sslmode=require` part of the URL triggers glob expansion.

**Fix:** Two options:
- (A) Prefix with `noglob`: `noglob DATABASE_URL='...' uv run ...`
- (B) `set -o noglob` for the session

**Lesson:** Any URL with `?query` parameters needs the `noglob` workaround in zsh. Bash doesn't have this issue.

---

## 12. Port scan timeout on Docker deploys

**Symptom:** Render logs show "Port scan timeout reached, no open ports detected" during the build, but uvicorn eventually starts and the service works.

**Why:** Render's port scanner starts before uvicorn finishes booting (which takes 2-3 min for first deploy because torch + transformers need to be downloaded and cached).

**Fix:** None needed — it's a false alarm. The service comes up after the scan timeout. Don't change the Dockerfile in response.

**Lesson:** Read the FULL logs, not just the first error. The "port scan timeout" appears in every slow Docker build.

---

## Deployment checklist (use this for next deploy)

- [ ] All API URLs in frontend use `import.meta.env.PUBLIC_API_URL` (or equivalent)
- [ ] `astro.config.mjs` has the right adapter
- [ ] `render.yaml` references a `Dockerfile` that handles the workspace
- [ ] `DATABASE_URL` is set on Render; URL uses `postgresql+psycopg://` prefix
- [ ] `migrations/env.py` reads `DATABASE_URL` from env, not from ini
- [ ] Alembic version is in sync with actual schema (`alembic current` matches `alembic heads`)
- [ ] Vercel: Deployment Protection is set to "Public"
- [ ] All SELECT queries that build a pydantic model include every column the model requires
- [ ] Use `noglob` (or `set -o noglob`) in zsh for any URL with `?query` params
- [ ] Trigger "Clear build cache & deploy" on Render if logs reference a line number that doesn't match local code
