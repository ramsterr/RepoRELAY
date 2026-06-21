# Deployment Issues

Things to watch for during deploys. One-time setup problems are not listed
— only stuff that keeps biting us.

## 1. Pydantic model + raw SQL drift

When you add a column to a pydantic model, every SQL query that maps
to that model needs the new column. Easy to miss when there are
multiple query branches (e.g. `/popular` with/without topic filter).

**Rule:** after changing a model, grep for the new field name and check
every SELECT that builds that model.

## 2. `noglob` for zsh + Neon URLs

Neon connection strings contain `?sslmode=require`. zsh treats `?` as
a glob, so any inline env var assignment breaks:

```bash
# breaks — "no matches found"
DATABASE_URL='postgresql://.../?sslmode=require' uv run ...

# works
noglob DATABASE_URL='postgresql://.../?sslmode=require' uv run ...
```

**Rule:** use `noglob` (or split into `export` + separate command) for
any URL with `?query` params.

## 3. Render running stale code

If Render's logs reference a line number that doesn't match the local
file, the deployed code is old. `Manual Deploy → Clear build cache &
deploy` forces a fresh build.

**Rule:** when a stack trace looks wrong, check the deployed commit
hash before debugging the code.

## 4. One Neon project per environment

It's easy to spin up a second Neon project "to test" and then forget
which one Render points at. Mismatched DBs cause confusing 500s
(table missing) or empty results (different project).

**Rule:** name Neon projects clearly (`reporelay-prod`, `reporelay-dev`),
keep Render's `DATABASE_URL` matching the prod one, and delete any
abandoned projects.

## 5. Migration state can drift

`alembic stamp` + `alembic upgrade` can leave `alembic_version` saying
"applied" while the actual table is missing. If a query fails with
"relation does not exist" but `alembic current` says the migration ran,
the version table is lying.

**Fix pattern:**
```bash
alembic stamp <last_good_revision>   # tell alembic "skip past this"
alembic upgrade head                 # apply only the new ones
```

**Rule:** if you ever `alembic stamp base` manually, follow up with
`alembic stamp <revision>` to skip past the existing schema.

## 6. Render's port-scan timeout is a false alarm

Slow Docker builds (2-3 min) trigger "No open ports detected" warnings
before uvicorn finishes booting. The service usually comes up after
the scan timeout — don't change the Dockerfile in response.

**Rule:** read the full logs, not just the first error. Look for
"Uvicorn running on" before assuming a failure.
