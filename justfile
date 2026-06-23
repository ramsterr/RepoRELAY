set dotenv-load := true

default:
    @just --list

sync:
    uv sync
    cd apps/site && pnpm install

up:
    docker compose -f infra/docker-compose.yml up -d postgres

down:
    docker compose -f infra/docker-compose.yml down

logs:
    docker compose -f infra/docker-compose.yml logs -f

psql:
    docker compose -f infra/docker-compose.yml exec postgres psql -U reporelay -d reporelay

api:
    uv run --package reporelay-mvp-api uvicorn reporelay_mvp_api.main:app --reload --port 8001

site:
    cd apps/site && pnpm dev

site-fast:
    cd apps/site && pnpm build:local && pnpm preview:local

dev:
    uv run --package reporelay-mvp-api uvicorn reporelay_mvp_api.main:app --reload --port 8001 &
    sleep 2
    cd apps/site && pnpm dev

build-site:
    cd apps/site && pnpm build

migrate:
    uv run --package reporelay-mvp alembic -c packages/mvp/alembic.ini upgrade head

new-migration MESSAGE:
    uv run --package reporelay-mvp alembic -c packages/mvp/alembic.ini revision --autogenerate -m "{{MESSAGE}}"

seed-and-embed TOPICS="" PER_TOPIC="130" LIMIT="7000":
    @echo "=== REPORE LAY SEED-AND-EMBED ==="
    @if [ -z "{{TOPICS}}" ]; then echo "topics: all 54 defaults"; else echo "topics: {{TOPICS}}"; fi
    @echo "per-topic: {{PER_TOPIC}}   embed-limit: {{LIMIT}}"
    @echo ""
    @echo "--- seeding ---"
    uv run --package reporelay-mvp reporelay-mvp seed-topics --per-topic {{PER_TOPIC}} --topics "{{TOPICS}}"
    @echo ""
    @echo "--- embedding ---"
    uv run --package reporelay-mvp reporelay-mvp embed --limit {{LIMIT}}
    @echo ""
    @echo "=== done ==="

mvp *ARGS:
    uv run --package reporelay-mvp reporelay-mvp {{ARGS}}

lint:
    uv run ruff check .
    cd apps/site && npx astro check 2>/dev/null || true

fmt:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run mypy apps packages

test:
    uv run pytest

check: lint typecheck test

clean:
    rm -rf .venv node_modules **/__pycache__ **/dist **/build apps/site/node_modules apps/site/.astro apps/site/dist
