set dotenv-load := true

default:
    @just --list

sync:
    uv sync
    pnpm install

up:
    docker compose -f infra/docker-compose.yml up -d

down:
    docker compose -f infra/docker-compose.yml down

logs:
    docker compose -f infra/docker-compose.yml logs -f

psql:
    docker compose -f infra/docker-compose.yml exec postgres psql -U reporelay -d reporelay

api:
    uv run --package reporelay-api uvicorn reporelay_api.main:app --reload --port 8000

mvp-api:
    uv run --package reporelay-mvp-api uvicorn reporelay_mvp_api.main:app --reload --port 8001

mvp-migrate:
    uv run --package reporelay-mvp alembic -c packages/mvp/alembic.ini upgrade head

mvp-new-migration MESSAGE:
    uv run --package reporelay-mvp alembic -c packages/mvp/alembic.ini revision --autogenerate -m "{{MESSAGE}}"

site:
    pnpm --filter reporelay-site dev

ingest *ARGS:
    uv run --package reporelay-ingest reporelay-ingest {{ARGS}}

mvp *ARGS:
    uv run --package reporelay-mvp reporelay-mvp {{ARGS}}

lint:
    uv run ruff check .
    pnpm -r exec biome check .

fmt:
    uv run ruff format .
    uv run ruff check --fix .
    pnpm -r exec biome check --write .

typecheck:
    uv run mypy apps packages
    pnpm -r exec tsc --noEmit

test:
    uv run pytest

check: lint typecheck test

clean:
    rm -rf .venv node_modules **/node_modules **/.venv **/__pycache__ **/dist **/build
