set dotenv-load := true

default:
    @just --list

sync:
    uv sync

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

migrate:
    uv run --package reporelay-mvp alembic -c packages/mvp/alembic.ini upgrade head

new-migration MESSAGE:
    uv run --package reporelay-mvp alembic -c packages/mvp/alembic.ini revision --autogenerate -m "{{MESSAGE}}"

mvp *ARGS:
    uv run --package reporelay-mvp reporelay-mvp {{ARGS}}

lint:
    uv run ruff check .

fmt:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run mypy apps packages

test:
    uv run pytest

check: lint typecheck test

clean:
    rm -rf .venv node_modules **/__pycache__ **/dist **/build
