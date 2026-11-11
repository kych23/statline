FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first so this layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY src ./src
COPY config ./config
COPY scripts/serve.py ./scripts/serve.py
COPY data/projections.parquet data/projections_2026_w0.parquet ./data/
COPY data/raw/adp_2026.parquet ./data/raw/adp_2026.parquet
RUN uv sync --locked --no-default-groups

RUN useradd --system --no-create-home statline
USER statline

ENV PORT=8080
CMD ["/app/.venv/bin/python", "scripts/serve.py"]
