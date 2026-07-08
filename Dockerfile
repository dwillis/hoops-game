FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# App code and the fitted data files the engine reads at startup.
# data/raw (400MB of raw ingest data) is excluded via .dockerignore.
COPY README.md ./
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "hoops", "serve", "--host", "0.0.0.0", "--port", "8000"]
