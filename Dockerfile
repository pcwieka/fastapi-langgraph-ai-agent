FROM python:3.11-slim

WORKDIR /app

# Install uv - fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies in a separate layer - cache hit when deps don't change
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    langgraph \
    langgraph-checkpoint-sqlite \
    aiosqlite \
    langchain-openai \
    python-dotenv \
    "chromadb>=1.5.9,<2.0.0"

# Copy application code, the offline embedding indexer, and the product catalog
COPY app/ app/
COPY embedding/ embedding/
COPY data/products.json data/products.json

# Run as non-root user
RUN useradd --create-home appuser && mkdir -p data && chown appuser:appuser data
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
