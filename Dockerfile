FROM python:3.11-slim

WORKDIR /app

# Install uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies in a separate layer — cache hit when deps don't change
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    langgraph \
    langgraph-checkpoint-sqlite \
    langchain-openai \
    python-dotenv

# Copy application code
COPY app/ app/

# Create data directory for SQLite checkpointer (mounted as volume in compose)
RUN mkdir -p data && chown appuser:appuser data

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
