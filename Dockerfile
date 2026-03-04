# ============================================================
#  Multi-stage Dockerfile
#  Stage 1 — trainer   : runs train.py and saves the model
#  Stage 2 — inference : serves the FastAPI app
# ============================================================

# ── Base image shared by both stages ────────────────────────
FROM python:3.11-slim AS base

# Keeps Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install uv (fast pip replacement used in this project)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# ── Training stage ───────────────────────────────────────────
FROM base AS trainer

# Install all dependencies (including dev/train extras)
RUN uv sync --frozen

# Copy source + data
COPY src/ ./src/
COPY data/ ./data/

# Default: run the training script
CMD ["uv", "run", "python", "src/train.py"]


# ── Inference stage ──────────────────────────────────────────
FROM base AS inference

# Install only runtime dependencies
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/

# Expose FastAPI port
EXPOSE 8000

# Environment variables (override at runtime)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_URI=models:/lstm-pollution/Production

# Start the API server
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
