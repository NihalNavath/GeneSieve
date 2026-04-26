# GeneSieve Environment — Docker Image
# Builds a lightweight FastAPI/OpenEnv server (no ML/training deps).

# ── Stage 1: Install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# Copy only the dependency files first (better layer caching)
COPY server/requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml

# Install all runtime dependencies into a prefix we can copy
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app/env

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Enable the web interface for OpenEnv (if applicable)
ENV ENABLE_WEB_INTERFACE=true

# Copy the full project
COPY . /app/env

# PYTHONPATH so `from models import ...` and `from server.XXX import ...` resolve correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
