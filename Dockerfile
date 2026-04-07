# Dockerfile
# ----------
# Lightweight container for the Digital Workspace Architect OpenEnv server.
#
# Build:  docker build -t workspace-architect .
# Run:    docker run -p 7860:7860 workspace-architect
# Docs:   http://localhost:7860/docs

# ---------------------------------------------------------------------------
# Stage 1 – dependency installation (leverage layer caching)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy dependency manifest first so pip layer is cached unless it changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 – runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security: run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy application source.
COPY schemas.py env.py server.py openenv.yaml ./

# Ensure correct ownership.
RUN chown -R appuser:appuser /app

USER appuser

# Port the OpenEnv spec declares.
EXPOSE 7860

# Health-check so Docker / Compose knows when the server is ready.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start Uvicorn.
# - --workers 1  keeps state in a single process (in-memory env is not
#   thread-safe across workers without extra locking).
# - --no-access-log keeps container logs clean; remove for debugging.
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--no-access-log"]
