# Interject web app (assistant + Drive) in one image: the React frontend is
# built in a Node stage and served by the FastAPI backend on a single port, so
# pages, API and websockets share one address (see server/app/main.py).
#
#   docker build -t interject .
#   docker run --rm -p 8000:8000 interject                     # offline demo engine, no keys
#   docker run --rm -p 8000:8000 -e LLM_API_KEY=gsk_... ... interject   # with a real model (README)
#
# Then open http://localhost:8000. The FDB-v3 benchmark is not in this image:
# it runs from scripts/run_fdb_v3.sh (LiveKit agent, audio models, ffmpeg).

# -- 1. frontend ------------------------------------------------------------------
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# -- 2. backend + built frontend ---------------------------------------------------
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000
WORKDIR /app/server
COPY server/requirements.txt ./
RUN pip install -r requirements.txt
COPY server/app ./app
COPY server/corpus ./corpus
COPY --from=web /web/dist /app/web/dist

# Accounts live in server/data (SQLite); mount a volume there to keep them.
RUN useradd --create-home --uid 10001 interject \
    && mkdir -p /app/server/data \
    && chown -R interject:interject /app/server/data
USER interject
VOLUME ["/app/server/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/api/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
