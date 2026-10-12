FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cosmetic_rec ./cosmetic_rec
COPY data ./data
COPY frontend ./frontend

# Train and register the first model at build time, so the container starts instantly. A mounted registry volume is
# seeded from this content the first time it is used.
RUN mkdir -p models/registry var && python -m cosmetic_rec.train \
    && useradd --create-home app && chown -R app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Listens on $PORT (hosts such as Render set it; default 8000) with WEB_CONCURRENCY workers (default 2; use 1 on a
# 512 MB host). The metrics directory is emptied at start so counters from a previous run are not mixed in.
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn cosmetic_rec.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-2}"]
