# 3.12 rather than 3.10: the pinned torch / sentence-transformers versions ship
# wheels for it, and it is closest to the interpreter the suite is tested on.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cache the embedding model under the mounted data volume so it survives
    # container restarts instead of being re-downloaded every boot.
    HF_HOME=/app/data/hf_home

WORKDIR /app

# curl is used by the API healthcheck in docker-compose.yml.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/cvs /app/data

# api/ and indexer/ are packages (they carry __init__.py), so `python -m
# indexer.run` resolves from the /app working directory.
COPY api/ /app/api/
COPY indexer/ /app/indexer/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
