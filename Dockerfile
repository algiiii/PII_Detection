# syntax=docker/dockerfile:1
# Portable Linux image for the PII detection app: all dependencies (incl. the
# heavy ML stack — torch/gliner/spaCy) install cleanly here, unlike the x86/Rosetta
# host venv. The source is bind-mounted at run time (see docker-compose.yml), so
# code changes need no rebuild.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

# libgomp1: OpenMP runtime required by torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY pii_detection ./pii_detection

# Install the package with every extra (runtime + dev), then the Italian spaCy
# model. The pip cache is mounted, so torch/spaCy are not re-downloaded when the
# dependency set changes across rebuilds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e '.[ropa,review,presidio,llm,ner,dev,extraction,eval]' \
    && pip install "https://github.com/explosion/spacy-models/releases/download/it_core_news_lg-3.8.0/it_core_news_lg-3.8.0-py3-none-any.whl"

# GLiNER and other HuggingFace models are downloaded on first use into HF_HOME,
# a named volume, so they persist and are not re-downloaded.

# Kept alive so the container can be exec'd into for tests/benchmarks; override
# the command to run the review app, the ingestion CLI, etc.
CMD ["sleep", "infinity"]
