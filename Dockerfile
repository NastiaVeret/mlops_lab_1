# syntax=docker/dockerfile:1
# Multi-stage: важкі збірки та pip у builder, у фіналі лише slim + venv + код.
ARG PYTHON_VERSION=3.11

# --- Етап 1: встановлення залежностей (повний базовий образ, зручно для wheels/збірки) ---
FROM python:${PYTHON_VERSION}-bookworm AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# DVC і MLflow явно; решта з requirements.txt (pandas, sklearn, hydra, optuna, jupyter тощо).
RUN pip install --upgrade pip setuptools wheel && \
    pip install dvc mlflow && \
    pip install -r requirements.txt

# --- Етап 2: легкий runtime ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Git — для DVC; шрифти/рендер — для matplotlib/seaborn без дисплея.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

COPY . .

# Приклад: docker run --rm -v "$PWD":/app IMAGE python src/train.py ...
CMD ["/bin/bash"]
