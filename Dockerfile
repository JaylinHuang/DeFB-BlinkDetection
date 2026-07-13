# DeFB Eye Detection API — Dockerfile
# Multi-stage build: base image with CUDA support for GPU inference

FROM python:3.12-slim AS base

LABEL maintainer="JaylinHuang <2810745803@qq.com>"
LABEL description="DeFB — Non-restricted Blink Detection API Service"

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./requirements.txt
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt

# Copy application code
COPY api/ ./api/
COPY yolo/ ./yolo/
COPY src/ ./src/
COPY configs/ ./configs/

# Create directory for model weights
RUN mkdir -p /app/weights

# Environment variables
ENV MODEL_PATH=/app/weights/best.pt
ENV CONF_THRESHOLD=0.15
ENV IOU_THRESHOLD=0.45
ENV IMAGE_SIZE=640
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
