# ==============================================================================
# Solar Panel AI Inspection System - FastAPI Backend Dockerfile
# Python 3.12 compatible, optimized for CPU inference with PyTorch & OpenCV
# ==============================================================================

FROM python:3.12-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install OS-level dependencies required for OpenCV and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels first for caching and fast builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, model checkpoints, and initial data directory
COPY configs /app/configs
COPY src /app/src
COPY models /app/models
COPY data /app/data

# Ensure data directory exists for SQLite database persistence
RUN mkdir -p /app/data

# Expose backend port
EXPOSE 8000

# Healthcheck definition targeting the canonical health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI application with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
