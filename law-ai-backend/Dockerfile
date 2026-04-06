# Web Backend Dockerfile
# Builds the FastAPI server that serves the Law AI API.
# Runs continuously as a long-lived service.
# Azure Files are mounted at /mnt/chromadb (vectors) and /mnt/data (SQLite).

# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create mount point directories
RUN mkdir -p /mnt/chromadb /mnt/data

EXPOSE 8000

# Use production-grade server settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120", "--loop", "asyncio"]
