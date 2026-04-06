#!/bin/bash
set -e

echo "Copying ChromaDB from Azure Files to local disk..."
cp -r /mnt/chromadb/* /app/local_chromadb/ 2>/dev/null || echo "No ChromaDB data found on mount, using empty local dir"
echo "ChromaDB copy complete."

echo "Setting up local data directory..."
mkdir -p /app/local_data
cp /mnt/data/sophieai.db /app/local_data/sophieai.db 2>/dev/null || echo "No existing DB found, will create fresh"
echo "Data setup complete. Starting uvicorn..."

exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120 --loop asyncio
