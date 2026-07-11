FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf_home

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/api /app/indexer /app/cvs /app/data

# Copy application code
COPY api/ /app/api/
COPY indexer/ /app/indexer/

# Expose port
EXPOSE 8000

# Default command runs the API service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
