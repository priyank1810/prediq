FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by native pip packages
# gcc/g++ for scikit-learn, numpy, etc.; libffi for cffi; git for some pip installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for runtime data and models
RUN mkdir -p /app/data /app/saved_models /app/logs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
