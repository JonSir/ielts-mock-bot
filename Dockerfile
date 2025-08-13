FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1

# Install ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Render exposes a PORT env var; use it
ENV PORT=10000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
