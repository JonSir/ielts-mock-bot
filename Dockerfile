FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
# Make CTranslate2 use a safe baseline so it runs on all CPUs (slower, but compatible)
ENV CT2_USE_BASELINE=1

# Install ffmpeg (audio) and libgomp1 (required by CTranslate2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Render will provide a PORT; use it
ENV PORT=10000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
