# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Download models
ARG TTS_MODEL=facebook/mms-tts-quz
ARG STT_MODEL=openai/whisper-small
ENV MODEL_PATH=/app/models
ENV HF_HOME=/app/models

RUN python -c "\
from transformers import AutoTokenizer, VitsModel, WhisperProcessor, WhisperForConditionalGeneration; \
print('Downloading TTS model...'); \
AutoTokenizer.from_pretrained('${TTS_MODEL}', cache_dir='/app/models'); \
VitsModel.from_pretrained('${TTS_MODEL}', cache_dir='/app/models'); \
print('Downloading STT model...'); \
WhisperProcessor.from_pretrained('${STT_MODEL}', cache_dir='/app/models'); \
WhisperForConditionalGeneration.from_pretrained('${STT_MODEL}', cache_dir='/app/models'); \
print('Models downloaded.')"

# Stage 2: Runner
FROM python:3.12-slim as runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/models

WORKDIR $APP_HOME

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and models
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models /app/models

# Copy application code
COPY app/ ./app/

# Create non-root user with home directory set to /app
RUN addgroup --system speech && \
    adduser --system --group --home /app speech && \
    chown -R speech:speech $APP_HOME

USER speech

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
