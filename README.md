# orbe-speech

Specialized speech microservice providing **native Quechua TTS** and multilingual STT for the orbe AI platform.

## Why a separate service?

| Concern | orbe | orbe-speech |
|---------|------|-------------|
| Runtime | Async I/O (FastAPI) | CPU-bound inference (PyTorch) |
| Memory | ~512 MB | ~4 GB (models + inference) |
| Scaling | Scale with chat load | Scale with TTS/STT request volume |
| Dependencies | Zero ML libs | PyTorch, Transformers, soundfile |

Separating CPU-bound inference from the async orbe service avoids event-loop blocking and allows independent scaling and resource allocation.

## Models

| Task | Model | Language | Size |
|------|-------|----------|------|
| TTS | `facebook/mms-tts-quz` | Cusco Quechua (`quz`) | ~300 MB |
| STT | `openai/whisper-small` | 99 languages incl. Quechua | ~244 MB |

Models are **pre-downloaded at Docker build time** — no cold-start downloads.

## API

```
POST /tts          → Synthesize Quechua text → WAV/MP3 bytes
POST /tts/stream   → Stream audio chunks
POST /stt          → Transcribe audio → text (multipart upload)
GET  /health       → Liveness + model load status
```

All endpoints require `X-API-Key` header matching `API_KEY` env var.

## Quick Start

```bash
docker-compose up -d

# Health check
curl http://localhost:8004/health

# Synthesize Quechua text
curl -X POST http://localhost:8004/tts \
  -H "X-API-Key: speech-dev-key-2024" \
  -H "Content-Type: application/json" \
  -d '{"text": "Allillanchu, imaynallan kachkanki", "language": "quz"}' \
  --output speech.wav

# Transcribe audio
curl -X POST http://localhost:8004/stt \
  -H "X-API-Key: speech-dev-key-2024" \
  -F "audio=@recording.wav" \
  -F "language=qu"
```

## Integration with orbe

Set these env vars in orbe:
```bash
HF_SPEECH_URL=http://orbe-speech:8004   # Docker Compose
HF_SPEECH_API_KEY=speech-dev-key-2024
```

When `HF_SPEECH_URL` is set, the `candidate` voice profile automatically routes `language=qu` requests to this service via `facebook/mms-tts-quz`.

## Extending to other languages

To add another MMS language (e.g. Aymara `ayr`):

1. Deploy a second instance with `TTS_MODEL_NAME=facebook/mms-tts-ayr` and `TTS_LANGUAGE=ayr`
2. Register a new `HuggingFaceSpeechProvider` in orbe pointing to the new URL
3. Add `"ayr": VoiceConfig(provider="hf-speech-ayr", voice_id="ayr", language_code="ayr")` to the voice profile overrides

No code changes required in orbe's core TTS logic.

## Resource Requirements (ECS Fargate)

- **CPU**: 4 vCPU (PyTorch inference is CPU-bound)
- **Memory**: 8 GB (models + inference buffers)
- **Storage**: Docker image ~3 GB (models baked in)
