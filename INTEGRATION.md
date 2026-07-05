# Integration Guide: politicai-speech

This document explains how to integrate `politicai-speech` into your PoliticAI services.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PoliticAI Core Service                   │
│                    (FastAPI, Async I/O)                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HTTP Client (requests, httpx, aiohttp)             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/gRPC
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              politicai-speech Microservice                  │
│              (FastAPI + PyTorch, CPU-bound)                 │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TTS: facebook/mms-tts-quz (Quechua)                │  │
│  │  STT: openai/whisper-small (99 languages)           │  │
│  │  Voice Cloning: MiniMax API                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Docker Compose Setup

### Local Development

Add to your `docker-compose.local.yml`:

```yaml
services:
  politicai-speech:
    image: politicai-org/politicai-speech:latest
    build:
      context: ../politicai-speech
      dockerfile: Dockerfile
    ports:
      - "8004:8004"
      - "50051:50051"  # gRPC
    environment:
      - API_KEY=speech-dev-key-2024
      - TTS_LANGUAGE=spa
      - TTS_MODEL_NAME=facebook/mms-tts-spa
      - STT_MODEL_NAME=openai/whisper-small
      - LOG_LEVEL=INFO
      - CORS_ORIGINS=["http://localhost:8003", "http://politicai-api:8003"]
    volumes:
      - politicai-speech-models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8004/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  politicai-api:
    # ... your main service
    environment:
      - SPEECH_URL=http://politicai-speech:8004
      - SPEECH_API_KEY=speech-dev-key-2024
    depends_on:
      politicai-speech:
        condition: service_healthy

volumes:
  politicai-speech-models:
```

### Production (AWS ECS)

```yaml
services:
  politicai-speech:
    image: politicai-org/politicai-speech:latest
    ports:
      - "8004:8004"
      - "50051:50051"
    environment:
      - API_KEY=${SPEECH_API_KEY}
      - TTS_LANGUAGE=spa
      - TTS_MODEL_NAME=facebook/mms-tts-spa
      - STT_MODEL_NAME=openai/whisper-small
      - LOG_LEVEL=INFO
      - CORS_ORIGINS=["https://politicai.example.com"]
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## HTTP Integration

### Python (FastAPI)

```python
import httpx
from typing import Optional

class SpeechClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def synthesize(
        self,
        text: str,
        language: str = "qu",
        output_format: str = "mp3"
    ) -> bytes:
        """Synthesize text to speech."""
        response = await self.client.post(
            f"{self.base_url}/tts",
            json={
                "text": text,
                "language": language,
                "output_format": output_format,
                "provider": "auto"
            },
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.content
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> dict:
        """Transcribe audio to text."""
        files = {"audio": ("audio.wav", audio_bytes)}
        data = {}
        if language:
            data["language"] = language
        
        response = await self.client.post(
            f"{self.base_url}/stt",
            files=files,
            data=data,
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()

# Usage in your FastAPI app
from fastapi import FastAPI

app = FastAPI()
speech_client = SpeechClient(
    base_url="http://politicai-speech:8004",
    api_key="speech-dev-key-2024"
)

@app.post("/api/voice/synthesize")
async def synthesize_voice(text: str, language: str = "qu"):
    audio = await speech_client.synthesize(text, language)
    return {"audio": audio, "format": "mp3"}
```

### JavaScript/TypeScript

```typescript
interface SpeechConfig {
  baseUrl: string;
  apiKey: string;
}

class SpeechClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(config: SpeechConfig) {
    this.baseUrl = config.baseUrl;
    this.apiKey = config.apiKey;
  }

  async synthesize(
    text: string,
    language: string = "qu",
    outputFormat: string = "mp3"
  ): Promise<ArrayBuffer> {
    const response = await fetch(`${this.baseUrl}/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey,
      },
      body: JSON.stringify({
        text,
        language,
        output_format: outputFormat,
        provider: "auto",
      }),
    });

    if (!response.ok) {
      throw new Error(`TTS failed: ${response.statusText}`);
    }

    return response.arrayBuffer();
  }

  async transcribe(
    audioBlob: Blob,
    language?: string
  ): Promise<{ text: string; language_detected: string }> {
    const formData = new FormData();
    formData.append("audio", audioBlob);
    if (language) {
      formData.append("language", language);
    }

    const response = await fetch(`${this.baseUrl}/stt`, {
      method: "POST",
      headers: {
        "X-API-Key": this.apiKey,
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`STT failed: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const speechClient = new SpeechClient({
  baseUrl: "http://localhost:8004",
  apiKey: "speech-dev-key-2024",
});

// Synthesize
const audio = await speechClient.synthesize("Allillanchu", "qu");

// Transcribe
const result = await speechClient.transcribe(audioBlob, "qu");
console.log(result.text);
```

## gRPC Integration

For low-latency streaming, use gRPC:

```python
import grpc
from app.orbe_speech_pb2_grpc import SpeechServiceStub
from app.orbe_speech_pb2 import SynthesizeRequest, TranscribeRequest

async def grpc_synthesize():
    async with grpc.aio.secure_channel(
        "politicai-speech:50051",
        grpc.ssl_channel_credentials()
    ) as channel:
        stub = SpeechServiceStub(channel)
        request = SynthesizeRequest(
            text="Allillanchu",
            language="qu",
            provider="mms",
            output_format="mp3"
        )
        response = await stub.SynthesizeStream(request)
        async for chunk in response:
            yield chunk.data
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `speech-dev-key-2024` | Shared secret for API authentication |
| `TTS_LANGUAGE` | `spa` | Default TTS language code |
| `TTS_MODEL_NAME` | `facebook/mms-tts-spa` | HuggingFace model ID for TTS |
| `STT_MODEL_NAME` | `openai/whisper-small` | HuggingFace model ID for STT |
| `MODEL_PATH` | `/app/models` | Path to pre-downloaded models |
| `CORS_ORIGINS` | `["http://localhost:8003"]` | Allowed CORS origins |
| `ELEVENLABS_API_KEY` | (optional) | ElevenLabs API key for voice cloning |
| `AWS_REGION` | `us-east-1` | AWS region for Polly |
| `MINIMAX_API_KEY` | (optional) | MiniMax API key for voice cloning |
| `MINIMAX_GROUP_ID` | (optional) | MiniMax group ID |

## Health Checks

```bash
# HTTP health check
curl http://localhost:8004/health

# Response
{
  "status": "ok",
  "tts_model_loaded": true,
  "stt_model_loaded": true,
  "tts_language": "spa",
  "ffmpeg_available": true,
  "pydub_available": true
}
```

## Troubleshooting

### Models not loading
- Check Docker volume mounts: `docker volume ls | grep politicai-speech-models`
- Verify disk space: `docker exec politicai-speech df -h /app/models`
- Check logs: `docker logs politicai-speech`

### Slow TTS/STT
- Increase CPU allocation (PyTorch is CPU-bound)
- Monitor: `docker stats politicai-speech`

### CORS errors
- Update `CORS_ORIGINS` environment variable
- Restart container: `docker-compose restart politicai-speech`

## Performance Tuning

### For production:
- **CPU**: Allocate 4+ vCPU (PyTorch inference is CPU-bound)
- **Memory**: 8 GB minimum (models + inference buffers)
- **Storage**: 3 GB for Docker image + models
- **Network**: Use gRPC for streaming (lower latency than HTTP)

### Caching:
- Cache synthesized audio by text hash
- Implement Redis cache layer for frequently used phrases

## Support

For issues or questions:
1. Check logs: `docker logs politicai-speech`
2. Review health endpoint: `curl http://localhost:8004/health`
3. Open an issue: https://github.com/politicai-org/politicai-speech/issues
