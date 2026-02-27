"""
orbe-speech — Specialized speech microservice.

Provides native Quechua TTS (facebook/mms-tts-quz) and multilingual STT
(openai/whisper-small) as HTTP endpoints consumed by the orbe AI microservice.

Models are loaded once at startup via FastAPI lifespan and held as singletons.
All CPU inference runs in a thread executor — the event loop is never blocked.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .config import get_settings
from .models import HealthResponse, STTResponse, TTSRequest, TTSResponse
from .services.stt_service import WhisperSTTService
from .services.tts_service import MmsTTSService

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton service holders ────────────────────────────────────────────────

_tts: MmsTTSService | None = None
_stt: WhisperSTTService | None = None


from .services.providers import ElevenLabsProvider, PollyProvider

# ... (global providers)
_elevenlabs: ElevenLabsProvider | None = None
_polly: PollyProvider | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    global _tts, _stt, _elevenlabs, _polly
    logger.info("Loading speech models (this takes ~30s on first run)...")

    # Local models (heavy)
    _tts = MmsTTSService(
        model_name=settings.tts_model_name,
        model_path=settings.model_path,
        language=settings.tts_language,
    )
    _stt = WhisperSTTService(
        model_name=settings.stt_model_name,
        model_path=settings.model_path,
    )

    # Cloud providers (lightweight clients)
    try:
        if settings.elevenlabs_api_key:
            _elevenlabs = ElevenLabsProvider()
            logger.info("ElevenLabs provider enabled")
    except Exception as e:
        logger.warning(f"Failed to init ElevenLabs: {e}")

    try:
        _polly = PollyProvider()
        logger.info("AWS Polly provider enabled")
    except Exception as e:
        logger.warning(f"Failed to init AWS Polly: {e}")

    logger.info("Speech models ready")
    yield

    _tts = None
    _stt = None
    _elevenlabs = None
    _polly = None


app = FastAPI(
    title="orbe-speech",
    version=settings.version,
    description="Native Quechua TTS + multilingual STT for the orbe AI platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Auth helper ──────────────────────────────────────────────────────────────

def _require_api_key(x_api_key: str | None) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header",
        )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        tts_model_loaded=_tts is not None,
        stt_model_loaded=_stt is not None,
        tts_language=settings.tts_language,
    )

@app.post("/tts", tags=["Text-to-Speech"])
async def text_to_speech(
    request: TTSRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Response:
    """
    Synthesize text to audio using MMS (Quechua), ElevenLabs (Cloned), or Polly (Fallback).
    """
    _require_api_key(x_api_key)

    if len(request.text) > settings.tts_max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {settings.tts_max_chars} characters",
        )

    # 1. Determine Provider
    provider = request.provider.lower()
    logger.info(f"TTS Request: text='{request.text[:20]}...', lang={request.language}, provider={provider}")
    
    # Auto-selection logic
    if provider == "auto":
        if request.language == "quz" or request.language == "qu":
            provider = "mms"
        elif request.language.startswith("es"):
            if _elevenlabs and settings.elevenlabs_api_key:
                provider = "elevenlabs"
            else:
                provider = "polly"
        else:
            provider = "polly" # Fallback for other languages

    # 2. Execute Synthesis
    try:
        audio_bytes = None
        content_type = "audio/mpeg"

        if provider == "mms":
            if _tts is None:
                 raise HTTPException(status_code=503, detail="MMS model not loaded")
            
            # Apply pitch shift (lower sample rate) to simulate Male voice for Quechua
            # Default MMS voice is often female/neutral. 
            # Lowering sample rate by ~15% makes it deeper (Male-like).
            # 22050Hz -> ~18742Hz
            # sample_rate_override = int(_tts.sample_rate * 0.75)
            # REVERT: User requested natural voice, no robotic pitch shift
            sample_rate_override = None
            
            # MMS returns wav
            audio_bytes = await _tts.synthesize(
                request.text, 
                output_format=request.output_format,
                sample_rate_override=sample_rate_override
            )
            content_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"

        elif provider == "elevenlabs":
            if _elevenlabs is None:
                # Allow instantiation if per-request key is provided even if global key is missing
                if request.api_key:
                     _elevenlabs = ElevenLabsProvider()
                else:
                     raise HTTPException(status_code=503, detail="ElevenLabs provider not configured globally and no key provided")
            
            audio_bytes = _elevenlabs.synthesize(request.text, voice_id=request.voice_id, api_key=request.api_key)
            content_type = "audio/mpeg"

        elif provider == "polly":
            if _polly is None:
                raise HTTPException(status_code=503, detail="AWS Polly provider not configured")
            audio_bytes = _polly.synthesize(request.text, voice_id=request.voice_id)
            content_type = "audio/mpeg"
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": "inline; filename=speech.audio"},
        )

    except Exception as e:
        logger.error(f"TTS Error ({provider}): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/stream", tags=["Text-to-Speech"])
async def text_to_speech_stream(
    request: TTSRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """
    Stream TTS audio — for MMS the whole waveform is generated at once
    then streamed in chunks. Useful for large responses.
    """
    _require_api_key(x_api_key)

    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    # Apply pitch shift (lower sample rate) to simulate Male voice for Quechua
    sample_rate_override = int(_tts.sample_rate * 0.85)
    
    audio_bytes = await _tts.synthesize(
        request.text, 
        output_format=request.output_format,
        sample_rate_override=sample_rate_override
    )

    async def _chunks():
        chunk_size = 8192
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    return StreamingResponse(
        _chunks(),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@app.post("/stt", response_model=STTResponse, tags=["Speech-to-Text"])
async def speech_to_text(
    audio: Annotated[UploadFile, File(description="Audio file (WAV, WEBM, MP3, OGG)")],
    language: Annotated[str | None, Form(description="Language hint, e.g. 'qu', 'es'")] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> STTResponse:
    """
    Transcribe audio to text using Whisper.

    Whisper auto-detects language if not provided.
    Supports Quechua (qu), Spanish (es), and 98 other languages.
    """
    _require_api_key(x_api_key)

    if _stt is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        result = await _stt.transcribe(audio_bytes, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return STTResponse(
        text=result["text"],
        language_detected=result.get("language_detected"),
    )
