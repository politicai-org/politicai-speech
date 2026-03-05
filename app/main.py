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
from .models import (
    HealthResponse,
    STTResponse,
    TTSRequest,
    TTSResponse,
    VoiceCloneRequest,
    VoiceCloneResponse,
)
from .services.stt_service import WhisperSTTService
from .services.tts_service import MmsTTSService

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton service holders ────────────────────────────────────────────────

_tts: MmsTTSService | None = None
_stt: WhisperSTTService | None = None


from .services.providers import ElevenLabsProvider, MinimaxProvider, PollyProvider

# ... (global providers)
_elevenlabs: ElevenLabsProvider | None = None
_polly: PollyProvider | None = None
_minimax: MinimaxProvider | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    global _tts, _stt, _elevenlabs, _polly, _minimax
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

    try:
        if settings.minimax_api_key and settings.minimax_group_id:
            _minimax = MinimaxProvider()
            logger.info("MiniMax provider enabled (voice cloning)")
    except Exception as e:
        logger.warning(f"Failed to init MiniMax: {e}")

    logger.info("Speech models ready")
    yield

    _tts = None
    _stt = None
    _elevenlabs = None
    _polly = None
    _minimax = None


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


async def _synthesize_with_provider(
    text: str,
    provider: str,
    language: str,
    output_format: str,
    voice_id: str | None = None,
    api_key: str | None = None,
    sample_rate_override: int | None = None,
) -> tuple[bytes, str]:
    """
    Shared synthesis logic between /tts and /tts/stream.
    
    Returns:
        tuple[bytes, str]: (audio_bytes, content_type)
    """
    audio_bytes = None
    content_type = "audio/mpeg"
    
    if provider == "mms":
        if _tts is None:
            raise HTTPException(status_code=503, detail="MMS model not loaded")
        
        audio_bytes = await _tts.synthesize(
            text,
            output_format=output_format,
            sample_rate_override=sample_rate_override
        )
        content_type = "audio/wav" if output_format == "wav" else "audio/mpeg"
    
    elif provider == "elevenlabs":
        # Use local variable to avoid UnboundLocalError due to potential assignment
        instance = _elevenlabs
        if instance is None:
            if api_key:
                from .services.providers import ElevenLabsProvider
                instance = ElevenLabsProvider()
            else:
                raise HTTPException(
                    status_code=503,
                    detail="ElevenLabs provider not configured globally and no key provided"
                )
        
        audio_bytes = instance.synthesize(text, voice_id=voice_id, api_key=api_key)
        content_type = "audio/mpeg"
    
    elif provider == "polly":
        if _polly is None:
            raise HTTPException(status_code=503, detail="AWS Polly provider not configured")
        audio_bytes = _polly.synthesize(text, voice_id=voice_id)
        content_type = "audio/mpeg"
    
    elif provider == "minimax":
        # Use local variable to avoid UnboundLocalError due to potential assignment
        instance = _minimax
        if instance is None:
            if api_key:
                from .services.providers import MinimaxProvider
                instance = MinimaxProvider()
            else:
                raise HTTPException(
                    status_code=503,
                    detail="MiniMax provider not configured globally and no key provided"
                )
        
        audio_bytes = instance.synthesize(text, voice_id=voice_id, api_key=api_key)
        content_type = "audio/mpeg"
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    return audio_bytes, content_type


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

    # 2. Execute Synthesis (using shared method)
    try:
        # Pitch shift disabled for natural voice (as requested by user)
        sample_rate_override = None
        
        audio_bytes, content_type = await _synthesize_with_provider(
            text=request.text,
            provider=provider,
            language=request.language,
            output_format=request.output_format,
            voice_id=request.voice_id,
            api_key=request.api_key,
            sample_rate_override=sample_rate_override,
        )

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

    # Determine provider (same auto-selection logic as /tts)
    provider = request.provider.lower()
    if provider == "auto":
        if request.language == "quz" or request.language == "qu":
            provider = "mms"
        elif request.language.startswith("es"):
            if _elevenlabs and settings.elevenlabs_api_key:
                provider = "elevenlabs"
            else:
                provider = "polly"
        else:
            provider = "polly"
    
    # Pitch shift disabled for natural voice (consistency with /tts)
    sample_rate_override = None
    
    # Use shared synthesis logic
    audio_bytes, content_type = await _synthesize_with_provider(
        text=request.text,
        provider=provider,
        language=request.language,
        output_format=request.output_format,
        voice_id=request.voice_id,
        api_key=request.api_key,
        sample_rate_override=sample_rate_override,
    )

    async def _chunks():
        chunk_size = 8192
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    return StreamingResponse(
        _chunks(),
        media_type=content_type,
        headers={"Content-Disposition": "inline; filename=speech.audio"},
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


@app.post("/voice/clone", response_model=VoiceCloneResponse, tags=["Voice-Cloning"])
async def clone_voice(
    audio: Annotated[UploadFile, File(description="Audio file for voice cloning (WAV, MP3, 10s-5min)")],
    voice_name: Annotated[str, Form(description="Name for the cloned voice")],
    api_key: Annotated[str | None, Form(description="Optional MiniMax API key override")] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> VoiceCloneResponse:
    """
    Clone a voice using MiniMax API.
    
    Requires:
    - Audio file: 10 seconds to 5 minutes, WAV or MP3
    - Voice name: Descriptive name for the cloned voice
    
    Returns the voice_id that can be used for TTS synthesis.
    """
    _require_api_key(x_api_key)
    
    if _minimax is None:
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="MiniMax provider not configured and no API key provided"
            )
    
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="Audio file too large (max 50MB)")
    
    try:
        if _minimax:
            voice_id = _minimax.clone_voice(audio_bytes, voice_name, api_key=api_key)
        else:
            temp_provider = MinimaxProvider()
            voice_id = temp_provider.clone_voice(audio_bytes, voice_name, api_key=api_key)
        
        logger.info(f"Successfully cloned voice: {voice_name} -> {voice_id}")
        
        return VoiceCloneResponse(
            voice_id=voice_id,
            voice_name=voice_name,
            status="ready"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error(f"Voice cloning failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error during voice cloning: {exc}")
        raise HTTPException(status_code=500, detail="Voice cloning failed")
