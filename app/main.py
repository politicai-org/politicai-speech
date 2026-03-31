"""
orbe-speech — Specialized speech microservice.

Provides native Quechua TTS (facebook/mms-tts-quz) and multilingual STT
(openai/whisper-small) as HTTP endpoints consumed by the orbe AI microservice.

Models are loaded once at startup via FastAPI lifespan and held as singletons.
All CPU inference runs in a thread executor — the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import io
import logging
import importlib.util
import os
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
import subprocess
from typing import Annotated, AsyncGenerator

import grpc
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .config import get_settings
from .models import (
    HealthResponse,
    STTResponse,
    TTSRequest,
    VoiceCloneResponse,
)
from .services.providers import ElevenLabsProvider, MinimaxProvider, PollyProvider
from .services.stt_service import WhisperSTTService
from .services.tts_service import MmsTTSService
from .orbe_speech_pb2 import AudioChunk, SynthesizeRequest, TranscriptEvent, TranscribeResponse
from .orbe_speech_pb2_grpc import SpeechServiceServicer, add_SpeechServiceServicer_to_server

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton service holders ────────────────────────────────────────────────

_tts: MmsTTSService | None = None
_stt: WhisperSTTService | None = None

# ... (global providers)
_elevenlabs: ElevenLabsProvider | None = None
_polly: PollyProvider | None = None
_minimax: MinimaxProvider | None = None
_grpc_server: grpc.aio.Server | None = None


def _require_grpc_api_key(context: grpc.aio.ServicerContext) -> None:
    api_key = ""
    for k, v in context.invocation_metadata():
        if str(k).lower() == "x-api-key":
            api_key = v
            break
    if api_key != settings.api_key:
        context.abort(grpc.StatusCode.PERMISSION_DENIED, "Invalid or missing X-API-Key")


class _GrpcSpeechService(SpeechServiceServicer):
    async def Transcribe(self, request, context):
        _require_grpc_api_key(context)
        if _stt is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "STT model not loaded")
        language = (request.language or "").strip() or None
        result = await _stt.transcribe(request.audio, language=language)
        return TranscribeResponse(
            text=result["text"],
            language_detected=result.get("language_detected") or "",
        )

    async def TranscribeStream(self, request_iterator, context):
        _require_grpc_api_key(context)
        if _stt is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "STT model not loaded")

        import io
        import time
        import wave

        audio_format = None
        sample_rate = 16000
        buf = bytearray()
        last_emitted = ""
        last_emit_ts = 0.0
        logging.getLogger("uvicorn.error").info("STT gRPC stream started")

        partial_min_seconds = settings.stt_partial_min_seconds
        partial_interval_seconds = settings.stt_partial_interval_seconds
        window_seconds = settings.stt_partial_window_seconds

        def _to_wav_bytes(pcm_bytes: bytes, sr: int) -> bytes:
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm_bytes)
            return wav_buf.getvalue()

        async def _emit_partial(pcm_bytes: bytes):
            nonlocal last_emitted, last_emit_ts
            if not pcm_bytes:
                return
            wav_bytes = _to_wav_bytes(pcm_bytes, sample_rate)
            result = await _stt.transcribe(wav_bytes, language=None)
            text = (result.get("text") or "").strip()
            if not text or text == last_emitted:
                return
            last_emitted = text
            last_emit_ts = time.monotonic()
            yield TranscriptEvent(
                text=text,
                is_final=False,
                language_detected=result.get("language_detected") or "",
            )

        async for msg in request_iterator:
            if msg.sample_rate:
                sample_rate = msg.sample_rate
            if msg.format:
                audio_format = msg.format
            if msg.data:
                buf.extend(msg.data)

                if (audio_format or "").lower() in ("pcm16", "pcm_s16le", "s16le"):
                    min_bytes = int(sample_rate * 2 * partial_min_seconds)
                    if len(buf) >= min_bytes:
                        now = time.monotonic()
                        if now - last_emit_ts >= partial_interval_seconds:
                            window_bytes = int(sample_rate * 2 * window_seconds)
                            pcm_window = bytes(buf[-window_bytes:])
                            async for ev in _emit_partial(pcm_window):
                                yield ev

            if msg.last:
                break

        audio_bytes = bytes(buf)
        if not audio_bytes:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty audio stream")

        if (audio_format or "").lower() in ("pcm16", "pcm_s16le", "s16le"):
            audio_bytes = _to_wav_bytes(audio_bytes, sample_rate)

        result = await _stt.transcribe(audio_bytes, language=None)
        text = (result.get("text") or "").strip()
        if text and text != last_emitted:
            yield TranscriptEvent(
                text=text,
                is_final=False,
                language_detected=result.get("language_detected") or "",
            )

        yield TranscriptEvent(
            text=text,
            is_final=True,
            language_detected=result.get("language_detected") or "",
        )
        logging.getLogger("uvicorn.error").info("STT gRPC stream finished")

    async def SynthesizeStream(self, request: SynthesizeRequest, context):
        _require_grpc_api_key(context)
        text = (request.text or "").strip()
        if not text:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty text")

        provider = (request.provider or "auto").strip()
        language = (request.language or "es").strip()
        output_format = (request.output_format or "mp3").strip()

        voice_id = (request.voice_id or "").strip() or None
        api_key = (request.api_key or "").strip() or None

        if provider == "auto":
            if language.startswith("qu"):
                provider = "mms"
            elif voice_id and api_key:
                provider = "elevenlabs"
            else:
                provider = "polly"

        async for chunk in _stream_with_provider(
            text=text,
            provider=provider,
            language=language,
            output_format=output_format,
            voice_id=voice_id,
            api_key=api_key,
            sample_rate_override=None,
        ):
            if chunk:
                yield AudioChunk(data=chunk, format=output_format, sample_rate=0, last=False)
        yield AudioChunk(data=b"", format=output_format, sample_rate=0, last=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    global _tts, _stt, _elevenlabs, _polly, _minimax, _grpc_server
    logger.info("Loading speech models (this takes ~30s on first run)...")

    model_path = Path(settings.model_path)
    cache_path = Path(os.environ.get("TRANSFORMERS_CACHE", str(model_path / "hub")))
    hf_home = Path(os.environ.get("HF_HOME", str(model_path)))

    try:
        model_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path))
    except PermissionError:
        fallback_base = Path("/tmp/hf-cache")
        fallback_home = fallback_base / "home"
        fallback_cache = fallback_base / "hub"
        fallback_home.mkdir(parents=True, exist_ok=True)
        fallback_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(fallback_home)
        os.environ["TRANSFORMERS_CACHE"] = str(fallback_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(fallback_cache)

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
    server = grpc.aio.server()
    add_SpeechServiceServicer_to_server(_GrpcSpeechService(), server)
    port = server.add_insecure_port("0.0.0.0:50051")
    await server.start()
    _grpc_server = server
    logging.getLogger("uvicorn.error").info(f"gRPC server listening on 0.0.0.0:{port}")
    yield

    if _grpc_server is not None:
        await _grpc_server.stop(grace=0)
        _grpc_server = None

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
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    pydub_ok = importlib.util.find_spec("pydub") is not None

    return HealthResponse(
        status="ok",
        tts_model_loaded=_tts is not None,
        stt_model_loaded=_stt is not None,
        tts_language=settings.tts_language,
        ffmpeg_available=ffmpeg_path is not None,
        pydub_available=pydub_ok,
    )

def _mp3_to_wav(mp3_bytes: bytes, sample_rate: int | None = None) -> bytes:
    if not mp3_bytes:
        return b""
    try:
        from pydub import AudioSegment  # type: ignore

        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        if sample_rate:
            seg = seg.set_frame_rate(sample_rate)
        seg = seg.set_channels(1)
        out = io.BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
        tmp_in.write(mp3_bytes)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        cmd = ["ffmpeg", "-y", "-i", tmp_in_path, "-ac", "1"]
        if sample_rate:
            cmd += ["-ar", str(sample_rate)]
        cmd += ["-f", "wav", tmp_out_path]
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace")[:500])
        with open(tmp_out_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_in_path)
        except Exception:
            pass
        try:
            os.unlink(tmp_out_path)
        except Exception:
            pass


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
    global _elevenlabs, _minimax, _polly, _tts

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
        
        audio_bytes = instance.synthesize(
            text, voice_id=voice_id, api_key=api_key, output_format=output_format
        )
        if output_format == "wav":
            audio_bytes = _mp3_to_wav(audio_bytes, sample_rate=sample_rate_override)
            content_type = "audio/wav"
        elif output_format == "pcm":
            content_type = "audio/l16;rate=16000"
        else:
            content_type = "audio/mpeg"
    
    elif provider == "polly":
        if _polly is None:
            raise HTTPException(status_code=503, detail="AWS Polly provider not configured")
        audio_bytes = _polly.synthesize(
            text, voice_id=voice_id, output_format=output_format
        )
        if output_format == "wav":
            audio_bytes = _mp3_to_wav(audio_bytes, sample_rate=sample_rate_override)
            content_type = "audio/wav"
        elif output_format == "pcm":
            content_type = "audio/l16;rate=16000"
        else:
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
        
        audio_bytes = instance.synthesize(
            text, voice_id=voice_id, api_key=api_key, output_format=output_format
        )
        if output_format == "wav":
            audio_bytes = _mp3_to_wav(audio_bytes, sample_rate=sample_rate_override)
            content_type = "audio/wav"
        elif output_format == "pcm":
            content_type = "audio/l16;rate=16000"
        else:
            content_type = "audio/mpeg"
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    return audio_bytes, content_type


async def _stream_with_provider(
    text: str,
    provider: str,
    language: str,
    output_format: str,
    voice_id: str | None = None,
    api_key: str | None = None,
    sample_rate_override: int | None = None,
) -> AsyncGenerator[bytes, None]:
    """
    Streaming synthesis logic. Yields audio chunks.
    """
    global _elevenlabs, _minimax, _polly, _tts
    
    if provider == "elevenlabs" and output_format != "wav":
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
        
        # Use the true streaming method
        async for chunk in instance.synthesize_stream(
            text, voice_id=voice_id, api_key=api_key, output_format=output_format
        ):
            yield chunk
            
    else:
        # Fallback to full synthesis for other providers (MMS, Polly, MiniMax)
        # This simulates streaming by chunking the result
        audio_bytes, _ = await _synthesize_with_provider(
            text=text,
            provider=provider,
            language=language,
            output_format=output_format,
            voice_id=voice_id,
            api_key=api_key,
            sample_rate_override=sample_rate_override,
        )
        
        chunk_size = 8192
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]


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
    
    logger.info(f"TTS Streaming Request: provider={provider}, format={request.output_format}, voice={request.voice_id}")
    
    if request.output_format == "wav":
        content_type = "audio/wav"
    elif request.output_format == "pcm":
        content_type = "audio/l16;rate=16000"
    else:
        content_type = "audio/mpeg"

    return StreamingResponse(
        _stream_with_provider(
            text=request.text,
            provider=provider,
            language=request.language,
            output_format=request.output_format,
            voice_id=request.voice_id,
            api_key=request.api_key,
            sample_rate_override=sample_rate_override,
        ),
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
