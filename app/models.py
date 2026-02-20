"""
Pydantic request/response models for orbe-speech API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Text-to-Speech synthesis request."""

    text: str = Field(..., min_length=1, max_length=4000)
    language: str = Field(
        default="quz",
        description="ISO 639-1/3 language code (es, quz, en).",
    )
    provider: str = Field(
        default="auto",
        description="TTS provider: 'auto', 'mms', 'elevenlabs', 'polly'.",
    )
    voice_id: str | None = Field(
        default=None,
        description="Specific voice ID for ElevenLabs or Polly.",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional provider-specific API Key (e.g. ElevenLabs XI-API-KEY) for BYOK scenarios.",
    )
    output_format: str = Field(
        default="mp3",
        description="Audio output format. Supported: wav, mp3, pcm.",
    )


class TTSResponse(BaseModel):
    """Metadata returned when audio is not streamed (informational only)."""

    language: str
    model_used: str
    sample_rate: int
    duration_seconds: float


class STTRequest(BaseModel):
    """
    Speech-to-Text transcription request (JSON body variant).
    For binary upload use the multipart form endpoint.
    """

    language: str | None = Field(
        default=None,
        description="Hint language code. If None, Whisper auto-detects.",
    )


class STTResponse(BaseModel):
    """Transcription result."""

    text: str
    language_detected: str | None = None
    confidence: float | None = None


class HealthResponse(BaseModel):
    status: str
    tts_model_loaded: bool
    stt_model_loaded: bool
    tts_language: str
