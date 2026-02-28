"""
orbe-speech configuration — loaded from environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    app_name: str = "orbe-speech"
    debug: bool = False
    version: str = "1.0.0"
    log_level: str = "INFO"

    # Model storage — pre-downloaded into Docker image at build time
    model_path: str = "/app/models"

    # TTS — MMS (Massively Multilingual Speech)
    # Supports language-specific models: facebook/mms-tts-{lang_code}
    # Default language served by this instance
    tts_language: str = "spa"
    tts_model_name: str = "facebook/mms-tts-spa"
    # Max input characters to prevent abuse / OOM
    tts_max_chars: int = 4000

    # STT — Whisper
    stt_model_name: str = "openai/whisper-small"
    # Expected sample rate for audio input
    stt_sample_rate: int = 16000

    # Security — simple shared-secret API key (same pattern as orbe)
    api_key: str = Field(default="speech-dev-key-2024")

    # ElevenLabs
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # Default generic voice

    # AWS Polly
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # MiniMax TTS (Voice Cloning)
    minimax_api_key: str | None = None
    minimax_group_id: str | None = None
    minimax_base_url: str = "https://api.minimax.chat/v1/text_to_speech"

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8003", "http://orbe-api:8003"]
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
