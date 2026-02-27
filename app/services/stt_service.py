"""
Whisper STT service.

Uses openai/whisper-small via HuggingFace transformers pipeline.
Whisper was trained on multilingual data including Quechua audio,
making it the correct choice for Quechua STT (not supported by AWS Transcribe).

Audio input accepted as raw bytes (WAV/WEBM/MP3) — converted via soundfile.
"""

from __future__ import annotations

import asyncio
import io
import logging

import numpy as np
import soundfile as sf
from transformers import pipeline

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz


class WhisperSTTService:
    """
    Speech-to-Text using openai/whisper-small.

    Handles Quechua, Spanish, and 99 other languages with auto-detection.
    Runs inference in a thread executor (CPU-bound, non-blocking).
    """

    def __init__(self, model_name: str, model_path: str) -> None:
        logger.info(f"Loading Whisper STT model: {model_name}")
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device="cpu",
        )
        logger.info(f"Whisper STT model loaded: {model_name}")

    def _transcribe_sync(self, audio_array: np.ndarray, language: str | None) -> dict:
        """Run Whisper inference synchronously (for executor)."""
        generate_kwargs: dict = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        result = self._pipe(
            {"array": audio_array, "sampling_rate": _TARGET_SAMPLE_RATE},
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        return result  # type: ignore

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio (WAV, WEBM, MP3, OGG, etc.)
            language: BCP-47 hint (e.g. "qu", "es"). None = auto-detect.

        Returns:
            {"text": "...", "language_detected": "..."}
        """
        audio_array = self._load_audio(audio_bytes)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._transcribe_sync, audio_array, language
        )
        return {
            "text": (result.get("text") or "").strip(),
            "language_detected": result.get("language"),
        }

    @staticmethod
    def _load_audio(audio_bytes: bytes) -> np.ndarray:
        """Load audio bytes and resample to 16kHz mono float32."""
        buf = io.BytesIO(audio_bytes)
        
        # Try to read directly with soundfile
        try:
            audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        except Exception as e:
            # If soundfile fails (e.g., WebM format), try ffmpeg conversion
            logger.info(f"Soundfile failed, attempting ffmpeg conversion: {e}")
            try:
                import subprocess
                import tempfile
                
                # Write input to temp file
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
                    tmp_in.write(audio_bytes)
                    tmp_in_path = tmp_in.name
                
                # Convert to WAV using ffmpeg
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                    tmp_out_path = tmp_out.name
                
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", tmp_in_path,
                        "-ar", "16000",  # 16kHz
                        "-ac", "1",      # mono
                        "-f", "wav",
                        tmp_out_path
                    ],
                    capture_output=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg conversion failed: {result.stderr.decode()}")
                
                # Read converted WAV
                with open(tmp_out_path, "rb") as f:
                    wav_bytes = f.read()
                
                # Cleanup temp files
                import os
                os.unlink(tmp_in_path)
                os.unlink(tmp_out_path)
                
                # Read with soundfile
                audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
                
            except Exception as ffmpeg_error:
                logger.error(f"FFmpeg conversion failed: {ffmpeg_error}")
                raise ValueError("Unsupported audio format. WebM conversion failed.")

        # Convert stereo to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != _TARGET_SAMPLE_RATE:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(int(sr), _TARGET_SAMPLE_RATE)
            audio = resample_poly(audio, _TARGET_SAMPLE_RATE // g, sr // g)

        return audio.astype(np.float32)
