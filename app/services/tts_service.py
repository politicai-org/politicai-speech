"""
MMS (Massively Multilingual Speech) TTS service.

Uses facebook/mms-tts-{lang_code} models — trained on actual recordings of
each language. For Quechua (quz), this produces correct native pronunciation
unlike cloud TTS providers that lack Quechua support.

Model is loaded once at startup and held in memory (Single Responsibility).
Synthesis is synchronous (PyTorch CPU inference) — exposed as async via
run_in_executor to avoid blocking the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 22050  # MMS models use 22050 Hz

# Supported output format → MIME type
MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}


class MmsTTSService:
    """
    Text-to-Speech using facebook/mms-tts-{language} models.

    Loads the model for one language at initialization.
    Add more languages by creating additional instances (or a pool pattern).
    """

    def __init__(self, model_name: str, model_path: str, language: str) -> None:
        self.language = language
        self._model_name = model_name
        self._model_path = os.path.join(model_path, model_name.replace("/", "--"))

        logger.info(f"Loading MMS TTS model: {model_name} (language={language})")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_path,
        )
        self._model = VitsModel.from_pretrained(
            model_name,
            cache_dir=model_path,
        )
        self._model.eval()
        logger.info(f"MMS TTS model loaded: {model_name}")

    @property
    def sample_rate(self) -> int:
        return self._model.config.sampling_rate if hasattr(self._model.config, "sampling_rate") else _SAMPLE_RATE

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous CPU inference — run via executor to stay non-blocking."""
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs).waveform
        return output.squeeze().numpy()

    async def synthesize(self, text: str, output_format: str = "wav", sample_rate_override: int | None = None) -> bytes:
        """
        Synthesize Quechua (or target language) text to audio bytes.

        Runs CPU inference in a thread executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        audio_array = await loop.run_in_executor(
            None, self._synthesize_sync, text
        )
        return self._encode(audio_array, output_format, sample_rate_override)

    def _encode(self, audio: np.ndarray, fmt: str, sample_rate_override: int | None = None) -> bytes:
        """Encode float32 numpy waveform to audio bytes."""
        buf = io.BytesIO()
        rate = sample_rate_override if sample_rate_override is not None else self.sample_rate
        
        if fmt == "mp3":
            # Encode to WAV first then transcode — requires ffmpeg via pydub
            # Fallback: return WAV if ffmpeg not available
            try:
                from pydub import AudioSegment  # type: ignore
                wav_buf = io.BytesIO()
                sf.write(wav_buf, audio, rate, format="WAV", subtype="PCM_16")
                wav_buf.seek(0)
                seg = AudioSegment.from_wav(wav_buf)
                seg.export(buf, format="mp3", bitrate="128k")
            except Exception:
                logger.warning("MP3 encoding failed (pydub/ffmpeg missing), returning WAV")
                sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
        else:
            sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
