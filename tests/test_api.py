"""
orbe-speech API tests.

Uses mocks for MmsTTSService and WhisperSTTService to avoid loading
actual PyTorch models during CI. Tests cover routing, auth, and
response format — not model accuracy.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app, _tts, _stt

API_KEY = "speech-dev-key-2024"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_services(monkeypatch):
    """
    Inject mock TTS and STT services so tests never load real models.
    Patches module-level singletons in app.main.
    """
    import app.main as main_module

    mock_tts = MagicMock()
    mock_tts.language = "quz"
    mock_tts.synthesize = AsyncMock(return_value=b"RIFF_fake_wav_bytes")

    mock_stt = MagicMock()
    mock_stt.transcribe = AsyncMock(
        return_value={"text": "Allillanchu", "language_detected": "qu"}
    )

    monkeypatch.setattr(main_module, "_tts", mock_tts)
    monkeypatch.setattr(main_module, "_stt", mock_stt)
    return mock_tts, mock_stt


@pytest.fixture
def client():
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["tts_model_loaded"] is True
    assert data["stt_model_loaded"] is True


# ── TTS /synthesize ───────────────────────────────────────────────────────────

def test_tts_synthesize_returns_audio(client):
    response = client.post(
        "/tts",
        headers={"X-API-Key": API_KEY},
        json={"text": "Allillanchu", "language": "quz"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 0


def test_tts_synthesize_rejects_missing_api_key(client):
    response = client.post("/tts", json={"text": "Allillanchu", "language": "quz"})
    assert response.status_code == 403


def test_tts_synthesize_rejects_wrong_api_key(client):
    response = client.post(
        "/tts",
        headers={"X-API-Key": "wrong-key"},
        json={"text": "Allillanchu", "language": "quz"},
    )
    assert response.status_code == 403


def test_tts_synthesize_rejects_wrong_language(client):
    response = client.post(
        "/tts",
        headers={"X-API-Key": API_KEY},
        json={"text": "Hello", "language": "en"},
    )
    assert response.status_code == 400
    assert "quz" in response.json()["detail"]


def test_tts_synthesize_rejects_empty_text(client):
    response = client.post(
        "/tts",
        headers={"X-API-Key": API_KEY},
        json={"text": "", "language": "quz"},
    )
    assert response.status_code == 422


def test_tts_synthesize_rejects_text_too_long(client):
    response = client.post(
        "/tts",
        headers={"X-API-Key": API_KEY},
        json={"text": "a" * 1001, "language": "quz"},
    )
    assert response.status_code == 400


# ── TTS /stream ───────────────────────────────────────────────────────────────

def test_tts_stream_returns_audio_chunks(client):
    response = client.post(
        "/tts/stream",
        headers={"X-API-Key": API_KEY},
        json={"text": "Allillanchu", "language": "quz"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")


# ── STT /stt ──────────────────────────────────────────────────────────────────

def test_stt_transcribes_audio(client):
    fake_wav = BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    response = client.post(
        "/stt",
        headers={"X-API-Key": API_KEY},
        files={"audio": ("test.wav", fake_wav, "audio/wav")},
        data={"language": "qu"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["text"] == "Allillanchu"


def test_stt_rejects_missing_api_key(client):
    fake_wav = BytesIO(b"RIFF_data")
    response = client.post(
        "/stt",
        files={"audio": ("test.wav", fake_wav, "audio/wav")},
    )
    assert response.status_code == 403


def test_stt_rejects_empty_audio(client):
    empty = BytesIO(b"")
    response = client.post(
        "/stt",
        headers={"X-API-Key": API_KEY},
        files={"audio": ("empty.wav", empty, "audio/wav")},
    )
    assert response.status_code == 400
