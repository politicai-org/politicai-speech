from abc import ABC, abstractmethod
import io
import requests
import boto3
from contextlib import closing
from ..config import get_settings

settings = get_settings()

class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    def synthesize(self, text: str, voice_id: str | None = None, api_key: str | None = None) -> bytes:
        pass

class ElevenLabsProvider(TTSProvider):
    def synthesize(self, text: str, voice_id: str | None = None, api_key: str | None = None) -> bytes:
        vid = voice_id or settings.elevenlabs_voice_id
        
        # Use provided key or global fallback
        key = api_key or settings.elevenlabs_api_key
        if not key:
            raise ValueError("ElevenLabs API key not configured (neither global nor per-request)")
            
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
        headers = {
            "xi-api-key": key,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs error: {response.text}")
            
        return response.content

class PollyProvider(TTSProvider):
    def __init__(self):
        self.client = boto3.client(
            "polly",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )

    def synthesize(self, text: str, voice_id: str | None = None) -> bytes:
        vid = voice_id or "Lupe" # Default Spanish voice
        
        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId=vid,
            Engine="neural"
        )
        
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                return stream.read()
        else:
            raise RuntimeError("Polly did not return audio stream")

class MmsProvider(TTSProvider):
    """Local MMS provider wrapper (uses existing MmsTTSService logic)"""
    def __init__(self, mms_service):
        self.service = mms_service
        
    def synthesize(self, text: str, voice_id: str | None = None) -> bytes:
        # MMS service returns wav bytes directly
        # Note: MMS service implementation details might need adjustment to match interface
        return self.service.synthesize(text)
