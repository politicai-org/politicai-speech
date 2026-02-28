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
        client_kwargs = {
            "service_name": "polly",
            "region_name": settings.aws_region,
        }
        if settings.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            
        self.client = boto3.client(**client_kwargs)

    def synthesize(self, text: str, voice_id: str | None = None) -> bytes:
        vid = voice_id or "Lupe" # Default Spanish voice
        
        text_type = "text"
        if text.strip().startswith("<speak>"):
            text_type = "ssml"

        response = self.client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId=vid,
            Engine="neural",
            TextType=text_type
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

class MinimaxProvider(TTSProvider):
    """MiniMax TTS provider with voice cloning support."""
    
    def synthesize(self, text: str, voice_id: str | None = None, api_key: str | None = None) -> bytes:
        key = api_key or settings.minimax_api_key
        group_id = settings.minimax_group_id
        
        if not key or not group_id:
            raise ValueError("MiniMax API key and group ID must be configured")
        
        if not voice_id:
            raise ValueError("voice_id is required for MiniMax TTS")
        
        url = f"{settings.minimax_base_url}"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "group_id": group_id,
            "voice_id": voice_id,
            "text": text,
            "model": "speech-01-turbo",
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
            "audio_sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            raise RuntimeError(f"MiniMax TTS error ({response.status_code}): {response.text}")
        
        result = response.json()
        
        if result.get("base_resp", {}).get("status_code") != 0:
            error_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax API error: {error_msg}")
        
        audio_data = result.get("data", {}).get("audio")
        if not audio_data:
            raise RuntimeError("MiniMax response missing audio data")
        
        import base64
        return base64.b64decode(audio_data)
    
    def clone_voice(self, audio_bytes: bytes, voice_name: str, api_key: str | None = None) -> str:
        """
        Clone a voice using MiniMax API.
        Returns the cloned voice_id.
        """
        key = api_key or settings.minimax_api_key
        group_id = settings.minimax_group_id
        
        if not key or not group_id:
            raise ValueError("MiniMax API key and group ID must be configured")
        
        clone_url = "https://api.minimax.chat/v1/voice_clone"
        headers = {
            "Authorization": f"Bearer {key}"
        }
        
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            "group_id": group_id,
            "voice_name": voice_name,
            "audio": audio_base64,
            "audio_format": "mp3"
        }
        
        response = requests.post(clone_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            raise RuntimeError(f"MiniMax voice clone error ({response.status_code}): {response.text}")
        
        result = response.json()
        
        if result.get("base_resp", {}).get("status_code") != 0:
            error_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax clone API error: {error_msg}")
        
        voice_id = result.get("data", {}).get("voice_id")
        if not voice_id:
            raise RuntimeError("MiniMax clone response missing voice_id")
        
        return voice_id
