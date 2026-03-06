from abc import ABC, abstractmethod
import io
from typing import AsyncIterator, Iterator
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

    async def synthesize_stream(self, text: str, voice_id: str | None = None, api_key: str | None = None) -> AsyncIterator[bytes]:
        """Default async implementation: synthesize full audio and yield it in one chunk.
        Override this for true streaming."""
        yield self.synthesize(text, voice_id, api_key)

class ElevenLabsProvider(TTSProvider):
    def synthesize(self, text: str, voice_id: str | None = None, api_key: str | None = None) -> bytes:
        vid = voice_id or settings.elevenlabs_voice_id
        if not vid:
            raise ValueError("voice_id is required for ElevenLabs TTS")
        
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

    async def synthesize_stream(self, text: str, voice_id: str | None = None, api_key: str | None = None) -> AsyncIterator[bytes]:
        """Async generator for streaming audio."""
        vid = voice_id or settings.elevenlabs_voice_id
        if not vid:
            raise ValueError("voice_id is required for ElevenLabs TTS")
        
        key = api_key or settings.elevenlabs_api_key
        if not key:
            raise ValueError("ElevenLabs API key not configured")
            
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream"
        
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
        
        # Try to use httpx for async streaming if available
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, json=data, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.read()
                        raise RuntimeError(f"ElevenLabs stream error: {error_text.decode()}")
                    
                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        if chunk:
                            yield chunk
            return
        except ImportError:
            pass
        except Exception:
            fallback_audio = self.synthesize(text, voice_id=vid, api_key=key)
            for i in range(0, len(fallback_audio), 1024):
                yield fallback_audio[i : i + 1024]
            return
            
        # Fallback to requests (blocking, but works)
        response = requests.post(url, json=data, headers=headers, stream=True)
        
        if response.status_code != 200:
            fallback_audio = self.synthesize(text, voice_id=vid, api_key=key)
            for i in range(0, len(fallback_audio), 1024):
                yield fallback_audio[i : i + 1024]
            return
            
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk


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
        Clone a voice using MiniMax API via File Upload + Voice Clone workflow.
        Returns the cloned voice_id (which is generated here and sent to API).
        """
        key = api_key or settings.minimax_api_key
        group_id = settings.minimax_group_id
        
        if not key or not group_id:
            raise ValueError("MiniMax API key and group ID must be configured")
            
        # 1. Upload file first
        upload_url = "https://api.minimax.io/v1/files/upload"
        headers_upload = {
            "Authorization": f"Bearer {key}"
        }
        
        # Use a generic filename but ensure extension matches content if possible
        # Assuming MP3 or WAV. API usually detects header.
        files = {
            'file': (f'{voice_name}.mp3', audio_bytes, 'audio/mpeg'),
            'purpose': (None, 'voice_clone')
        }
        
        upload_resp = requests.post(upload_url, headers=headers_upload, files=files, timeout=60)
        if upload_resp.status_code != 200:
             raise RuntimeError(f"MiniMax upload error ({upload_resp.status_code}): {upload_resp.text}")
        
        upload_data = upload_resp.json()
        if upload_data.get("base_resp", {}).get("status_code") != 0:
             error_msg = upload_data.get("base_resp", {}).get("status_msg", "Unknown error")
             raise RuntimeError(f"MiniMax upload API error: {error_msg}")
             
        file_id = upload_data.get("file", {}).get("file_id")
        if not file_id:
             raise RuntimeError("MiniMax upload response missing file_id")

        # 2. Clone voice using file_id
        clone_url = "https://api.minimax.io/v1/voice_clone"
        headers_clone = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        # Generate valid voice_id: alphanumeric, start with letter, >=8 chars
        import uuid
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9]', '', voice_name)
        if not safe_name:
            safe_name = "Voice"
        # Ensure start with letter
        if not safe_name[0].isalpha():
            safe_name = "V" + safe_name
            
        # Append UUID to ensure uniqueness and length
        generated_voice_id = f"{safe_name[:10]}{str(uuid.uuid4().hex)[:10]}"
        
        payload = {
            "group_id": group_id,
            "voice_id": generated_voice_id,
            "file_id": file_id
        }
        
        response = requests.post(clone_url, json=payload, headers=headers_clone, timeout=60)
        
        if response.status_code != 200:
            raise RuntimeError(f"MiniMax voice clone error ({response.status_code}): {response.text}")
        
        result = response.json()
        
        if result.get("base_resp", {}).get("status_code") != 0:
            error_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax clone API error: {error_msg}")
        
        # If success, the voice_id we sent is now valid
        return generated_voice_id
