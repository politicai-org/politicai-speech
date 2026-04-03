# Arquitectura de orbe-speech

`orbe-speech` es un microservicio especializado en procesamiento de voz (TTS y STT) diseñado para soportar múltiples idiomas, con énfasis en el soporte nativo de Quechua.

## Componentes Principales

- **FastAPI**: Proporciona el servidor HTTP para integraciones RESTful.
- **gRPC**: Servidor de alto rendimiento para streaming bidireccional de audio (STT parcial y TTS en tiempo real).
- **Transformers (HuggingFace)**:
  - **MMS (Massively Multilingual Speech)**: Utiliza modelos `facebook/mms-tts-quz` para síntesis de voz nativa en Quechua (VITS).
  - **Whisper**: Utiliza `openai/whisper-small` para transcripción multilingüe robusta.
- **Proveedores Cloud (Integraciones)**:
  - **ElevenLabs**: Para clonación de voz premium y TTS de alta calidad en español/inglés.
  - **AWS Polly**: Proveedor de respaldo para múltiples idiomas.
  - **MiniMax**: Especializado en clonación de voz y síntesis avanzada.

## Estructura de Capas

1. **API (Main)**: Define los endpoints HTTP/gRPC y gestiona el ciclo de vida de los modelos (lifespan).
2. **Servicios (Services)**:
   - `MmsTTSService`: Lógica de inferencia CPU para modelos VITS locales.
   - `WhisperSTTService`: Lógica de transcripción usando Whisper.
   - `Providers`: Adaptadores para servicios externos (ElevenLabs, Polly, MiniMax).
3. **Modelos (Models)**: Definiciones de Pydantic para validación de datos.

## Puntos de Integración

- **Entrada**: Recibe texto o audio vía HTTP POST o gRPC streams.
- **Salida**: Retorna audio (WAV/MP3/PCM) o texto transcrito.
- **Seguridad**: Requiere `X-API-Key` en los headers o metadatos de gRPC.
