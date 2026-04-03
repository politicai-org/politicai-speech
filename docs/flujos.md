# Flujos de Datos en orbe-speech

`orbe-speech` maneja flujos de datos para síntesis de voz (TTS) y transcripción de voz (STT).

## 1. Síntesis de Voz (TTS)

### Flujo HTTP POST (`/tts`)
1. **Entrada**: Texto (UTF-8) y parámetros de idioma/proveedor.
2. **Validación**: Verifica `X-API-Key` y longitud de texto.
3. **Selección de Proveedor**:
   - `auto` + idioma "quz"/"qu" -> **MMS (Local)**.
   - `auto` + idioma "es" -> **ElevenLabs** (si está configurado) o **Polly**.
   - Forzado (e.g., `provider="elevenlabs"`) -> Usa el proveedor especificado.
4. **Procesamiento**:
   - **MMS (Local)**: Inferencia CPU con modelo VITS -> Waveform -> Codificación audio (WAV/MP3).
   - **Cloud**: Request a API externa -> Audio raw.
5. **Salida**: Bytes de audio con `Content-Type` correspondiente.

### Flujo gRPC Stream (`SynthesizeStream`)
1. Inicia sesión gRPC con metadatos de autenticación.
2. Envía `SynthesizeRequest`.
3. El servidor genera el audio (completo o parcial) y emite `AudioChunk` secuencialmente a través del stream.

## 2. Transcripción de Voz (STT)

### Flujo HTTP POST (`/stt`)
1. **Entrada**: Archivo de audio (WAV, WEBM, MP3, OGG) y hint de idioma opcional.
2. **Procesamiento**:
   - Whisper recibe los bytes de audio.
   - Inferencia CPU -> Texto transcrito + Idioma detectado.
3. **Salida**: JSON con texto e idioma detectado.

### Flujo gRPC Stream (`TranscribeStream`)
1. El cliente envía ráfagas de audio PCM (`AudioChunk`).
2. El servidor acumula audio y realiza transcripciones parciales cada `settings.stt_partial_interval_seconds`.
3. Emite eventos de transcripción en tiempo real (`TranscriptEvent`) hasta recibir la ráfaga final (`last=True`).

## Soporte Quechua (MMS)
El soporte para Quechua es crítico y se maneja mediante modelos entrenados específicamente (`facebook/mms-tts-quz`). Este flujo es local y no depende de APIs externas, garantizando la pronunciación nativa correcta.
