# Pruebas en orbe-speech

`orbe-speech` cuenta con un conjunto de pruebas unitarias e integración para validar el funcionamiento de TTS y STT multilingüe.

## 1. Pruebas Unitarias/Integración

Las pruebas se encuentran principalmente en la carpeta `tests/`.

### Archivos de Prueba Principales
- `tests/test_tts.py`: Valida síntesis de voz, selección de proveedores y codificación de audio (MMS, Polly, ElevenLabs).
- `tests/test_stt.py`: Valida transcripción asincrónica con Whisper, detección de idiomas y manejo de archivos de audio.
- `tests/test_grpc.py`: Valida la comunicación gRPC, autenticación por API-Key y streams bidireccionales.
- `test_orbe_chat.py`: Script de prueba rápido para validar la integración general de chat basada en audio.

## 2. Ejecución de Pruebas

Para ejecutar el conjunto completo de pruebas:
```bash
pytest tests/
```

### Pruebas de Sistema/E2E
Se recomienda ejecutar `tests/test_tts.py` para verificar que los modelos (especialmente MMS para Quechua) están cargados correctamente y generan audio reproducible.

## 3. Validación de Modelos
Durante el arranque, el servicio registra la carga exitosa (`MMS TTS model loaded`). Las pruebas de integración verifican que el retorno del endpoint `/health` indique `tts_model_loaded: true` y `stt_model_loaded: true`.
