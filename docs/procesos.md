# Procesos en orbe-speech

`orbe-speech` es un microservicio Python basado en FastAPI que requiere dependencias de sistema particulares para el procesamiento de audio.

## 1. Configuración del Entorno Local

### Requisitos Previos
- Python 3.10 o superior.
- **FFmpeg**: Necesario para transcodificación de formatos de audio (MP3, WAV).
- **Pydub**: Librería de manipulación de audio.
- **Transformers (HuggingFace)**: Entorno local con caché configurado para modelos pesados (VITS, Whisper).

### Instalación de Dependencias
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar requerimientos
pip install -r requirements.txt
```

### Configuración
Configura las variables de entorno necesarias en `.env`:
- `API_KEY`: Clave para autenticación `X-API-Key`.
- `MODEL_PATH`: Directorio donde se descargarán y cargarán los modelos (MMS, Whisper).
- `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: Para Polly.
- `ELEVENLABS_API_KEY`: Para ElevenLabs.
- `MINIMAX_API_KEY`, `MINIMAX_GROUP_ID`: Para MiniMax.

## 2. Ejecución

### Modo Desarrollo
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Usando Docker
Modo producción recomendado:
```bash
docker compose up --build -d
```
El contenedor expone tanto el puerto HTTP (8000) como el puerto gRPC (50051).

## 3. Despliegue y Recursos
Los modelos (especialmente Whisper y MMS) requieren una cantidad considerable de memoria RAM (~2-4GB) y cargan durante el arranque del servicio (lifespan event). En entornos restringidos, el arranque inicial puede tomar ~30s mientras se descargan/cachean los modelos en `MODEL_PATH`.
