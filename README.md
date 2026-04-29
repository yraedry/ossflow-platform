# ossflow-platform

Backend orquestado del ecosistema OSSFlow para procesamiento de instruccionales BJJ.

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `ossflow-api` | 8000 | Gateway FastAPI: orquesta backends, gestiona pipelines y biblioteca |
| `ossflow-splitter` | 8001 | Fragmentación en capítulos (modo signal o timestamps) |
| `ossflow-subtitle` | 8002 | Generación de subtítulos con WhisperX |
| `ossflow-dubbing` | 8003 | Doblaje vía ElevenLabs |
| `ossflow-telegram` | 8004 | Descarga de media desde canales Telegram |

## Arranque

```bash
docker compose build
docker compose up -d
```

Requiere `ossflow-base:latest` construido previamente desde `ossflow-core`.
