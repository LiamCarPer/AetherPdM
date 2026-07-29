# API Base URLs

| Environment | URL | Notes |
|-------------|-----|-------|
| **Local dev** | `http://localhost:8000` | `uv run uvicorn aether_pdm.serve.app:app --reload` |
| **Docker** | `http://api:8000` | Service name in `docker-compose.yml` — only resolvable inside the Docker network |
| **Docker host** | `http://localhost:8000` | Mapped via `ports: - 8000:8000` in docker-compose |

## Swagger Docs

- Local: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json
