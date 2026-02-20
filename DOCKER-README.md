# Koios Docker Setup

This guide explains how to run the Koios RAG system using Docker containers.

## Architecture

The Docker setup consists of three containers running on a shared network (`koios_net`):

1. **Ollama Container** - Runs the Ollama service with llama3.2 model
2. **FastAPI Container** - Exposes the RAG model via REST API
3. **Streamlit Container** - Provides the web UI for interacting with the system

All containers can communicate with each other using service names (e.g., `http://ollama:11434`).

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose V2
- At least 8GB of available RAM (for llama3.2 model)
- At least 10GB of free disk space

## Quick Start

### 1. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp src/.env.example src/.env
```

Edit `src/.env` to set your preferences:

```bash
# For Docker deployment, use the ollama service name
OPENAI_URL=http://ollama:11434

# Enable/disable internet search
KOIOS_ENABLE_INTERNET_SEARCH=False

# Optional: Add your Hugging Face token
# HF_TOKEN=your_huggingface_token_here
```

### 2. Build the Containers

```bash
docker compose build
```

This will build the FastAPI and Streamlit containers. The Ollama container uses the official pre-built image.

### 3. Start the Services

```bash
docker compose up -d
```

The first startup will take several minutes as Ollama downloads the llama3.2 model (~2GB).

### 4. Monitor the Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f ollama
docker compose logs -f api
docker compose logs -f streamlit
```

### 5. Access the Services

Once all services are healthy:

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Ollama API**: http://localhost:11434

## Using the FastAPI Endpoint

### Health Check

```bash
curl http://localhost:8000/health
```

### Get Available Models

```bash
curl http://localhost:8000/models
```

### Query via GET

```bash
curl "http://localhost:8000/query?query=What%20is%20machine%20learning?&name=John"
```

### Query via POST

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "name": "John",
    "model": "llama3.2",
    "temperature": 0.7,
    "enable_internet_search": false
  }'
```

### Example Response

```json
{
  "query": "What is machine learning?",
  "name": "John",
  "generation": "Machine learning is a subset of artificial intelligence...",
  "model": "llama3.2"
}
```

## Using the Streamlit UI

1. Open http://localhost:8501 in your browser
2. Configure settings in the sidebar:
   - Select the LLM model
   - Adjust creativity (temperature)
   - Enable/disable internet search
   - Upload PDF documents for RAG
3. Enter your research question in the chat input
4. View the generated response

## Managing the Containers

### Stop the Services

```bash
docker compose down
```

### Stop and Remove Volumes (including model data)

```bash
docker compose down -v
```

### Restart a Specific Service

```bash
docker compose restart api
docker compose restart streamlit
docker compose restart ollama
```

### View Container Status

```bash
docker compose ps
```

### Rebuild After Code Changes

```bash
docker compose build api streamlit
docker compose up -d
```

## Troubleshooting

### Ollama Model Not Loading

If the Ollama container fails to pull the model:

```bash
# Check Ollama logs
docker compose logs ollama

# Manually pull the model
docker compose exec ollama ollama pull llama3.2
```

### API Connection Issues

If the API can't connect to Ollama:

1. Verify Ollama is healthy: `docker compose ps`
2. Check the network: `docker network inspect koios_koios_net`
3. Verify environment variable: `OPENAI_URL=http://ollama:11434`

### Port Conflicts

If ports 8000, 8501, or 11434 are already in use, modify `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"  # Change host port (left side)
```

### Out of Memory

If containers crash due to memory:

1. Increase Docker memory limit (Docker Desktop settings)
2. Use a smaller model (e.g., `llama3.2:1b`)
3. Close other applications

### Rebuilding from Scratch

```bash
# Stop and remove everything
docker compose down -v

# Remove images
docker rmi koios-api koios-streamlit

# Rebuild
docker compose build --no-cache
docker compose up -d
```

## Development Mode

For development with hot-reload:

1. The API and Streamlit containers mount source code as volumes
2. Changes to Python files will automatically reload the services
3. No rebuild needed for code changes

## Production Considerations

For production deployment:

1. Remove `--reload` flag from uvicorn command in `Dockerfile.api`
2. Set proper resource limits in `docker-compose.yml`
3. Use environment-specific `.env` files
4. Enable HTTPS with a reverse proxy (nginx, traefik)
5. Set up proper logging and monitoring
6. Use Docker secrets for sensitive data
7. Consider using a managed Ollama service for better performance

## Network Architecture

```
┌─────────────────────────────────────────┐
│         koios_net (bridge)              │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Ollama   │  │ FastAPI  │  │Streamlit││
│  │ :11434   │◄─┤ :8000    │◄─┤ :8501   ││
│  └──────────┘  └──────────┘  └────────┘│
│       ▲             ▲             ▲     │
└───────┼─────────────┼─────────────┼─────┘
        │             │             │
    localhost     localhost     localhost
      :11434        :8000         :8501
```

## Volume Persistence

- `ollama_data`: Stores downloaded models (persists between restarts)
- `./db`: ChromaDB vector store (mounted from host)
- `./temp_uploads`: Temporary PDF uploads (mounted from host)

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
