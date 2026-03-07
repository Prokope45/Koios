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
- `make` (pre-installed on macOS and most Linux distros; available via Git Bash on Windows)
- At least 8GB of available RAM (for llama3.2 model)
- At least 10GB of free disk space

## Quick Start

### 1. Configure Environment Variables

Copy the example environment file and configure it:

`bash
cp ./.env.example ./.env
`

Edit `./.env` to set your preferences:

`bash
# For Docker deployment, use the ollama service name
OPENAI_URL=http://ollama:11434

# Enable/disable internet search
ENABLE_INTERNET_SEARCH=False

# Optional: Add your Hugging Face token
# HF_TOKEN=your_huggingface_token_here
`

### 2. Build the Containers

```bash
make build
```

This single command will:
1. **Detect your host OS** (macOS, Windows, or Linux)
2. **Configure platform-specific settings** automatically:
   - **macOS**: Enables Docker Desktop model runner on port 12434, skips GPU packages
   - **Windows/Linux with NVIDIA GPU**: Enables GPU passthrough, installs CUDA packages
   - **Windows/Linux without GPU**: Falls back to CPU-only configuration
3. **Build all Docker images** with the appropriate settings

### 3. Start the Services

```bash
make up
```

The first startup will take several minutes as Ollama downloads the llama3.2 model (~2GB).

### 4. Monitor the Logs

`bash
# View all logs
make logs

# View specific service logs
make logs-ollama
make logs-api
make logs-streamlit
```

### 5. Access the Services

Once all services are healthy:

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Ollama API**: http://localhost:11434

## Available Make Commands

```
make help          Show all available commands
make setup         Detect host OS and generate platform-specific configuration
make build         Detect platform, configure, and build all Docker images
make up            Start all services in detached mode
make down          Stop and remove all containers
make restart       Restart all services
make logs          Tail logs for all services
make logs-ollama   Tail Ollama service logs
make logs-api      Tail API service logs
make logs-streamlit Tail Streamlit service logs
make ps            Show status of all containers
make rebuild       Force rebuild all images without cache
make clean         Stop containers, remove volumes, and delete generated config files
make clean-images  Remove Koios Docker images
make reset         Full reset: remove containers, volumes, images, and generated files
```

## Platform-Specific Behavior

### macOS
- GPU passthrough is **not** supported on macOS Docker
- Docker Desktop model runner is automatically enabled on TCP port 12434 for efficient LLM inference
- NVIDIA/CUDA Python packages are **not** installed
- `docker-compose.override.yml` is generated with `OLLAMA_NUM_GPU=0`

### Windows / Linux with NVIDIA GPU
- GPU passthrough is enabled via the NVIDIA container runtime
- CUDA and NVIDIA Python packages are installed from `src/requirements-gpu.txt`
- `docker-compose.override.yml` is generated with full GPU reservation settings
- `OLLAMA_NUM_GPU=99` is set for maximum GPU utilization

### Windows / Linux without NVIDIA GPU
- Falls back to CPU-only configuration
- No GPU packages installed

## How It Works

The build system uses three auto-generated files (excluded from git):

| File | Purpose |
|------|---------|
| `docker-compose.override.yml` | Injects GPU deploy settings and `OLLAMA_NUM_GPU` for the Ollama container |
| `.env.build` | Sets `INSTALL_GPU_PACKAGES=true/false` for Dockerfile build args |

Docker Compose automatically merges `docker-compose.override.yml` with `docker-compose.yml` at runtime, so no manual file editing is needed.

## Using the FastAPI Endpoint

### Health Check

`bash
curl http://localhost:8000/health
`

### Get Available Models

`bash
curl http://localhost:8000/models
`

### Query via GET

`bash
curl "http://localhost:8000/query?query=What%20is%20machine%20learning?&name=John"
`

### Query via POST

`bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "name": "John",
    "model": "llama3.2",
    "temperature": 0.7,
    "enable_internet_search": false
  }'
`

### Example Response

`json
{
  "query": "What is machine learning?",
  "name": "John",
  "generation": "Machine learning is a subset of artificial intelligence...",
  "model": "llama3.2"
}
`

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
make down
```

### Restart All Services

```bash
make restart
```

### View Container Status

```bash
make ps
```

### Rebuild After Code Changes

```bash
make build
make up
```

### Force Rebuild Without Cache

```bash
make rebuild
make up
```

### Full Reset (removes everything)

```bash
make reset
```

## Troubleshooting

### Ollama Model Not Loading

If the Ollama container fails to pull the model:

`bash
# Check Ollama logs
make logs-ollama

# Manually pull the model
docker compose exec ollama ollama pull llama3.2:3b
```

### API Connection Issues

If the API can't connect to Ollama:

1. Verify Ollama is healthy: `make ps`
2. Check the network: `docker network inspect koios_koios_net`
3. Verify environment variable: `OPENAI_URL=http://ollama:11434`

### macOS: Docker Model Runner Issues

If the Docker Desktop model runner fails to enable automatically:

```bash
docker desktop enable model-runner --tcp 12434
```

Then re-run `make build`.

### Port Conflicts

If ports 8000, 8501, or 11434 are already in use, modify `docker-compose.yml`:

`yaml
ports:
  - "8080:8000"  # Change host port (left side)
`

### Out of Memory

If containers crash due to memory:

1. Increase Docker memory limit (Docker Desktop settings)
2. Use a smaller model (e.g., `llama3.2:1b`)
3. Close other applications

### Rebuilding from Scratch

```bash
make reset
make build
make up
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

`
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
`

## Volume Persistence

- `ollama_data`: Stores downloaded models (persists between restarts)
- `./db`: ChromaDB vector store (mounted from host)
- `./temp_uploads`: Temporary PDF uploads (mounted from host)

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
