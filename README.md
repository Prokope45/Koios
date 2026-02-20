# Koios
---

Based on this [Medium article](https://medium.com/@sahin.samia/how-to-build-a-interactive-personal-ai-research-agent-with-llama-3-2-b2a390eed63e)[1], this project seeks to create an AI research agent that takes the research question as input and either generates an answer based on its knowledge or queries the DuckDuckGo web API for more context. If it cannot find any additional information, it will return a message indicating that it does not have enough information to answer the question.

The major difference between this and the article demonstration is that this project is structured in a object-oriented manner, and will include other methods of information querying using Wikipedia and the Google search API.

## Features:
- **RAG (Retrieval-Augmented Generation)**: Uses local LLMs to answer questions.
- **Optional Internet Search**: Can be toggled on/off. Uses DuckDuckGo and Wikipedia as fallback.
- **Document Store**: Upload PDF documents to provide local context for the model.
- **Streamlit UI**: Interactive chat interface with settings and document management.

## Configuration:
The following environment variables can be set in `src/.env`:
- `OPENAI_URL`: URL for the OpenAI-compatible API (e.g., LM Studio).
- `KOIOS_ENABLE_INTERNET_SEARCH`: Set to `True` to enable internet search by default.

## Getting Started:

### Option 1: Docker (Recommended)
The easiest way to run Koios is using Docker, which includes Ollama with llama3.2, FastAPI, and Streamlit:

```bash
# Configure environment
cp src/.env.example src/.env

# Build and start all services
docker compose up -d

# Access the services
# - Streamlit UI: http://localhost:8501
# - FastAPI: http://localhost:8000/docs
# - Ollama: http://localhost:11434
```

See [DOCKER-README.md](DOCKER-README.md) for detailed Docker setup instructions.

### Option 2: Local Development
It is assumed that you are running a local model using the developer server on LM Studio.

1. Create virtual environment `python3 -m venv venv`
2. Activate virtual environment `source venv/bin/activate`
3. Install dependencies `pip install -r src/requirements.txt`
4. Configure environment `cp src/.env.example src/.env`
5. Run the program either by:
    - Running the webapp `python3 -m src app`
    - Directly query the model `python3 -m src "<research question to ask>"`

## Source
[1] https://medium.com/@sahin.samia/how-to-build-a-interactive-personal-ai-research-agent-with-llama-3-2-b2a390eed63e