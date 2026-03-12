# =============================================================================
# Koios Makefile
# Automates platform detection, build, and container lifecycle management.
# All Docker files live in build/; generated config files are also in build/.
# =============================================================================

COMPOSE = docker compose --project-directory . -f build/docker-compose.yml -f build/docker-compose.override.yml
ENV_FILE = build/.env.build

.PHONY: help setup build up-d up down restart logs logs-ollama logs-api logs-streamlit ps rebuild clean clean-images reset

# Default target
.DEFAULT_GOAL := help

help: ## Show available make targets
	@echo ""
	@echo "  Koios - Available Commands"
	@echo "============================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup: ## Detect host OS and generate platform-specific configuration
	@chmod +x build/setup.sh
	@build/setup.sh

build: setup ## Detect platform, configure, and build all Docker images
	@set -a && . $(ENV_FILE) && set +a && $(COMPOSE) build

up-d: ## Start all services in detached mode
	$(COMPOSE) up -d

up:  ## Start all services in non-detached mode
	$(COMPOSE) up

down: ## Stop and remove all containers
	$(COMPOSE) down

restart: down up ## Restart all services

logs: ## Tail logs for all services (Ctrl+C to exit)
	$(COMPOSE) logs -f

logs-ollama: ## Tail Ollama service logs
	$(COMPOSE) logs -f ollama

logs-api: ## Tail API service logs
	$(COMPOSE) logs -f api

logs-streamlit: ## Tail Streamlit service logs
	$(COMPOSE) logs -f streamlit

ps: ## Show status of all containers
	$(COMPOSE) ps

rebuild: setup ## Force rebuild all images without cache
	@set -a && . $(ENV_FILE) && set +a && $(COMPOSE) build --no-cache

clean: ## Stop containers, remove volumes, and delete generated config files
	-$(COMPOSE) down -v 2>/dev/null || true
	rm -f build/docker-compose.override.yml build/.env.build

clean-images: ## Remove Koios Docker images
	-docker rmi koios-api koios-streamlit 2>/dev/null || true

reset: clean clean-images ## Full reset: remove containers, volumes, images, and generated files
	@echo "✓ Full reset complete."
