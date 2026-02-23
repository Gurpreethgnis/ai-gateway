# AI Gateway Makefile
# Docker management and development helpers

.PHONY: up down logs build restart shell db-shell ollama-pull ollama-list test lint

# =============================================================================
# Docker Commands
# =============================================================================

up:
	docker-compose up -d
	@echo "Gateway running at http://localhost:8000"

down:
	docker-compose down

logs:
	docker-compose logs -f gateway

logs-all:
	docker-compose logs -f

build:
	docker-compose build --no-cache gateway

restart:
	docker-compose restart gateway

# =============================================================================
# Shell Access
# =============================================================================

shell:
	docker-compose exec gateway /bin/bash

db-shell:
	docker-compose exec postgres psql -U gateway -d gateway

redis-cli:
	docker-compose exec redis redis-cli

# =============================================================================
# Ollama Model Management
# =============================================================================

# Usage: make ollama-pull MODEL=llama3.2:latest
ollama-pull:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make ollama-pull MODEL=<model-name>"; exit 1; fi
	docker-compose exec ollama ollama pull $(MODEL)

ollama-list:
	docker-compose exec ollama ollama list

ollama-rm:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make ollama-rm MODEL=<model-name>"; exit 1; fi
	docker-compose exec ollama ollama rm $(MODEL)

# Common models
pull-llama:
	docker-compose exec ollama ollama pull llama3.2:latest

pull-qwen-coder:
	docker-compose exec ollama ollama pull qwen2.5-coder:14b

pull-deepseek:
	docker-compose exec ollama ollama pull deepseek-coder-v2:16b

# =============================================================================
# Database
# =============================================================================

db-migrate:
	docker-compose exec gateway python -c "from gateway.db import create_tables; import asyncio; asyncio.run(create_tables())"

db-reset:
	docker-compose down -v postgres
	docker-compose up -d postgres
	@echo "Waiting for postgres..."
	@sleep 5
	$(MAKE) db-migrate

# =============================================================================
# Development
# =============================================================================

dev:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest test_gateway.py -v

lint:
	ruff check gateway/

format:
	ruff format gateway/

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

env-example:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example - please fill in your API keys"; else echo ".env already exists"; fi

setup: env-example install
	@echo "Setup complete. Run 'make up' to start services."

# =============================================================================
# Production
# =============================================================================

deploy:
	git add -A && git commit -m "Deploy: $$(date +%Y-%m-%d_%H:%M)" && git push origin main

health:
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Gateway not responding"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	docker-compose down -v
	rm -rf __pycache__ gateway/__pycache__ gateway/**/__pycache__
	rm -rf .pytest_cache

prune:
	docker system prune -f
	docker volume prune -f
