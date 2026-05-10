SHELL := /bin/sh

COMPOSE ?= docker compose
POSTGRES_SERVICE ?= postgres
POSTGRES_USER ?= codegraph
POSTGRES_DB ?= codegraph
QDRANT_URL ?= http://localhost:6333

.PHONY: up down logs ps health migrate reset

up:
	$(COMPOSE) up -d postgres qdrant

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f postgres qdrant

ps:
	$(COMPOSE) ps

health:
	$(COMPOSE) exec $(POSTGRES_SERVICE) pg_isready -U $(POSTGRES_USER) -d $(POSTGRES_DB)
	curl -fsS $(QDRANT_URL)/healthz

migrate:
	$(COMPOSE) exec -T $(POSTGRES_SERVICE) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -v ON_ERROR_STOP=1 -f /migrations/001_init.sql

reset:
	$(COMPOSE) down -v
	$(COMPOSE) up -d postgres qdrant

