# AI Codebase Knowledge Engine

Rust MVP for indexing source repositories into searchable code knowledge:

- Postgres stores repository metadata, file inventory, chunk provenance, and query logs.
- Qdrant stores vector embeddings for code chunks and supports semantic retrieval.
- The Rust API/worker layer is expected to ingest repositories, chunk files, embed chunks, write vectors to Qdrant, and keep relational metadata in Postgres.

## Local Setup

Prerequisites:

- Docker Desktop or compatible Docker Engine
- `docker compose`
- `psql` optional, only needed if you prefer running migrations from the host
- Rust toolchain for the application once the crate is added

Create a local env file:

```sh
cp .env.example .env
```

Start infrastructure:

```sh
make up
```

Run database migrations:

```sh
make migrate
```

Check services:

```sh
make health
```

Stop services:

```sh
make down
```

Reset all local state:

```sh
make reset
```

## Services

| Service | URL | Purpose |
| --- | --- | --- |
| Postgres | `postgres://codegraph:codegraph@localhost:5432/codegraph` | Relational source of truth |
| Qdrant HTTP | `http://localhost:6333` | Vector database API |
| Qdrant gRPC | `localhost:6334` | Optional high-throughput vector API |

## Architecture

```text
Repository URL/path
      |
      v
Rust ingestion worker
      |
      +--> file discovery and language detection
      +--> chunking and symbol-aware metadata
      +--> embedding model
               |
               +--> Qdrant collection: code_chunks
               |
               +--> Postgres tables:
                    repositories
                    files
                    chunks
                    query_logs
```

Postgres owns durable metadata and audit history. Qdrant owns nearest-neighbor search over embedding vectors. Chunks bridge the two stores through `chunks.qdrant_point_id`.

## Database Schema

The initial migration creates:

- `repositories`: one row per indexed repository.
- `files`: current indexed files for a repository, including path, language, hash, and size.
- `chunks`: searchable code chunks with byte/line ranges and Qdrant point IDs.
- `query_logs`: request/response metadata for retrieval observability and tuning.

Apply migrations with:

```sh
make migrate
```

The migration is idempotent and safe to rerun locally.

## Qdrant Demo

Create a small demo collection:

```sh
curl -sS -X PUT http://localhost:6333/collections/code_chunks_demo \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 3,
      "distance": "Cosine"
    }
  }'
```

Insert a demo vector:

```sh
curl -sS -X PUT http://localhost:6333/collections/code_chunks_demo/points \
  -H 'Content-Type: application/json' \
  -d '{
    "points": [
      {
        "id": "00000000-0000-0000-0000-000000000001",
        "vector": [0.01, 0.02, 0.03],
        "payload": {
          "repository_id": "demo",
          "path": "src/main.rs",
          "language": "rust",
          "symbol": "main"
        }
      }
    ]
  }'
```

Search the collection:

```sh
curl -sS -X POST http://localhost:6333/collections/code_chunks_demo/points/search \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.01, 0.02, 0.03],
    "limit": 5,
    "with_payload": true
  }'
```

## Run The API

In another terminal:

```sh
cargo run --bin api
```

The MVP exposes `GET /health`, `POST /repos`, and `POST /query`.

## Terminal-First Usage

You can skip the HTTP API entirely for day-to-day use. Start the backing stores once:

```sh
make up
make migrate
cargo install --path crates/api --bin insight
```

Then go into any cloned repository and ask a question:

```sh
cd /path/to/cloned/repo
insight "Where is authentication handled?"
```

Optional flags:

```sh
insight --path /path/to/repo --repo-name my-project --branch main --top-k 8 "How does billing work?"
```

## API Demo Curls

Ingest a repository:

```sh
curl -sS -X POST http://localhost:8080/repos \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_name": "rustlings",
    "repo_url": "https://github.com/rust-lang/rustlings",
    "branch": "main"
  }'
```

For a local checkout, replace `repo_url` with `local_path`.

Ask a codebase question:

```sh
curl -sS -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_id": "REPLACE_WITH_REPO_ID_FROM_INDEX_RESPONSE",
    "question": "Where is exercise progress calculated?",
    "top_k": 8
  }'
```

## Make Targets

```sh
make up          # Start Postgres and Qdrant
make down        # Stop containers
make logs        # Follow infrastructure logs
make ps          # Show container status
make health      # Check Postgres and Qdrant availability
make migrate     # Apply SQL migrations
make reset       # Destroy local volumes and restart clean infrastructure
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Keep secrets out of git.

Important variables:

- `DATABASE_URL`: application Postgres connection string.
- `QDRANT_URL`: Qdrant HTTP endpoint.
- `QDRANT_COLLECTION`: collection name for chunk embeddings.
- `PROVIDER_MODE`: `mock` for local deterministic embeddings/answers, or `openai_compatible`.
- `EMBEDDING_DIM`: embedding vector dimension. Must match the configured embedding model and Qdrant collection.
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `CHAT_MODEL`: OpenAI-compatible provider settings.

## What Is Implemented

- Repository ingestion from local paths or Git URLs.
- Source walking with common vendor/build directories ignored.
- Extension-based language detection.
- Function/class-like symbol chunking for Rust, TypeScript/JavaScript, and Python, with line-based fallback chunking.
- Postgres metadata persistence for repositories, files, chunks, and query logs.
- Qdrant collection creation, vector upsert, and vector search.
- Keyword fallback via Postgres `ILIKE`.
- Mock and OpenAI-compatible provider abstractions for embeddings and chat answers.

## Current MVP Limits

- Tree-sitter is not wired in yet; chunking is regex-assisted plus robust line fallback.
- Repo indexing is synchronous in the API request.
- Existing chunk cleanup for deleted files is not implemented.
- Qdrant failures during query fall back to keyword search, but ingestion requires Qdrant to be available.
- `EMBEDDING_MODEL`: embedding model identifier used by the application.
