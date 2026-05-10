use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub database_url: String,
    pub qdrant_url: String,
    pub qdrant_grpc_url: String,
    pub qdrant_collection: String,
    pub embedding_dim: usize,
    pub provider: ProviderConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub mode: ProviderMode,
    pub openai_base_url: Option<String>,
    pub openai_api_key: Option<String>,
    pub embedding_model: String,
    pub chat_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderMode {
    Mock,
    OpenAiCompatible,
}

impl AppConfig {
    pub fn from_env() -> Self {
        let provider_mode = std::env::var("PROVIDER_MODE").unwrap_or_else(|_| "mock".to_string());
        let mode = match provider_mode.as_str() {
            "openai" | "openai_compatible" => ProviderMode::OpenAiCompatible,
            _ => ProviderMode::Mock,
        };

        Self {
            database_url: std::env::var("DATABASE_URL").unwrap_or_else(|_| {
                "postgres://codegraph:codegraph@localhost:5432/codegraph".to_string()
            }),
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_grpc_url: std::env::var("QDRANT_GRPC_URL")
                .unwrap_or_else(|_| "http://localhost:6334".to_string()),
            qdrant_collection: std::env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "code_chunks".to_string()),
            embedding_dim: std::env::var("EMBEDDING_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128),
            provider: ProviderConfig {
                mode,
                openai_base_url: std::env::var("OPENAI_BASE_URL").ok(),
                openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
                embedding_model: std::env::var("EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string()),
                chat_model: std::env::var("CHAT_MODEL")
                    .unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoRequest {
    pub repo_name: String,
    pub branch: Option<String>,
    pub repo_url: Option<String>,
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoResponse {
    pub repo_id: Uuid,
    pub files_indexed: usize,
    pub chunks_indexed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub repo_id: Uuid,
    pub question: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub answer: String,
    pub citations: Vec<Citation>,
    pub retrieval_trace: Vec<RetrievalTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub file_path: String,
    pub start_line: i32,
    pub end_line: i32,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalTrace {
    pub chunk_id: Uuid,
    pub score: f32,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub id: Option<Uuid>,
    pub repo_id: Option<Uuid>,
    pub file_path: String,
    pub language: String,
    pub symbol_name: Option<String>,
    pub symbol_type: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub content: String,
    pub content_hash: String,
    pub vector_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk_id: Uuid,
    pub repo_id: Uuid,
    pub file_path: String,
    pub language: String,
    pub start_line: i32,
    pub end_line: i32,
    pub content: String,
    pub score: f32,
    pub method: String,
}
