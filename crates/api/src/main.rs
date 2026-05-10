use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use code_graph_db::Database;
use code_graph_indexer::Indexer;
use code_graph_llm::{build_providers, ChatProvider, EmbeddingProvider};
use code_graph_retriever::{QdrantClient, Retriever};
use code_graph_shared::{
    AppConfig, Citation, QueryRequest, QueryResponse, RepoRequest, RepoResponse, RetrievalTrace,
};
use std::{net::SocketAddr, sync::Arc};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    db: Database,
    indexer: Indexer,
    retriever: Retriever,
    embeddings: Arc<dyn EmbeddingProvider>,
    chat: Arc<dyn ChatProvider>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "code_graph_api=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = AppConfig::from_env();
    let db = Database::connect(&config.database_url).await?;
    let providers = build_providers(&config.provider, config.embedding_dim)?;
    let qdrant = QdrantClient::new(
        config.qdrant_url.clone(),
        config.qdrant_collection.clone(),
        providers.embeddings.dimensions(),
    );
    let indexer = Indexer::new(db.clone(), qdrant.clone(), providers.embeddings.clone());
    let retriever = Retriever::new(db.clone(), qdrant);

    let state = AppState {
        db,
        indexer,
        retriever,
        embeddings: providers.embeddings,
        chat: providers.chat,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/repos", post(index_repo))
        .route("/query", post(query))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port = std::env::var("API_PORT").unwrap_or_else(|_| "8080".to_string());
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    tracing::info!(%addr, "starting API");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn index_repo(
    State(state): State<AppState>,
    Json(payload): Json<RepoRequest>,
) -> Result<Json<RepoResponse>, ApiError> {
    let summary = state.indexer.index_repo(payload).await?;
    Ok(Json(RepoResponse {
        repo_id: summary.repo_id,
        files_indexed: summary.files_indexed,
        chunks_indexed: summary.chunks_indexed,
    }))
}

async fn query(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let top_k = payload.top_k.unwrap_or(8).clamp(1, 20);
    let query_vector = state.embeddings.embed(&payload.question).await?;
    let hits = state
        .retriever
        .retrieve(payload.repo_id, query_vector, &payload.question, top_k)
        .await?;
    let answer = state.chat.answer(&payload.question, &hits).await?;
    state
        .db
        .log_query(payload.repo_id, &payload.question, &answer)
        .await?;

    let citations = hits
        .iter()
        .map(|hit| Citation {
            file_path: hit.file_path.clone(),
            start_line: hit.start_line,
            end_line: hit.end_line,
            snippet: hit.content.clone(),
        })
        .collect();
    let retrieval_trace = hits
        .iter()
        .map(|hit| RetrievalTrace {
            chunk_id: hit.chunk_id,
            score: hit.score,
            method: hit.method.clone(),
        })
        .collect();

    Ok(Json(QueryResponse {
        answer,
        citations,
        retrieval_trace,
    }))
}

struct ApiError(anyhow::Error);

impl<E> From<E> for ApiError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        tracing::error!(error = ?self.0, "request failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": self.0.to_string()
            })),
        )
            .into_response()
    }
}
