use anyhow::Context;
use code_graph_db::Database;
use code_graph_shared::{CodeChunk, SearchHit};
use qdrant_client::{
    client::{QdrantClient as GrpcQdrantClient, QdrantClientConfig},
    prelude::Value,
    qdrant::{
        vectors_config::Config, Condition, CreateCollection, Distance, Filter, PointId,
        PointStruct, SearchPoints, VectorParams, VectorsConfig,
    },
};
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

#[derive(Clone)]
pub struct QdrantClient {
    client: Arc<GrpcQdrantClient>,
    collection: String,
    vector_size: usize,
}

impl QdrantClient {
    pub fn new(grpc_url: String, collection: String, vector_size: usize) -> anyhow::Result<Self> {
        let config = QdrantClientConfig::from_url(grpc_url.trim_end_matches('/'));
        Ok(Self {
            client: Arc::new(GrpcQdrantClient::new(Some(config))?),
            collection,
            vector_size,
        })
    }

    pub async fn ensure_collection(&self) -> anyhow::Result<()> {
        if self
            .client
            .collection_exists(&self.collection)
            .await
            .context("failed to check qdrant collection")?
        {
            return Ok(());
        }

        self.client
            .create_collection(&CreateCollection {
                collection_name: self.collection.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: self.vector_size as u64,
                        distance: Distance::Cosine as i32,
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await
            .context("qdrant collection creation failed")?;
        Ok(())
    }

    pub async fn upsert_chunk(
        &self,
        chunk_id: Uuid,
        repo_id: Uuid,
        chunk: &CodeChunk,
        vector: Vec<f32>,
    ) -> anyhow::Result<Uuid> {
        let vector_id = chunk.vector_id.unwrap_or_else(Uuid::new_v4);

        let payload = HashMap::<String, Value>::from([
            ("chunk_id".to_string(), chunk_id.to_string().into()),
            ("repo_id".to_string(), repo_id.to_string().into()),
            ("file_path".to_string(), chunk.file_path.clone().into()),
            ("language".to_string(), chunk.language.clone().into()),
            ("start_line".to_string(), i64::from(chunk.start_line).into()),
            ("end_line".to_string(), i64::from(chunk.end_line).into()),
        ]);
        let point = PointStruct {
            id: Some(PointId::from(vector_id.to_string())),
            vectors: Some(vector.into()),
            payload,
        };

        self.client
            .upsert_points_blocking(&self.collection, None, vec![point], None)
            .await
            .context("qdrant point upsert failed")?;
        Ok(vector_id)
    }

    pub async fn upsert_chunks_batch(&self, chunks: Vec<QdrantChunkUpsert>) -> anyhow::Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let points = chunks
            .into_iter()
            .map(|item| {
                let payload = HashMap::<String, Value>::from([
                    ("chunk_id".to_string(), item.chunk_id.to_string().into()),
                    ("repo_id".to_string(), item.repo_id.to_string().into()),
                    ("file_path".to_string(), item.file_path.into()),
                    ("language".to_string(), item.language.into()),
                    ("start_line".to_string(), i64::from(item.start_line).into()),
                    ("end_line".to_string(), i64::from(item.end_line).into()),
                ]);
                PointStruct {
                    id: Some(PointId::from(item.vector_id.to_string())),
                    vectors: Some(item.vector.into()),
                    payload,
                }
            })
            .collect::<Vec<_>>();

        self.client
            .upsert_points_blocking(&self.collection, None, points, None)
            .await
            .context("qdrant batch point upsert failed")?;
        Ok(())
    }

    pub async fn search(
        &self,
        repo_id: Uuid,
        vector: Vec<f32>,
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorHit>> {
        let response = self
            .client
            .search_points(&SearchPoints {
                collection_name: self.collection.clone(),
                vector,
                filter: Some(Filter::must([Condition::matches(
                    "repo_id",
                    repo_id.to_string(),
                )])),
                limit: top_k as u64,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await
            .context("qdrant search failed")?;

        Ok(response
            .result
            .into_iter()
            .filter_map(|item| {
                let chunk_id = item
                    .payload
                    .get("chunk_id")?
                    .kind
                    .as_ref()
                    .and_then(qdrant_value_as_str)?;
                Some(VectorHit {
                    chunk_id: Uuid::parse_str(chunk_id).ok()?,
                    score: item.score,
                })
            })
            .collect())
    }
}

pub struct QdrantChunkUpsert {
    pub chunk_id: Uuid,
    pub repo_id: Uuid,
    pub vector_id: Uuid,
    pub file_path: String,
    pub language: String,
    pub start_line: i32,
    pub end_line: i32,
    pub vector: Vec<f32>,
}

fn qdrant_value_as_str(kind: &qdrant_client::qdrant::value::Kind) -> Option<&str> {
    match kind {
        qdrant_client::qdrant::value::Kind::StringValue(value) => Some(value),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub chunk_id: Uuid,
    pub score: f32,
}

#[derive(Clone)]
pub struct Retriever {
    db: Database,
    qdrant: QdrantClient,
}

impl Retriever {
    pub fn new(db: Database, qdrant: QdrantClient) -> Self {
        Self { db, qdrant }
    }

    pub async fn retrieve(
        &self,
        repo_id: Uuid,
        query_vector: Vec<f32>,
        question: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let mut merged: HashMap<Uuid, SearchHit> = HashMap::new();

        if let Ok(vector_hits) = self.qdrant.search(repo_id, query_vector, top_k).await {
            for hit in vector_hits {
                if let Ok(mut chunk) = self.db.get_chunk(hit.chunk_id).await {
                    chunk.score = hit.score;
                    chunk.method = "vector".to_string();
                    merged.insert(chunk.chunk_id, chunk);
                }
            }
        }

        for hit in self
            .db
            .keyword_search(repo_id, question, top_k as i64)
            .await?
        {
            merged
                .entry(hit.chunk_id)
                .and_modify(|existing| {
                    existing.score += hit.score;
                    existing.method = format!("{}+keyword", existing.method);
                })
                .or_insert(hit);
        }

        let mut hits: Vec<_> = merged.into_values().collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(top_k);
        Ok(hits)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct UpsertPayload {
    pub chunk_id: Uuid,
    pub repo_id: Uuid,
}
