use anyhow::Context;
use code_graph_db::Database;
use code_graph_shared::{CodeChunk, SearchHit};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Clone)]
pub struct QdrantClient {
    client: reqwest::Client,
    base_url: String,
    collection: String,
    vector_size: usize,
}

impl QdrantClient {
    pub fn new(base_url: String, collection: String, vector_size: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            collection,
            vector_size,
        }
    }

    pub async fn ensure_collection(&self) -> anyhow::Result<()> {
        let response = self
            .client
            .put(format!("{}/collections/{}", self.base_url, self.collection))
            .json(&json!({
                "vectors": {
                    "size": self.vector_size,
                    "distance": "Cosine"
                }
            }))
            .send()
            .await
            .context("failed to create qdrant collection")?;
        if response.status() != StatusCode::CONFLICT {
            response
                .error_for_status()
                .context("qdrant collection creation failed")?;
        }
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
        self.client
            .put(format!(
                "{}/collections/{}/points",
                self.base_url, self.collection
            ))
            .json(&json!({
                "points": [{
                    "id": vector_id,
                    "vector": vector,
                    "payload": {
                        "chunk_id": chunk_id,
                        "repo_id": repo_id,
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line
                    }
                }]
            }))
            .send()
            .await
            .context("failed to upsert qdrant point")?
            .error_for_status()
            .context("qdrant point upsert failed")?;
        Ok(vector_id)
    }

    pub async fn search(
        &self,
        repo_id: Uuid,
        vector: Vec<f32>,
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorHit>> {
        let resp: QdrantSearchResponse = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, self.collection
            ))
            .json(&json!({
                "vector": vector,
                "limit": top_k,
                "with_payload": true,
                "filter": {
                    "must": [{
                        "key": "repo_id",
                        "match": { "value": repo_id }
                    }]
                }
            }))
            .send()
            .await
            .context("failed to search qdrant")?
            .error_for_status()
            .context("qdrant search failed")?
            .json()
            .await
            .context("failed to parse qdrant search response")?;

        Ok(resp
            .result
            .into_iter()
            .filter_map(|item| {
                let chunk_id = item.payload.and_then(|payload| payload.chunk_id)?;
                Some(VectorHit {
                    chunk_id,
                    score: item.score,
                })
            })
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub chunk_id: Uuid,
    pub score: f32,
}

#[derive(Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantScoredPoint>,
}

#[derive(Deserialize)]
struct QdrantScoredPoint {
    score: f32,
    payload: Option<QdrantPayload>,
}

#[derive(Deserialize)]
struct QdrantPayload {
    chunk_id: Option<Uuid>,
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
