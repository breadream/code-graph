use anyhow::{anyhow, Context};
use async_trait::async_trait;
use code_graph_shared::{ProviderConfig, ProviderMode, SearchHit};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, input: &str) -> anyhow::Result<Vec<f32>>;
    async fn embed_batch(&self, inputs: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(inputs.len());
        for input in inputs {
            embeddings.push(self.embed(input).await?);
        }
        Ok(embeddings)
    }
    fn dimensions(&self) -> usize;
}

#[async_trait]
pub trait ChatProvider: Send + Sync {
    async fn answer(&self, question: &str, context: &[SearchHit]) -> anyhow::Result<String>;
}

pub struct Providers {
    pub embeddings: Arc<dyn EmbeddingProvider>,
    pub chat: Arc<dyn ChatProvider>,
}

pub fn build_providers(config: &ProviderConfig, embedding_dim: usize) -> anyhow::Result<Providers> {
    match config.mode {
        ProviderMode::Mock => {
            let mock = Arc::new(MockProvider { dim: embedding_dim });
            Ok(Providers {
                embeddings: mock.clone(),
                chat: mock,
            })
        }
        ProviderMode::OpenAiCompatible => {
            let provider = Arc::new(OpenAiCompatibleProvider::new(
                config.clone(),
                embedding_dim,
            )?);
            Ok(Providers {
                embeddings: provider.clone(),
                chat: provider,
            })
        }
    }
}

pub struct MockProvider {
    dim: usize,
}

#[async_trait]
impl EmbeddingProvider for MockProvider {
    async fn embed(&self, input: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.embed_one(input))
    }

    async fn embed_batch(&self, inputs: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(inputs.iter().map(|input| self.embed_one(input)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dim
    }
}

impl MockProvider {
    fn embed_one(&self, input: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dim];
        for token in input.split(|c: char| !c.is_alphanumeric()) {
            if token.is_empty() {
                continue;
            }
            let mut hasher = Sha256::new();
            hasher.update(token.to_lowercase().as_bytes());
            let hash = hasher.finalize();
            let idx = u64::from_le_bytes(hash[0..8].try_into().unwrap()) as usize % self.dim;
            vector[idx] += 1.0;
        }
        normalize(&mut vector);
        vector
    }
}

#[async_trait]
impl ChatProvider for MockProvider {
    async fn answer(&self, question: &str, context: &[SearchHit]) -> anyhow::Result<String> {
        if context.is_empty() {
            return Ok(
                "Insufficient evidence in the retrieved context.\n\nRelevant files:\n- none"
                    .to_string(),
            );
        }

        let files = context
            .iter()
            .take(5)
            .map(|hit| format!("- {}:{}-{}", hit.file_path, hit.start_line, hit.end_line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(
            "Based on the retrieved code, the best starting points for \"{question}\" are the cited snippets below. This mock answer does not infer beyond retrieved context.\n\nRelevant files:\n{files}"
        ))
    }
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector {
            *v /= norm;
        }
    }
}

pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    embedding_model: String,
    chat_model: String,
    dim: usize,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: ProviderConfig, dim: usize) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: config
                .openai_base_url
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
                .trim_end_matches('/')
                .to_string(),
            api_key: config
                .openai_api_key
                .ok_or_else(|| anyhow!("OPENAI_API_KEY is required for openai-compatible mode"))?,
            embedding_model: config.embedding_model,
            chat_model: config.chat_model,
            dim,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatibleProvider {
    async fn embed(&self, input: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[input.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embedding response was empty"))
    }

    async fn embed_batch(&self, inputs: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let resp: EmbeddingResponse = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&EmbeddingRequest {
                model: &self.embedding_model,
                input: inputs,
            })
            .send()
            .await
            .context("embedding request failed")?
            .error_for_status()
            .context("embedding provider returned an error")?
            .json()
            .await
            .context("failed to parse embedding response")?;

        let mut data = resp.data;
        data.sort_by_key(|item| item.index);
        let embeddings = data
            .into_iter()
            .map(|item| item.embedding)
            .collect::<Vec<_>>();
        if embeddings.len() != inputs.len() {
            return Err(anyhow!(
                "embedding response count mismatch: expected {}, got {}",
                inputs.len(),
                embeddings.len()
            ));
        }
        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dim
    }
}

#[async_trait]
impl ChatProvider for OpenAiCompatibleProvider {
    async fn answer(&self, question: &str, context: &[SearchHit]) -> anyhow::Result<String> {
        let context_text = context
            .iter()
            .map(|hit| {
                format!(
                    "FILE: {}:{}-{}\n{}\n",
                    hit.file_path, hit.start_line, hit.end_line, hit.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n");

        let system = "You answer questions about a codebase using only retrieved context. Cite files and line numbers. If evidence is insufficient, say so. Include a 'Relevant files' section.";
        let user = format!("Question: {question}\n\nRetrieved context:\n{context_text}");

        let resp: ChatResponse = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&ChatRequest {
                model: &self.chat_model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: system.to_string(),
                    },
                    ChatMessage {
                        role: "user",
                        content: user,
                    },
                ],
                temperature: 0.1,
            })
            .send()
            .await
            .context("chat request failed")?
            .error_for_status()
            .context("chat provider returned an error")?
            .json()
            .await
            .context("failed to parse chat response")?;

        resp.choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("chat response was empty"))
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use code_graph_shared::SearchHit;
    use uuid::Uuid;

    #[tokio::test]
    async fn mock_provider_embeds_and_answers_with_citations_context() {
        let provider = MockProvider { dim: 16 };
        let embedding = provider.embed("auth middleware auth").await.unwrap();
        assert_eq!(embedding.len(), 16);
        assert!(embedding.iter().any(|v| *v > 0.0));

        let hit = SearchHit {
            chunk_id: Uuid::new_v4(),
            repo_id: Uuid::new_v4(),
            file_path: "src/auth.rs".to_string(),
            language: "rust".to_string(),
            start_line: 10,
            end_line: 20,
            content: "fn authenticate() {}".to_string(),
            score: 0.9,
            method: "vector".to_string(),
        };
        let answer = provider
            .answer("Where is authentication handled?", &[hit])
            .await
            .unwrap();
        assert!(answer.contains("Relevant files"));
        assert!(answer.contains("src/auth.rs:10-20"));
    }
}
