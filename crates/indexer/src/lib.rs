use anyhow::{anyhow, Context};
use code_graph_db::Database;
use code_graph_llm::EmbeddingProvider;
use code_graph_parser::{chunk_source, hash_content, ChunkOptions};
use code_graph_retriever::QdrantClient;
use code_graph_shared::RepoRequest;
use git2::{build::RepoBuilder, Repository};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;
use uuid::Uuid;
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Clone)]
pub struct IndexSummary {
    pub repo_id: Uuid,
    pub files_indexed: usize,
    pub chunks_indexed: usize,
}

#[derive(Clone)]
pub struct Indexer {
    db: Database,
    qdrant: QdrantClient,
    embeddings: Arc<dyn EmbeddingProvider>,
}

impl Indexer {
    pub fn new(db: Database, qdrant: QdrantClient, embeddings: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            db,
            qdrant,
            embeddings,
        }
    }

    pub async fn index_repo(&self, request: RepoRequest) -> anyhow::Result<IndexSummary> {
        let branch = request.branch.clone().unwrap_or_else(|| "main".to_string());
        let checkout = Checkout::prepare(&request, &branch)?;
        let commit_sha = git_commit_sha(checkout.path()).ok();
        let repo_id = self
            .db
            .upsert_repository(
                &request.repo_name,
                request
                    .repo_url
                    .as_deref()
                    .or(request.local_path.as_deref()),
                &branch,
                commit_sha.as_deref(),
            )
            .await?;

        self.qdrant.ensure_collection().await?;

        let mut files_indexed = 0usize;
        let mut chunks_indexed = 0usize;

        for entry in WalkDir::new(checkout.path())
            .into_iter()
            .filter_entry(|entry| !is_ignored(entry))
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
        {
            let path = entry.path();
            if !is_likely_source(path) {
                continue;
            }
            let content = match tokio::fs::read_to_string(path).await {
                Ok(content) => content,
                Err(_) => continue,
            };
            if content.trim().is_empty() {
                continue;
            }

            let rel_path = path
                .strip_prefix(checkout.path())
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let chunks = chunk_source(&rel_path, &content, &ChunkOptions::default());
            if chunks.is_empty() {
                continue;
            }

            let file_hash = hash_content(&content);
            let file_id = self
                .db
                .upsert_file(repo_id, &rel_path, &chunks[0].language, &file_hash)
                .await?;
            files_indexed += 1;

            for mut chunk in chunks {
                chunk.repo_id = Some(repo_id);
                chunk.vector_id = Some(Uuid::new_v4());
                let vector = self.embeddings.embed(&chunk.content).await?;
                let chunk_id = self.db.insert_chunk(repo_id, file_id, &chunk).await?;
                self.qdrant
                    .upsert_chunk(chunk_id, repo_id, &chunk, vector)
                    .await?;
                chunks_indexed += 1;
            }
        }

        Ok(IndexSummary {
            repo_id,
            files_indexed,
            chunks_indexed,
        })
    }
}

struct Checkout {
    path: PathBuf,
    _temp_dir: Option<TempDir>,
}

impl Checkout {
    fn prepare(request: &RepoRequest, branch: &str) -> anyhow::Result<Self> {
        if let Some(local_path) = &request.local_path {
            let path = PathBuf::from(local_path);
            if !path.exists() {
                return Err(anyhow!("local_path does not exist: {local_path}"));
            }
            return Ok(Self {
                path,
                _temp_dir: None,
            });
        }

        let repo_url = request
            .repo_url
            .as_ref()
            .ok_or_else(|| anyhow!("repo_url or local_path is required"))?;
        let temp_dir = TempDir::new().context("failed to create clone temp dir")?;
        let mut builder = RepoBuilder::new();
        builder.branch(branch);
        builder
            .clone(repo_url, temp_dir.path())
            .with_context(|| format!("failed to clone {repo_url}"))?;
        Ok(Self {
            path: temp_dir.path().to_path_buf(),
            _temp_dir: Some(temp_dir),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

fn git_commit_sha(path: &Path) -> anyhow::Result<String> {
    let repo = Repository::discover(path)?;
    let head = repo.head()?.peel_to_commit()?;
    Ok(head.id().to_string())
}

fn is_ignored(entry: &DirEntry) -> bool {
    let ignored = [
        ".git",
        "target",
        "node_modules",
        "dist",
        "build",
        ".next",
        "coverage",
        "vendor",
    ];
    entry
        .file_name()
        .to_str()
        .map(|name| ignored.contains(&name))
        .unwrap_or(false)
}

fn is_likely_source(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default(),
        "rs" | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "cs"
            | "rb"
            | "md"
            | "toml"
            | "yaml"
            | "yml"
            | "json"
    )
}
