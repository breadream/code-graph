use anyhow::{anyhow, Context};
use code_graph_db::Database;
use code_graph_llm::EmbeddingProvider;
use code_graph_parser::{chunk_source, hash_content, ChunkOptions};
use code_graph_retriever::{QdrantChunkUpsert, QdrantClient};
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

#[derive(Debug, Clone)]
pub enum IndexProgress {
    Preparing,
    DiscoveredFiles {
        total_files: usize,
    },
    FileStarted {
        current_file: usize,
        total_files: usize,
        path: String,
    },
    FileSkipped {
        current_file: usize,
        total_files: usize,
        path: String,
        chunks: usize,
    },
    EmbeddingBatch {
        current_file: usize,
        total_files: usize,
        batch_start: usize,
        batch_end: usize,
        file_chunks: usize,
    },
    UpsertingBatch {
        current_file: usize,
        total_files: usize,
        batch_start: usize,
        batch_end: usize,
        file_chunks: usize,
    },
    ChunkIndexed {
        current_file: usize,
        total_files: usize,
        file_chunks: usize,
        total_chunks: usize,
    },
    Finished {
        files_indexed: usize,
        chunks_indexed: usize,
    },
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
        self.index_repo_with_progress(request, |_| {}).await
    }

    pub async fn index_repo_with_progress<F>(
        &self,
        request: RepoRequest,
        mut progress: F,
    ) -> anyhow::Result<IndexSummary>
    where
        F: FnMut(IndexProgress),
    {
        progress(IndexProgress::Preparing);
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

        let files = discover_source_files(checkout.path());
        let total_files = files.len();
        progress(IndexProgress::DiscoveredFiles { total_files });

        let mut files_indexed = 0usize;
        let mut chunks_indexed = 0usize;
        let mut seen_paths = Vec::with_capacity(total_files);

        for (file_idx, path) in files.iter().enumerate() {
            let current_file = file_idx + 1;
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
            seen_paths.push(rel_path.clone());
            progress(IndexProgress::FileStarted {
                current_file,
                total_files,
                path: rel_path.clone(),
            });
            let chunks = chunk_source(&rel_path, &content, &ChunkOptions::default());
            if chunks.is_empty() {
                continue;
            }

            let file_hash = hash_content(&content);
            if let Some(existing) = self.db.get_indexed_file(repo_id, &rel_path).await? {
                if existing.content_hash == file_hash && existing.chunk_count > 0 {
                    files_indexed += 1;
                    chunks_indexed += existing.chunk_count as usize;
                    progress(IndexProgress::FileSkipped {
                        current_file,
                        total_files,
                        path: rel_path,
                        chunks: existing.chunk_count as usize,
                    });
                    continue;
                }
                self.db.delete_file(repo_id, &rel_path).await?;
            }

            let file_id = self
                .db
                .upsert_file(repo_id, &rel_path, &chunks[0].language, &file_hash)
                .await?;
            files_indexed += 1;

            let file_chunks = chunks.len();
            for (batch_start, batch) in chunks.chunks(32).enumerate() {
                let start = batch_start * 32 + 1;
                let end = start + batch.len() - 1;
                progress(IndexProgress::EmbeddingBatch {
                    current_file,
                    total_files,
                    batch_start: start,
                    batch_end: end,
                    file_chunks,
                });
                let inputs = batch
                    .iter()
                    .map(|chunk| chunk.content.clone())
                    .collect::<Vec<_>>();
                let vectors = self.embeddings.embed_batch(&inputs).await?;
                progress(IndexProgress::UpsertingBatch {
                    current_file,
                    total_files,
                    batch_start: start,
                    batch_end: end,
                    file_chunks,
                });
                let mut upserts = Vec::with_capacity(batch.len());
                for (chunk, vector) in batch.iter().cloned().zip(vectors) {
                    let mut chunk = chunk;
                    chunk.repo_id = Some(repo_id);
                    chunk.vector_id = Some(Uuid::new_v4());
                    let persisted = self.db.insert_chunk(repo_id, file_id, &chunk).await?;
                    chunk.vector_id = Some(persisted.vector_id);
                    upserts.push(QdrantChunkUpsert {
                        chunk_id: persisted.chunk_id,
                        repo_id,
                        vector_id: persisted.vector_id,
                        file_path: chunk.file_path,
                        language: chunk.language,
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        vector,
                    });
                }
                self.qdrant.upsert_chunks_batch(upserts).await?;
                chunks_indexed += batch.len();
                progress(IndexProgress::ChunkIndexed {
                    current_file,
                    total_files,
                    file_chunks: end,
                    total_chunks: chunks_indexed,
                });
            }
        }

        self.db.delete_files_except(repo_id, &seen_paths).await?;
        progress(IndexProgress::Finished {
            files_indexed,
            chunks_indexed,
        });
        Ok(IndexSummary {
            repo_id,
            files_indexed,
            chunks_indexed,
        })
    }
}

fn discover_source_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| !is_ignored(entry))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| is_likely_source(path))
        .collect()
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
