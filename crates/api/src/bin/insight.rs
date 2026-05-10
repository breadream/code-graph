use anyhow::{anyhow, Context};
use code_graph_db::Database;
use code_graph_indexer::{IndexProgress, Indexer};
use code_graph_llm::{build_providers, ChatProvider, EmbeddingProvider};
use code_graph_retriever::{QdrantClient, Retriever};
use code_graph_shared::{AppConfig, RepoRequest};
use git2::Repository;
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let args = Args::parse()?;
    let repo_path = args.path.canonicalize().with_context(|| {
        format!(
            "failed to resolve repo path {}",
            args.path.to_string_lossy()
        )
    })?;
    let repo_name = args
        .repo_name
        .clone()
        .unwrap_or_else(|| infer_repo_name(&repo_path));
    let branch = args
        .branch
        .clone()
        .unwrap_or_else(|| infer_branch(&repo_path));

    let config = AppConfig::from_env();
    let db = Database::connect(&config.database_url).await?;
    let providers = build_providers(&config.provider, config.embedding_dim)?;
    let qdrant = QdrantClient::new(
        config.qdrant_grpc_url,
        config.qdrant_collection,
        providers.embeddings.dimensions(),
    )?;
    let indexer = Indexer::new(db.clone(), qdrant.clone(), providers.embeddings.clone());
    let retriever = Retriever::new(db.clone(), qdrant);

    match args.command {
        Command::Index | Command::Refresh => {
            index_repo(&indexer, &repo_path, repo_name, branch).await?;
        }
        Command::Ask { question } => {
            let repo_id = require_indexed_repo(&db, &repo_name, &branch).await?;
            ask_repo(
                &db,
                &retriever,
                providers.embeddings,
                providers.chat,
                repo_id,
                &question,
                args.top_k,
            )
            .await?;
        }
        Command::Status => {
            print_status(&db, &repo_name, &branch).await?;
        }
        Command::IndexAndAsk { question } => {
            let summary = index_repo(&indexer, &repo_path, repo_name, branch).await?;
            ask_repo(
                &db,
                &retriever,
                providers.embeddings,
                providers.chat,
                summary.repo_id,
                &question,
                args.top_k,
            )
            .await?;
        }
    }

    Ok(())
}

async fn index_repo(
    indexer: &Indexer,
    repo_path: &Path,
    repo_name: String,
    branch: String,
) -> anyhow::Result<code_graph_indexer::IndexSummary> {
    println!("Indexing {} ({})...", repo_path.display(), branch);
    let mut last_line_len = 0usize;
    let summary = indexer
        .index_repo_with_progress(
            RepoRequest {
                repo_name,
                branch: Some(branch),
                repo_url: None,
                local_path: Some(repo_path.to_string_lossy().to_string()),
            },
            |event| render_index_progress(event, &mut last_line_len),
        )
        .await?;
    finish_progress_line(&mut last_line_len);
    println!(
        "Indexed {} files and {} chunks. repo_id={}",
        summary.files_indexed, summary.chunks_indexed, summary.repo_id
    );
    Ok(summary)
}

async fn ask_repo(
    db: &Database,
    retriever: &Retriever,
    embeddings: Arc<dyn EmbeddingProvider>,
    chat: Arc<dyn ChatProvider>,
    repo_id: Uuid,
    question: &str,
    top_k: usize,
) -> anyhow::Result<()> {
    eprintln!("Query: embedding question...");
    let query_vector = embeddings.embed(question).await?;
    eprintln!("Query: retrieving top {top_k} chunks...");
    let hits = retriever
        .retrieve(repo_id, query_vector, question, top_k)
        .await?;
    eprintln!(
        "Query: generating cited answer from {} chunks...",
        hits.len()
    );
    let answer = chat.answer(question, &hits).await?;
    eprintln!("Query: logging result...");
    db.log_query(repo_id, question, &answer).await?;

    println!("\nAnswer\n{answer}\n");
    println!("Citations");
    for hit in &hits {
        println!(
            "- {}:{}-{} [{:.3}, {}]",
            hit.file_path, hit.start_line, hit.end_line, hit.score, hit.method
        );
    }
    Ok(())
}

fn render_index_progress(event: IndexProgress, last_line_len: &mut usize) {
    let message = match event {
        IndexProgress::Preparing => "Preparing repository...".to_string(),
        IndexProgress::DiscoveredFiles { total_files } => {
            format!("Discovered {total_files} source-like files.")
        }
        IndexProgress::FileStarted {
            current_file,
            total_files,
            path,
        } => {
            format!(
                "[{current_file}/{total_files}] reading {}",
                truncate_middle(&path, 70)
            )
        }
        IndexProgress::FileSkipped {
            current_file,
            total_files,
            path,
            chunks,
        } => {
            format!(
                "[{current_file}/{total_files}] unchanged, reused {chunks} chunks from {}",
                truncate_middle(&path, 55)
            )
        }
        IndexProgress::EmbeddingBatch {
            current_file,
            total_files,
            batch_start,
            batch_end,
            file_chunks,
        } => {
            format!(
                "[{current_file}/{total_files}] embedding chunks {batch_start}-{batch_end}/{file_chunks}"
            )
        }
        IndexProgress::UpsertingBatch {
            current_file,
            total_files,
            batch_start,
            batch_end,
            file_chunks,
        } => {
            format!(
                "[{current_file}/{total_files}] upserting chunks {batch_start}-{batch_end}/{file_chunks}"
            )
        }
        IndexProgress::ChunkIndexed {
            current_file,
            total_files,
            file_chunks,
            total_chunks,
        } => {
            format!(
                "[{current_file}/{total_files}] embedded {file_chunks} chunks in file, {total_chunks} total chunks"
            )
        }
        IndexProgress::Finished {
            files_indexed,
            chunks_indexed,
        } => format!("Finished indexing {files_indexed} files and {chunks_indexed} chunks."),
    };
    rewrite_stderr_line(&message, last_line_len);
}

fn rewrite_stderr_line(message: &str, last_line_len: &mut usize) {
    let clear = " ".repeat(last_line_len.saturating_sub(message.len()));
    eprint!("\r{message}{clear}");
    let _ = io::stderr().flush();
    *last_line_len = message.len();
}

fn finish_progress_line(last_line_len: &mut usize) {
    if *last_line_len > 0 {
        eprintln!();
        *last_line_len = 0;
    }
}

fn truncate_middle(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }
    let keep_each_side = max_chars.saturating_sub(3) / 2;
    let start = value.chars().take(keep_each_side).collect::<String>();
    let end = value
        .chars()
        .rev()
        .take(keep_each_side)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{start}...{end}")
}

async fn require_indexed_repo(
    db: &Database,
    repo_name: &str,
    branch: &str,
) -> anyhow::Result<Uuid> {
    let repo = db
        .get_repository(repo_name, branch)
        .await?
        .ok_or_else(|| {
            anyhow!(
                "no index found for repo '{repo_name}' on branch '{branch}'. Run `insight index` first."
            )
        })?;
    let stats = db.repository_stats(repo.id).await?;
    if stats.chunks == 0 {
        return Err(anyhow!(
            "index for repo '{}' on branch '{}' has no chunks. Run `insight index` first.",
            repo.name,
            repo.branch
        ));
    }
    Ok(repo.id)
}

async fn print_status(db: &Database, repo_name: &str, branch: &str) -> anyhow::Result<()> {
    match db.get_repository(repo_name, branch).await? {
        Some(repo) => {
            let stats = db.repository_stats(repo.id).await?;
            println!("Repository: {}", repo.name);
            println!("Branch: {}", repo.branch);
            println!("Repo ID: {}", repo.id);
            println!(
                "Commit: {}",
                repo.commit_sha.as_deref().unwrap_or("unknown")
            );
            println!("Files indexed: {}", stats.files);
            println!("Chunks indexed: {}", stats.chunks);
        }
        None => {
            println!("No index found for repo '{repo_name}' on branch '{branch}'.");
            println!("Run `insight index` from the repo directory first.");
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
enum Command {
    Index,
    Refresh,
    Ask { question: String },
    Status,
    IndexAndAsk { question: String },
}

struct Args {
    command: Command,
    path: PathBuf,
    repo_name: Option<String>,
    branch: Option<String>,
    top_k: usize,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut raw = std::env::args().skip(1);
        let mut path = PathBuf::from(".");
        let mut repo_name = None;
        let mut branch = None;
        let mut top_k = 8usize;
        let mut command_name = None;
        let mut question_parts = Vec::new();

        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--path" => {
                    path =
                        PathBuf::from(raw.next().ok_or_else(|| anyhow!("--path needs a value"))?);
                }
                "--repo-name" => {
                    repo_name = Some(
                        raw.next()
                            .ok_or_else(|| anyhow!("--repo-name needs a value"))?,
                    );
                }
                "--branch" => {
                    branch = Some(
                        raw.next()
                            .ok_or_else(|| anyhow!("--branch needs a value"))?,
                    );
                }
                "--top-k" => {
                    top_k = raw
                        .next()
                        .ok_or_else(|| anyhow!("--top-k needs a value"))?
                        .parse()
                        .context("--top-k must be a number")?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "index" | "refresh" | "ask" | "status" if command_name.is_none() => {
                    command_name = Some(arg);
                }
                value => question_parts.push(value.to_string()),
            }
        }

        let question = question_parts.join(" ");
        let command = match command_name.as_deref() {
            Some("index") => Command::Index,
            Some("refresh") => Command::Refresh,
            Some("status") => Command::Status,
            Some("ask") => {
                if question.trim().is_empty() {
                    return Err(anyhow!("question is required for `insight ask`"));
                }
                Command::Ask { question }
            }
            None => {
                if question.trim().is_empty() {
                    print_help();
                    return Err(anyhow!("question is required"));
                }
                Command::IndexAndAsk { question }
            }
            Some(other) => return Err(anyhow!("unknown command: {other}")),
        };

        Ok(Self {
            command,
            path,
            repo_name,
            branch,
            top_k: top_k.clamp(1, 20),
        })
    }
}

fn infer_repo_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("codebase")
        .to_string()
}

fn infer_branch(path: &Path) -> String {
    Repository::discover(path)
        .ok()
        .and_then(|repo| repo.head().ok()?.shorthand().map(ToString::to_string))
        .unwrap_or_else(|| "main".to_string())
}

fn print_help() {
    eprintln!(
        "Usage:
  insight index [--path PATH] [--repo-name NAME] [--branch BRANCH]
  insight ask [--path PATH] [--repo-name NAME] [--branch BRANCH] [--top-k N] \"question\"
  insight status [--path PATH] [--repo-name NAME] [--branch BRANCH]
  insight refresh [--path PATH] [--repo-name NAME] [--branch BRANCH]
  insight [--path PATH] [--repo-name NAME] [--branch BRANCH] [--top-k N] \"question\"

Fast repeat workflow:
  cd /path/to/repo
  insight index
  insight ask \"Where is authentication handled?\"

The final one-shot form keeps the old behavior: it indexes first, then asks."
    );
}
