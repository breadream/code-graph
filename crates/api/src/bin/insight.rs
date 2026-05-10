use anyhow::{anyhow, Context};
use code_graph_db::Database;
use code_graph_indexer::Indexer;
use code_graph_llm::build_providers;
use code_graph_retriever::{QdrantClient, Retriever};
use code_graph_shared::{AppConfig, RepoRequest};
use git2::Repository;
use std::path::{Path, PathBuf};

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
        .unwrap_or_else(|| infer_repo_name(&repo_path));
    let branch = args.branch.unwrap_or_else(|| infer_branch(&repo_path));

    let config = AppConfig::from_env();
    let db = Database::connect(&config.database_url).await?;
    let providers = build_providers(&config.provider, config.embedding_dim)?;
    let qdrant = QdrantClient::new(
        config.qdrant_url,
        config.qdrant_collection,
        providers.embeddings.dimensions(),
    );

    let indexer = Indexer::new(db.clone(), qdrant.clone(), providers.embeddings.clone());
    let retriever = Retriever::new(db.clone(), qdrant);

    println!("Indexing {} ({})...", repo_path.display(), branch);
    let summary = indexer
        .index_repo(RepoRequest {
            repo_name,
            branch: Some(branch),
            repo_url: None,
            local_path: Some(repo_path.to_string_lossy().to_string()),
        })
        .await?;
    println!(
        "Indexed {} files and {} chunks. repo_id={}",
        summary.files_indexed, summary.chunks_indexed, summary.repo_id
    );

    let query_vector = providers.embeddings.embed(&args.question).await?;
    let hits = retriever
        .retrieve(summary.repo_id, query_vector, &args.question, args.top_k)
        .await?;
    let answer = providers.chat.answer(&args.question, &hits).await?;
    db.log_query(summary.repo_id, &args.question, &answer)
        .await?;

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

struct Args {
    question: String,
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
                value => question_parts.push(value.to_string()),
            }
        }

        let question = question_parts.join(" ");
        if question.trim().is_empty() {
            print_help();
            return Err(anyhow!("question is required"));
        }

        Ok(Self {
            question,
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
        "Usage: insight [--path PATH] [--repo-name NAME] [--branch BRANCH] [--top-k N] \"question\"\n\nExample:\n  cd /path/to/repo\n  insight \"Where is authentication handled?\""
    );
}
