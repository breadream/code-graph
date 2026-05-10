use code_graph_shared::{CodeChunk, SearchHit};
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[derive(Clone)]
pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        Ok(Self {
            pool: PgPool::connect(database_url).await?,
        })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    pub async fn upsert_repository(
        &self,
        name: &str,
        url: Option<&str>,
        branch: &str,
        commit_sha: Option<&str>,
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        let row = sqlx::query(
            r#"
            INSERT INTO repositories (id, name, url, branch, commit_sha)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (name, branch) DO UPDATE
              SET url = EXCLUDED.url,
                  commit_sha = EXCLUDED.commit_sha,
                  updated_at = now()
            RETURNING id
            "#,
        )
        .bind(id)
        .bind(name)
        .bind(url)
        .bind(branch)
        .bind(commit_sha)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get("id"))
    }

    pub async fn upsert_file(
        &self,
        repo_id: Uuid,
        path: &str,
        language: &str,
        content_hash: &str,
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        let row = sqlx::query(
            r#"
            INSERT INTO files (id, repo_id, path, language, content_hash)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (repo_id, path) DO UPDATE
              SET language = EXCLUDED.language,
                  content_hash = EXCLUDED.content_hash,
                  updated_at = now()
            RETURNING id
            "#,
        )
        .bind(id)
        .bind(repo_id)
        .bind(path)
        .bind(language)
        .bind(content_hash)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get("id"))
    }

    pub async fn insert_chunk(
        &self,
        repo_id: Uuid,
        file_id: Uuid,
        chunk: &CodeChunk,
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        let vector_id = chunk.vector_id.unwrap_or_else(Uuid::new_v4);
        sqlx::query(
            r#"
            INSERT INTO chunks
              (id, repo_id, file_id, file_path, language, symbol_name, symbol_type,
               start_line, end_line, content, content_hash, vector_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (repo_id, file_path, content_hash) DO UPDATE
              SET symbol_name = EXCLUDED.symbol_name,
                  symbol_type = EXCLUDED.symbol_type,
                  start_line = EXCLUDED.start_line,
                  end_line = EXCLUDED.end_line,
                  content = EXCLUDED.content,
                  vector_id = EXCLUDED.vector_id
            "#,
        )
        .bind(id)
        .bind(repo_id)
        .bind(file_id)
        .bind(&chunk.file_path)
        .bind(&chunk.language)
        .bind(&chunk.symbol_name)
        .bind(&chunk.symbol_type)
        .bind(chunk.start_line)
        .bind(chunk.end_line)
        .bind(&chunk.content)
        .bind(&chunk.content_hash)
        .bind(vector_id)
        .execute(&self.pool)
        .await?;

        let row = sqlx::query(
            "SELECT id FROM chunks WHERE repo_id = $1 AND file_path = $2 AND content_hash = $3",
        )
        .bind(repo_id)
        .bind(&chunk.file_path)
        .bind(&chunk.content_hash)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get("id"))
    }

    pub async fn get_chunk(&self, chunk_id: Uuid) -> anyhow::Result<SearchHit> {
        let row = sqlx::query(
            r#"
            SELECT id, repo_id, file_path, language, start_line, end_line, content
            FROM chunks WHERE id = $1
            "#,
        )
        .bind(chunk_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(SearchHit {
            chunk_id: row.get("id"),
            repo_id: row.get("repo_id"),
            file_path: row.get("file_path"),
            language: row.get("language"),
            start_line: row.get("start_line"),
            end_line: row.get("end_line"),
            content: row.get("content"),
            score: 0.0,
            method: "db".to_string(),
        })
    }

    pub async fn keyword_search(
        &self,
        repo_id: Uuid,
        question: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let terms = search_terms(question);
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let rows = sqlx::query(
            r#"
            SELECT id, repo_id, file_path, language, start_line, end_line, content
            FROM chunks
            WHERE repo_id = $1
            ORDER BY updated_at DESC
            LIMIT 2000
            "#,
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;

        let mut hits = rows
            .into_iter()
            .filter_map(|row| {
                let file_path: String = row.get("file_path");
                let language: String = row.get("language");
                let content: String = row.get("content");
                let score = keyword_score(&terms, &file_path, &language, &content);
                if score <= 0.0 {
                    return None;
                }
                Some(SearchHit {
                    chunk_id: row.get("id"),
                    repo_id: row.get("repo_id"),
                    file_path,
                    language,
                    start_line: row.get("start_line"),
                    end_line: row.get("end_line"),
                    content,
                    score,
                    method: "keyword".to_string(),
                })
            })
            .collect::<Vec<_>>();

        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit as usize);
        Ok(hits)
    }

    pub async fn log_query(
        &self,
        repo_id: Uuid,
        question: &str,
        answer: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO query_logs (id, repo_id, question, answer) VALUES ($1, $2, $3, $4)",
        )
        .bind(Uuid::new_v4())
        .bind(repo_id)
        .bind(question)
        .bind(answer)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

fn search_terms(question: &str) -> Vec<String> {
    let stopwords = [
        "a", "an", "and", "are", "can", "does", "for", "from", "how", "in", "is", "it", "of", "on",
        "or", "the", "to", "what", "when", "where", "which", "who", "why",
    ];
    let mut terms = Vec::new();
    for raw in question
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(str::to_lowercase)
        .filter(|token| token.len() > 2 && !stopwords.contains(&token.as_str()))
    {
        push_unique(&mut terms, raw.clone());
        push_unique(&mut terms, stem(&raw));
        match raw.as_str() {
            "repository" | "repositories" => push_unique(&mut terms, "repo".to_string()),
            "repo" | "repos" => push_unique(&mut terms, "repository".to_string()),
            "authentication" | "authenticate" | "authorization" => {
                push_unique(&mut terms, "auth".to_string());
            }
            "index" | "indexing" | "indexed" => {
                push_unique(&mut terms, "indexer".to_string());
                push_unique(&mut terms, "index_repo".to_string());
            }
            _ => {}
        }
    }
    terms
}

fn push_unique(terms: &mut Vec<String>, term: String) {
    if term.len() > 2 && !terms.contains(&term) {
        terms.push(term);
    }
}

fn stem(token: &str) -> String {
    for suffix in ["ing", "ied", "ed", "es", "s"] {
        if token.len() > suffix.len() + 3 && token.ends_with(suffix) {
            return token.trim_end_matches(suffix).to_string();
        }
    }
    token.to_string()
}

fn keyword_score(terms: &[String], file_path: &str, language: &str, content: &str) -> f32 {
    let path = file_path.to_lowercase();
    let body = content.to_lowercase();
    let mut score = 0.0;

    for term in terms {
        let path_hits = path.matches(term).count() as f32;
        let body_hits = body.matches(term).count().min(8) as f32;
        score += path_hits * 1.4;
        score += body_hits * 0.3;
    }

    if path.ends_with(".md") && !terms.iter().any(|term| term == "readme" || term == "docs") {
        score *= 0.45;
    }
    if language != "markdown" {
        score += 0.2;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_code_search_terms() {
        let terms = search_terms("Where is repository indexing implemented?");
        assert!(terms.contains(&"repo".to_string()));
        assert!(terms.contains(&"index".to_string()));
        assert!(terms.contains(&"indexer".to_string()));
    }

    #[test]
    fn scores_code_paths_above_readme_for_code_questions() {
        let terms = search_terms("Where is repository indexing implemented?");
        let code_score = keyword_score(
            &terms,
            "crates/indexer/src/lib.rs",
            "rust",
            "pub async fn index_repo(repo: RepoRequest) {}",
        );
        let readme_score = keyword_score(
            &terms,
            "README.md",
            "markdown",
            "Repository indexing docs mention index commands.",
        );
        assert!(code_score > readme_score);
    }
}
