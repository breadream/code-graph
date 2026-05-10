use code_graph_shared::CodeChunk;
use regex::Regex;
use sha2::{Digest, Sha256};

const DEFAULT_MAX_CHARS: usize = 3_500;

#[derive(Debug, Clone)]
pub struct ChunkOptions {
    pub max_chars: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_MAX_CHARS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Markdown,
    Other(String),
}

impl Language {
    pub fn as_str(&self) -> &str {
        match self {
            Language::Rust => "rust",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
            Language::Python => "python",
            Language::Markdown => "markdown",
            Language::Other(v) => v,
        }
    }
}

pub fn detect_language(path: &str) -> Language {
    match path.rsplit('.').next().unwrap_or_default() {
        "rs" => Language::Rust,
        "ts" | "tsx" => Language::TypeScript,
        "js" | "jsx" => Language::JavaScript,
        "py" => Language::Python,
        "md" | "mdx" => Language::Markdown,
        other => Language::Other(other.to_string()),
    }
}

pub fn chunk_source(file_path: &str, content: &str, options: &ChunkOptions) -> Vec<CodeChunk> {
    let language = detect_language(file_path);
    let symbol_chunks = match language {
        Language::Rust => {
            chunk_by_symbols(file_path, content, language.as_str(), rust_symbol_regex())
        }
        Language::TypeScript | Language::JavaScript => {
            chunk_by_symbols(file_path, content, language.as_str(), ts_symbol_regex())
        }
        Language::Python => {
            chunk_by_symbols(file_path, content, language.as_str(), python_symbol_regex())
        }
        _ => Vec::new(),
    };

    if symbol_chunks.is_empty() {
        return fallback_line_chunks(file_path, content, language.as_str(), options.max_chars);
    }

    symbol_chunks
        .into_iter()
        .flat_map(|chunk| {
            if chunk.content.len() > options.max_chars {
                fallback_line_chunks(
                    file_path,
                    &chunk.content,
                    language.as_str(),
                    options.max_chars,
                )
                .into_iter()
                .map(move |mut sub| {
                    sub.start_line += chunk.start_line - 1;
                    sub.end_line += chunk.start_line - 1;
                    sub.symbol_name = chunk.symbol_name.clone();
                    sub.symbol_type = chunk.symbol_type.clone();
                    sub.content_hash = hash_content(&sub.content);
                    sub
                })
                .collect()
            } else {
                vec![chunk]
            }
        })
        .collect()
}

fn rust_symbol_regex() -> Regex {
    Regex::new(r"^\s*(pub\s+)?(async\s+)?(fn|struct|enum|trait|impl)\s+([A-Za-z0-9_]+)?").unwrap()
}

fn ts_symbol_regex() -> Regex {
    Regex::new(
        r"^\s*(export\s+)?(async\s+)?(function|class|interface|type|const)\s+([A-Za-z0-9_]+)",
    )
    .unwrap()
}

fn python_symbol_regex() -> Regex {
    Regex::new(r"^\s*(async\s+)?(def|class)\s+([A-Za-z0-9_]+)").unwrap()
}

fn chunk_by_symbols(file_path: &str, content: &str, language: &str, re: Regex) -> Vec<CodeChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut starts = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        if let Some(caps) = re.captures(line) {
            let symbol_type = caps
                .get(3)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().to_string());
            let symbol_name = caps
                .get(4)
                .or_else(|| caps.get(3))
                .map(|m| m.as_str().to_string());
            starts.push((idx, symbol_name, symbol_type));
        }
    }

    starts
        .iter()
        .enumerate()
        .filter_map(|(i, (start, symbol_name, symbol_type))| {
            let end_exclusive = starts
                .get(i + 1)
                .map(|(next, _, _)| *next)
                .unwrap_or(lines.len());
            let chunk_content = lines[*start..end_exclusive].join("\n");
            if chunk_content.trim().is_empty() {
                return None;
            }
            Some(make_chunk(
                file_path,
                language,
                symbol_name.clone(),
                symbol_type.clone(),
                *start + 1,
                end_exclusive,
                chunk_content,
            ))
        })
        .collect()
}

fn fallback_line_chunks(
    file_path: &str,
    content: &str,
    language: &str,
    max_chars: usize,
) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let mut buf = Vec::new();
    let mut start_line = 1usize;
    let mut current_len = 0usize;

    for (idx, line) in content.lines().enumerate() {
        let additional = line.len() + 1;
        if !buf.is_empty() && current_len + additional > max_chars {
            chunks.push(make_chunk(
                file_path,
                language,
                None,
                None,
                start_line,
                idx,
                buf.join("\n"),
            ));
            buf.clear();
            start_line = idx + 1;
            current_len = 0;
        }
        buf.push(line);
        current_len += additional;
    }

    if !buf.is_empty() {
        chunks.push(make_chunk(
            file_path,
            language,
            None,
            None,
            start_line,
            start_line + buf.len() - 1,
            buf.join("\n"),
        ));
    }

    chunks
}

fn make_chunk(
    file_path: &str,
    language: &str,
    symbol_name: Option<String>,
    symbol_type: Option<String>,
    start_line: usize,
    end_line: usize,
    content: String,
) -> CodeChunk {
    CodeChunk {
        id: None,
        repo_id: None,
        file_path: file_path.to_string(),
        language: language.to_string(),
        symbol_name,
        symbol_type,
        start_line: start_line as i32,
        end_line: end_line as i32,
        content_hash: hash_content(&content),
        content,
        vector_id: None,
    }
}

pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_rust_symbols() {
        let src = "pub fn alpha() {}\n\nstruct Beta {\n value: i32,\n}\n";
        let chunks = chunk_source("src/lib.rs", src, &ChunkOptions::default());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].symbol_name.as_deref(), Some("alpha"));
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[1].symbol_type.as_deref(), Some("struct"));
    }

    #[test]
    fn falls_back_to_line_chunks() {
        let src = "a\nb\nc\nd\n";
        let chunks = chunk_source("README.txt", src, &ChunkOptions { max_chars: 4 });
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_line, 1);
        assert!(!chunks[0].content_hash.is_empty());
    }
}
