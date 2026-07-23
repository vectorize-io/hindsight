//! Minimal Hindsight API client — only the knowledge-base read endpoints needed
//! to mirror a bank's knowledge base into a folder. Uses a blocking `reqwest`
//! client so the sync engine stays plain synchronous code.
//!
//! Two endpoints, fetched once per sync:
//!  - GET /knowledge-base/tree   → the folder/page hierarchy (no page bodies)
//!  - GET /knowledge-base/export → every page's OKF markdown in one bundle
//! We join them by page id, so a bank of any size is two HTTP calls.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;

/// A folder or page in the knowledge-base tree.
#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    /// "folder" or "page".
    pub kind: String,
    pub name: String,
    #[serde(default)]
    pub mission: Option<String>,
    #[serde(default)]
    pub managed: bool,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub children: Vec<KnowledgeNode>,
}

impl KnowledgeNode {
    pub fn is_folder(&self) -> bool {
        self.kind == "folder"
    }
}

/// The joined tree + page bodies for one bank.
pub struct KnowledgeSnapshot {
    /// Top-level folder/page nodes (each with nested `children`).
    pub roots: Vec<KnowledgeNode>,
    /// page id → its full OKF markdown document (frontmatter + body).
    pub content: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TreeResponse {
    #[serde(default)]
    roots: Vec<KnowledgeNode>,
}

#[derive(Debug, Deserialize)]
struct ExportFile {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ExportResponse {
    #[serde(default)]
    files: Vec<ExportFile>,
}

pub struct FsClient {
    api_url: String,
    api_token: Option<String>,
    http: reqwest::blocking::Client,
}

impl FsClient {
    pub fn new(api_url: &str, api_token: Option<&str>) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .user_agent(concat!("hindsight-cli-fs/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            api_url: api_url.trim_end_matches('/').to_string(),
            api_token: api_token.map(|s| s.to_string()),
            http,
        })
    }

    fn base(&self, bank_id: &str) -> String {
        format!(
            "{}/v1/default/banks/{}/knowledge-base",
            self.api_url,
            urlencode(bank_id)
        )
    }

    fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        what: &str,
        bank_id: &str,
    ) -> Result<T> {
        let mut req = self.http.get(url).header("Accept", "application/json");
        if let Some(token) = &self.api_token {
            req = req.bearer_auth(token);
        }
        let resp = req.send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            let snippet = body.chars().take(200).collect::<String>();
            return Err(anyhow!(
                "Failed to {} for bank \"{}\" (HTTP {}){}",
                what,
                bank_id,
                status.as_u16(),
                if snippet.is_empty() {
                    String::new()
                } else {
                    format!(": {}", snippet)
                }
            ));
        }
        Ok(resp.json::<T>()?)
    }

    /// Fetch the knowledge-base tree + page contents and join them.
    pub fn load_knowledge(&self, bank_id: &str) -> Result<KnowledgeSnapshot> {
        let base = self.base(bank_id);
        let tree: TreeResponse = self.get_json(
            &format!("{}/tree", base),
            "fetch knowledge-base tree",
            bank_id,
        )?;
        let bundle: ExportResponse = self.get_json(
            &format!("{}/export", base),
            "export knowledge base",
            bank_id,
        )?;

        // The bundle holds `<page-id>.md` (the page doc), `index.md`, and
        // `<page-id>.log.md` (history). We only want the page docs.
        let mut content = HashMap::new();
        for file in bundle.files {
            if file.path == "index.md" || file.path.ends_with(".log.md") {
                continue;
            }
            if let Some(id) = file.path.strip_suffix(".md") {
                content.insert(id.to_string(), file.content);
            }
        }

        Ok(KnowledgeSnapshot {
            roots: tree.roots,
            content,
        })
    }
}

/// Percent-encode a path segment (bank ids are usually URL-safe, but guard
/// against slashes/spaces without pulling in a URL-encoding crate).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}
