//! Persistent sync state for a mount. Tracks which file mirrors each knowledge
//! page (keyed by its relative path) and a content hash so unchanged pages are
//! not rewritten (keeps mtimes stable for editors, watchers, and `ls -la`),
//! plus the folder directories created, so removed folders are pruned.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::paths::{CONTROL_DIR, STATE_FILE};

/// One mirrored page: relative path + a content hash of the rendered document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    /// Path relative to the mount root (e.g. "policies/billing.md").
    pub file: String,
    /// sha256 of the rendered document.
    pub hash: String,
}

/// The persisted state file (`<dir>/.hindsight-fs/state.json`).
#[derive(Debug, Serialize, Deserialize)]
pub struct SyncState {
    pub version: u32,
    #[serde(rename = "bankId")]
    pub bank_id: String,
    #[serde(rename = "apiUrl")]
    pub api_url: String,
    #[serde(rename = "lastSyncAt")]
    pub last_sync_at: Option<String>,
    #[serde(rename = "lastSyncOk")]
    pub last_sync_ok: bool,
    #[serde(rename = "lastError")]
    pub last_error: Option<String>,
    /// Relative page path → file entry.
    pub files: BTreeMap<String, FileEntry>,
    /// Folder directories created (relative paths), for pruning removed folders.
    #[serde(default)]
    pub dirs: Vec<String>,
}

impl SyncState {
    pub fn empty(bank_id: &str, api_url: &str) -> Self {
        SyncState {
            version: 1,
            bank_id: bank_id.to_string(),
            api_url: api_url.to_string(),
            last_sync_at: None,
            last_sync_ok: false,
            last_error: None,
            files: BTreeMap::new(),
            dirs: Vec::new(),
        }
    }
}

pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn state_path(dir: &Path) -> PathBuf {
    dir.join(CONTROL_DIR).join(STATE_FILE)
}

pub fn load_state(dir: &Path, bank_id: &str, api_url: &str) -> SyncState {
    if let Ok(raw) = fs::read_to_string(state_path(dir)) {
        if let Ok(state) = serde_json::from_str::<SyncState>(&raw) {
            if state.version == 1 {
                return state;
            }
        }
    }
    SyncState::empty(bank_id, api_url)
}

pub fn save_state(dir: &Path, state: &SyncState) -> Result<()> {
    let cdir = dir.join(CONTROL_DIR);
    fs::create_dir_all(&cdir)?;
    let mut body = serde_json::to_string_pretty(state)?;
    body.push('\n');
    fs::write(state_path(dir), body)?;
    Ok(())
}
