//! The sync engine: fetch a bank's knowledge base (folder/page tree + page
//! contents) and mirror it as a nested folder of markdown files in the mount
//! directory — folders become directories, pages become `.md` files.
//!
//! The mirror is strictly one-way (API → disk). Two mechanisms enforce that:
//!
//!  1. Mirrored files are written read-only (mode 0444 unless `writable`), so an
//!     agent's in-place edit or editor-save fails with EACCES.
//!  2. Every pass compares the *on-disk* bytes against the freshly rendered
//!     content, so any drift — a tampered file, a force-chmod edit, a partial
//!     write — is reverted on the next tick, even when the page is unchanged
//!     server-side.

use anyhow::Result;
use chrono::Utc;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::client::FsClient;
use super::config::MountConfig;
use super::format::{plan_mirror, render_index};
use super::paths::{CONTROL_DIR, INDEX_FILE};
use super::state::{hash_content, load_state, save_state, FileEntry, SyncState};

#[cfg(unix)]
const READONLY_MODE: u32 = 0o444;
#[cfg(unix)]
const WRITABLE_MODE: u32 = 0o644;

/// Summary of a single sync pass.
pub struct SyncResult {
    /// Pages mirrored.
    pub total: usize,
    /// Folder directories in the mirror.
    pub folders: usize,
    /// Files (re)written because they were new, changed, or tampered with.
    pub written: usize,
    /// Files left untouched because disk already matched the API.
    pub unchanged: usize,
    /// Files removed because their page no longer exists in the bank.
    pub removed: usize,
    /// Subset of `written` that were rewritten because the on-disk copy drifted.
    pub reverted: usize,
}

fn mode_for(config: &MountConfig) -> u32 {
    #[cfg(unix)]
    {
        if config.writable {
            WRITABLE_MODE
        } else {
            READONLY_MODE
        }
    }
    #[cfg(not(unix))]
    {
        let _ = config;
        0
    }
}

#[cfg(unix)]
fn set_mode(file: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(file, fs::Permissions::from_mode(mode));
}
#[cfg(not(unix))]
fn set_mode(_file: &Path, _mode: u32) {}

/// Write `content` to `file` atomically (temp file + rename within the same dir).
fn atomic_write(file: &Path, content: &str, mode: u32) -> Result<()> {
    let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&tmp, content)?;
    fs::rename(&tmp, file)?;
    set_mode(file, mode);
    Ok(())
}

fn read_file_or_none(file: &Path) -> Option<String> {
    fs::read_to_string(file).ok()
}

/// Perform a single sync pass.
///
/// Pruning of files/folders for removed pages happens only after a successful
/// fetch, so a transient API/network error never wipes the existing mirror.
pub fn run_sync(config: &MountConfig) -> Result<SyncResult> {
    let synced_at = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let mode = mode_for(config);
    fs::create_dir_all(config.dir.join(CONTROL_DIR))?;

    let mut state = load_state(&config.dir, &config.bank_id, &config.api_url);
    let client = FsClient::new(&config.api_url, config.api_token.as_deref())?;

    let snapshot = match client.load_knowledge(&config.bank_id) {
        Ok(s) => s,
        Err(err) => {
            state.last_sync_at = Some(synced_at);
            state.last_sync_ok = false;
            state.last_error = Some(err.to_string());
            let _ = save_state(&config.dir, &state);
            return Err(err);
        }
    };

    let plan = plan_mirror(&snapshot);

    // Create folder directories first (parents before children — plan.dirs is
    // in tree order). The control dir is excluded by the .hindsight-fs prefix.
    for dir in &plan.dirs {
        fs::create_dir_all(config.dir.join(dir))?;
    }

    let mut written = 0;
    let mut unchanged = 0;
    let mut reverted = 0;
    let mut next_files: BTreeMap<String, FileEntry> = BTreeMap::new();

    for page in &plan.files {
        let hash = hash_content(&page.content);
        let prev = state.files.get(&page.rel_path);
        let abs_path = config.dir.join(&page.rel_path);
        if let Some(parent) = abs_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Source of truth is the bytes on disk — so a local edit is detected
        // and overwritten even when the page is identical to last time.
        let on_disk = read_file_or_none(&abs_path);
        match &on_disk {
            Some(disk) if disk == &page.content => {
                set_mode(&abs_path, mode);
                unchanged += 1;
            }
            _ => {
                atomic_write(&abs_path, &page.content, mode)?;
                written += 1;
                if on_disk.is_some() && prev.map(|p| p.hash == hash).unwrap_or(false) {
                    reverted += 1;
                }
            }
        }
        next_files.insert(
            page.rel_path.clone(),
            FileEntry {
                file: page.rel_path.clone(),
                hash,
            },
        );
    }

    // Prune files whose pages no longer exist.
    let mut removed = 0;
    for (rel, entry) in &state.files {
        if !next_files.contains_key(rel) {
            safe_unlink(&config.dir.join(&entry.file));
            removed += 1;
        }
    }

    // Prune folder directories that no longer exist (deepest first so they empty
    // out before removal); only remove ones we created and that are now gone.
    let live_dirs: std::collections::HashSet<&String> = plan.dirs.iter().collect();
    let mut gone_dirs: Vec<&String> = state
        .dirs
        .iter()
        .filter(|d| !live_dirs.contains(*d))
        .collect();
    gone_dirs.sort_by(|a, b| b.len().cmp(&a.len()));
    for dir in gone_dirs {
        let _ = fs::remove_dir(config.dir.join(dir)); // only succeeds if empty
    }

    let new_state = SyncState {
        version: 1,
        bank_id: config.bank_id.clone(),
        api_url: config.api_url.clone(),
        last_sync_at: Some(synced_at),
        last_sync_ok: true,
        last_error: None,
        files: next_files,
        dirs: plan.dirs.clone(),
    };
    save_state(&config.dir, &new_state)?;

    atomic_write(
        &config.dir.join(CONTROL_DIR).join(INDEX_FILE),
        &render_index(&snapshot, &config.bank_id, &config.api_url),
        mode,
    )?;

    Ok(SyncResult {
        total: plan.page_count,
        folders: plan.folder_count,
        written,
        unchanged,
        removed,
        reverted,
    })
}

fn safe_unlink(p: &Path) {
    // Drop the read-only bit first so the unlink succeeds.
    set_mode(p, mode_writable());
    let _ = fs::remove_file(p);
}

fn mode_writable() -> u32 {
    #[cfg(unix)]
    {
        WRITABLE_MODE
    }
    #[cfg(not(unix))]
    {
        0
    }
}
