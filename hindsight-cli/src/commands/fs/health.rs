//! Health assessment for a mount — used by `status` (human + JSON) and
//! reusable for programmatic healthchecks/watchdogs.
//!
//! Two orthogonal signals are combined into one verdict:
//!  - liveness: is the daemon process actually alive?
//!  - freshness: did a sync succeed recently (within the stale threshold)?

use chrono::Utc;
use serde::Serialize;

use super::config::MountConfig;
use super::daemon::daemon_status;
use super::state::load_state;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Ok,
    Stale,
    Failed,
    Dead,
}

impl HealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Ok => "ok",
            HealthStatus::Stale => "stale",
            HealthStatus::Failed => "failed",
            HealthStatus::Dead => "dead",
        }
    }
}

/// A serializable health report (used for `--output json`).
#[derive(Debug, Serialize)]
pub struct HealthReport {
    pub healthy: bool,
    pub status: String,
    pub mount: String,
    pub bank: String,
    pub api_url: String,
    pub mode: String,
    pub daemon_running: bool,
    pub daemon_pid: Option<i32>,
    pub daemon_started_at: Option<String>,
    pub daemon_interval_seconds: Option<u64>,
    pub last_sync_at: Option<String>,
    pub last_sync_ok: bool,
    pub last_sync_age_seconds: Option<i64>,
    pub last_error: Option<String>,
    pub stale_after_seconds: u64,
    pub mirrored_files: usize,
}

/// Compute a health report for the mount at `config.dir`.
///
/// `stale_after_seconds` overrides the default threshold of
/// `max(interval * 3, 15)`.
pub fn compute_health(config: &MountConfig, stale_after_override: Option<u64>) -> HealthReport {
    let ds = daemon_status(&config.dir);
    let state = load_state(&config.dir, &config.bank_id, &config.api_url);

    let interval_seconds = ds
        .record
        .as_ref()
        .map(|r| r.interval_seconds)
        .unwrap_or(config.interval_seconds);
    let stale_after_seconds =
        stale_after_override.unwrap_or_else(|| std::cmp::max(interval_seconds * 3, 15));

    let age_seconds = state.last_sync_at.as_deref().and_then(|ts| {
        chrono::DateTime::parse_from_rfc3339(ts).ok().map(|parsed| {
            let secs = (Utc::now() - parsed.with_timezone(&Utc)).num_seconds();
            secs.max(0)
        })
    });

    let status = if !ds.running {
        HealthStatus::Dead
    } else if state.last_sync_at.is_none() {
        HealthStatus::Stale // up but hasn't completed a first sync yet
    } else if !state.last_sync_ok {
        HealthStatus::Failed // looping but the API keeps erroring
    } else if age_seconds
        .map(|a| a >= stale_after_seconds as i64)
        .unwrap_or(true)
    {
        HealthStatus::Stale // alive but wedged — no fresh sync
    } else {
        HealthStatus::Ok
    };

    HealthReport {
        healthy: status == HealthStatus::Ok,
        status: status.as_str().to_string(),
        mount: config.dir.display().to_string(),
        bank: if state.bank_id.is_empty() {
            config.bank_id.clone()
        } else {
            state.bank_id.clone()
        },
        api_url: if state.api_url.is_empty() {
            config.api_url.clone()
        } else {
            state.api_url.clone()
        },
        mode: if config.writable {
            "writable".to_string()
        } else {
            "read-only".to_string()
        },
        daemon_running: ds.running,
        daemon_pid: ds.pid,
        daemon_started_at: ds.record.as_ref().map(|r| r.started_at.clone()),
        daemon_interval_seconds: ds.record.as_ref().map(|r| r.interval_seconds),
        last_sync_at: state.last_sync_at.clone(),
        last_sync_ok: state.last_sync_ok,
        last_sync_age_seconds: age_seconds,
        last_error: state.last_error.clone(),
        stale_after_seconds,
        mirrored_files: state.files.len(),
    }
}
