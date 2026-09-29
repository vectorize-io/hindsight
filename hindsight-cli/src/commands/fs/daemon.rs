//! Background daemon management. `start` spawns a detached copy of this CLI in
//! `fs run` mode (the hidden loop entrypoint), writing a pidfile and a log under
//! the mount's control directory. `stop` signals it; `status` reports liveness.
//!
//! POSIX-only: the detached refresh loop relies on `setsid`/`kill`. On non-unix
//! targets the daemon commands return an error rather than pretending to work.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use super::config::{save_config, MountConfig};
use super::paths::{CONTROL_DIR, LOG_FILE, PID_FILE};

/// Persisted daemon metadata (`<dir>/.hindsight-fs/daemon.pid`).
#[derive(Debug, Serialize, Deserialize)]
pub struct PidRecord {
    pub pid: i32,
    #[serde(rename = "bankId")]
    pub bank_id: String,
    #[serde(rename = "intervalSeconds")]
    pub interval_seconds: u64,
    #[serde(rename = "startedAt")]
    pub started_at: String,
}

fn control_path(dir: &Path, file: &str) -> PathBuf {
    dir.join(CONTROL_DIR).join(file)
}

pub fn log_path(dir: &Path) -> PathBuf {
    control_path(dir, LOG_FILE)
}

/// True if a process with `pid` is currently alive.
#[cfg(unix)]
pub fn is_alive(pid: i32) -> bool {
    // kill(pid, 0): 0 => alive; EPERM => alive but not ours; ESRCH => gone.
    let rc = unsafe { libc::kill(pid, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}
#[cfg(not(unix))]
pub fn is_alive(_pid: i32) -> bool {
    false
}

pub fn read_pid_record(dir: &Path) -> Option<PidRecord> {
    let raw = fs::read_to_string(control_path(dir, PID_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

fn remove_pid_file(dir: &Path) {
    let _ = fs::remove_file(control_path(dir, PID_FILE));
}

/// Outcome of a `start` request.
pub struct StartResult {
    pub pid: i32,
    pub already_running: bool,
}

#[cfg(unix)]
pub fn start_daemon(config: &MountConfig) -> Result<StartResult> {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    if let Some(existing) = read_pid_record(&config.dir) {
        if is_alive(existing.pid) {
            return Ok(StartResult {
                pid: existing.pid,
                already_running: true,
            });
        }
    }

    fs::create_dir_all(config.dir.join(CONTROL_DIR))?;
    save_config(config)?;

    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(control_path(&config.dir, LOG_FILE))?;
    let log_err = log.try_clone()?;

    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(exe);
    // `dir` is the positional arg on the hidden `fs run` subcommand.
    cmd.arg("fs")
        .arg("run")
        .arg(&config.dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    // Detach from the controlling terminal so the loop survives shell exit.
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }

    let child = cmd.spawn()?;
    let pid = child.id() as i32;
    // Intentionally do not wait — the child is detached and outlives us.

    let record = PidRecord {
        pid,
        bank_id: config.bank_id.clone(),
        interval_seconds: config.interval_seconds,
        started_at: now_iso(),
    };
    let mut body = serde_json::to_string_pretty(&record)?;
    body.push('\n');
    fs::write(control_path(&config.dir, PID_FILE), body)?;

    Ok(StartResult {
        pid,
        already_running: false,
    })
}
#[cfg(not(unix))]
pub fn start_daemon(_config: &MountConfig) -> Result<StartResult> {
    Err(anyhow::anyhow!(
        "Background daemon mode is only supported on Unix. Use `hindsight fs mount` (foreground) instead."
    ))
}

/// Outcome of a `stop` request.
pub struct StopResult {
    pub stopped: bool,
    pub pid: Option<i32>,
}

#[cfg(unix)]
pub fn stop_daemon(dir: &Path) -> Result<StopResult> {
    let record = match read_pid_record(dir) {
        Some(r) => r,
        None => {
            return Ok(StopResult {
                stopped: false,
                pid: None,
            })
        }
    };

    if !is_alive(record.pid) {
        remove_pid_file(dir);
        return Ok(StopResult {
            stopped: false,
            pid: Some(record.pid),
        });
    }

    unsafe {
        libc::kill(record.pid, libc::SIGTERM);
    }

    // Give it a moment to exit, then force-kill if needed.
    for _ in 0..50 {
        if !is_alive(record.pid) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if is_alive(record.pid) {
        unsafe {
            libc::kill(record.pid, libc::SIGKILL);
        }
    }

    remove_pid_file(dir);
    Ok(StopResult {
        stopped: true,
        pid: Some(record.pid),
    })
}
#[cfg(not(unix))]
pub fn stop_daemon(_dir: &Path) -> Result<StopResult> {
    Ok(StopResult {
        stopped: false,
        pid: None,
    })
}

/// Liveness snapshot for `status`/health.
pub struct DaemonStatus {
    pub running: bool,
    pub pid: Option<i32>,
    pub record: Option<PidRecord>,
}

pub fn daemon_status(dir: &Path) -> DaemonStatus {
    match read_pid_record(dir) {
        None => DaemonStatus {
            running: false,
            pid: None,
            record: None,
        },
        Some(record) => {
            let running = is_alive(record.pid);
            if !running {
                remove_pid_file(dir);
            }
            DaemonStatus {
                running,
                pid: if running { Some(record.pid) } else { None },
                record: Some(record),
            }
        }
    }
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
