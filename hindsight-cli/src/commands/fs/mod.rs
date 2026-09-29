//! `hindsight fs` — mirror a Hindsight bank's knowledge base (folders + pages)
//! as a folder of markdown files that stay current via a background refresh
//! loop. Once mounted, ordinary shell tools (ls, cat, grep, find, rg, fzf …)
//! work against real files.
//!
//! Ported from the standalone `hindsight-fs` TS tool into the Rust CLI.

mod client;
mod config;
mod daemon;
mod format;
mod health;
mod paths;
mod state;
mod sync;

use anyhow::Result;
use clap::{Args, Subcommand};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::output::OutputFormat;
use config::{resolve_config, save_config, Baseline, ConfigOverrides, MountConfig};

/// Options shared by every `fs` subcommand.
#[derive(Args, Debug, Clone)]
pub struct FsCommonArgs {
    /// Mount directory (default: ./hindsight-fs; env: HINDSIGHT_FS_DIR)
    pub dir: Option<String>,

    /// Bank to mirror (env: HINDSIGHT_BANK_ID)
    #[arg(short = 'b', long)]
    pub bank: Option<String>,

    /// API base URL override (defaults to the CLI's configured endpoint)
    #[arg(short = 'u', long = "api-url")]
    pub api_url: Option<String>,

    /// Bearer token override
    #[arg(short = 't', long)]
    pub token: Option<String>,

    /// Refresh interval in seconds (default: 30)
    #[arg(short = 'i', long)]
    pub interval: Option<u64>,

    /// Make mirrored files editable (default: read-only; still one-way)
    #[arg(long)]
    pub writable: bool,
}

/// `hindsight fs` subcommands.
#[derive(Subcommand, Debug)]
pub enum FsCommands {
    /// Mirror the bank into <dir> and keep it refreshed (foreground; Ctrl-C to stop)
    Mount {
        #[command(flatten)]
        common: FsCommonArgs,
        /// Run a single pass instead of looping
        #[arg(long)]
        once: bool,
        /// Run in the background (alias for `fs start`)
        #[arg(long)]
        detach: bool,
    },
    /// Mount in the background (alias for: mount --detach)
    Start {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Stop the background daemon for <dir>
    Stop {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Restart the background daemon
    Restart {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Run a single refresh pass and exit
    Sync {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Show daemon + last-sync health for <dir> (exits non-zero when unhealthy)
    Status {
        #[command(flatten)]
        common: FsCommonArgs,
        /// Seconds before a sync is considered "stale" (default: max(interval×3, 15))
        #[arg(long = "stale-after")]
        stale_after: Option<u64>,
    },
    /// List the bank's knowledge-base folders + pages (no files written)
    List {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Print the tail of the background daemon log
    Logs {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Stop the daemon and delete mirrored files + control data
    Unmount {
        #[command(flatten)]
        common: FsCommonArgs,
    },
    /// Hidden daemon loop entrypoint (used internally by `fs start`)
    #[command(hide = true)]
    Run {
        #[command(flatten)]
        common: FsCommonArgs,
    },
}

fn overrides_from(common: &FsCommonArgs) -> ConfigOverrides {
    ConfigOverrides {
        dir: common.dir.clone(),
        api_url: common.api_url.clone(),
        api_token: common.token.clone(),
        bank_id: common.bank.clone(),
        interval_seconds: common.interval,
        writable: common.writable,
    }
}

/// Dispatch an `fs` subcommand. `base_api_url`/`base_api_key` are the CLI's
/// already-resolved endpoint (profile/env/default), used as the baseline below
/// explicit flags and any saved mount config.
pub fn dispatch(
    command: FsCommands,
    base_api_url: &str,
    base_api_key: Option<&str>,
    output_format: OutputFormat,
) -> Result<()> {
    let baseline = Baseline {
        api_url: base_api_url.to_string(),
        api_key: base_api_key.map(|s| s.to_string()),
    };

    match command {
        FsCommands::Mount {
            common,
            once,
            detach,
        } => cmd_mount(&common, once, detach, &baseline),
        FsCommands::Start { common } => cmd_start(&common, &baseline),
        FsCommands::Stop { common } => cmd_stop(&common, &baseline),
        FsCommands::Restart { common } => {
            cmd_stop(&common, &baseline)?;
            cmd_start(&common, &baseline)
        }
        FsCommands::Sync { common } => cmd_sync(&common, &baseline),
        FsCommands::Status {
            common,
            stale_after,
        } => cmd_status(&common, stale_after, &baseline, output_format),
        FsCommands::List { common } => cmd_list(&common, &baseline),
        FsCommands::Logs { common } => cmd_logs(&common, &baseline),
        FsCommands::Unmount { common } => cmd_unmount(&common, &baseline),
        FsCommands::Run { common } => cmd_run(&common, &baseline),
    }
}

// ── Refresh loop ───────────────────────────────────────

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_signal(_sig: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

#[cfg(unix)]
fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
}
#[cfg(not(unix))]
fn install_signal_handlers() {}

fn stamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// The refresh loop shared by foreground `mount` and the background daemon:
/// an immediate sync, then repeat every `interval_seconds` until stopped.
fn run_loop<F: Fn(&str)>(config: &MountConfig, log: F) {
    install_signal_handlers();
    log(&format!(
        "mounting bank \"{}\" at {} (every {}s, {})",
        config.bank_id,
        config.dir.display(),
        config.interval_seconds,
        config.api_url
    ));

    while !STOP.load(Ordering::SeqCst) {
        match sync::run_sync(config) {
            Ok(r) => {
                let reverted = if r.reverted > 0 {
                    format!(", {} reverted", r.reverted)
                } else {
                    String::new()
                };
                log(&format!(
                    "synced {} pages / {} folders — {} updated, {} unchanged, {} removed{}",
                    r.total, r.folders, r.written, r.unchanged, r.removed, reverted
                ));
            }
            Err(e) => log(&format!("sync failed: {}", e)),
        }
        if STOP.load(Ordering::SeqCst) {
            break;
        }
        // Sleep the interval in small increments so a signal stops us promptly.
        let mut slept = 0u64;
        let target = config.interval_seconds * 1000;
        while slept < target && !STOP.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(100));
            slept += 100;
        }
    }

    log("mount stopped");
}

// ── Commands ───────────────────────────────────────────

fn cmd_sync(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, true)?;
    save_config(&config)?;
    let r = sync::run_sync(&config)?;
    let reverted = if r.reverted > 0 {
        format!(", {} reverted", r.reverted)
    } else {
        String::new()
    };
    println!(
        "Synced {} pages / {} folders into {} ({} updated, {} unchanged, {} removed{})",
        r.total,
        r.folders,
        config.dir.display(),
        r.written,
        r.unchanged,
        r.removed,
        reverted
    );
    Ok(())
}

fn cmd_mount(common: &FsCommonArgs, once: bool, detach: bool, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, true)?;
    save_config(&config)?;

    if once {
        return cmd_sync(common, baseline);
    }
    if detach {
        return start_background(&config);
    }

    // Foreground: run until Ctrl-C.
    run_loop(&config, |m| eprintln!("[{}] {}", stamp(), m));
    Ok(())
}

fn cmd_start(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, true)?;
    save_config(&config)?;
    start_background(&config)
}

fn start_background(config: &MountConfig) -> Result<()> {
    let r = daemon::start_daemon(config)?;
    if r.already_running {
        println!(
            "Already mounted at {} (daemon pid {}).",
            config.dir.display(),
            r.pid
        );
    } else {
        println!(
            "Mounted bank \"{}\" at {} in background (pid {}).",
            config.bank_id,
            config.dir.display(),
            r.pid
        );
        println!(
            "Logs: {} — stop with: hindsight fs stop {}",
            daemon::log_path(&config.dir).display(),
            config.dir.display()
        );
    }
    Ok(())
}

fn cmd_stop(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, false)?;
    let r = daemon::stop_daemon(&config.dir)?;
    if r.stopped {
        println!(
            "Stopped daemon (pid {}) for {}.",
            r.pid.unwrap_or(0),
            config.dir.display()
        );
    } else if let Some(pid) = r.pid {
        println!(
            "Daemon for {} was not running (cleaned up stale pid {}).",
            config.dir.display(),
            pid
        );
    } else {
        println!("No daemon registered for {}.", config.dir.display());
    }
    Ok(())
}

fn cmd_status(
    common: &FsCommonArgs,
    stale_after: Option<u64>,
    baseline: &Baseline,
    output_format: OutputFormat,
) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, false)?;
    let report = health::compute_health(&config, stale_after);

    if output_format == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let age = match report.last_sync_age_seconds {
            None => "never".to_string(),
            Some(a) => format!("{}s ago", a),
        };
        println!("Mount:    {}", report.mount);
        println!(
            "Bank:     {}",
            if report.bank.is_empty() {
                "(unset)"
            } else {
                &report.bank
            }
        );
        println!("API:      {}", report.api_url);
        println!(
            "Mode:     {}",
            if report.mode == "writable" {
                "writable (one-way; edits reverted on refresh)"
            } else {
                "read-only (edits blocked)"
            }
        );
        println!(
            "Daemon:   {}",
            match report.daemon_pid {
                Some(pid) if report.daemon_running => format!("running (pid {})", pid),
                _ => "stopped".to_string(),
            }
        );
        if let Some(started) = &report.daemon_started_at {
            println!(
                "Interval: {}s (started {})",
                report.daemon_interval_seconds.unwrap_or(0),
                started
            );
        }
        println!(
            "Last sync: {} ({}){}",
            report.last_sync_at.as_deref().unwrap_or("never"),
            age,
            if report.last_sync_ok { "" } else { " (FAILED)" }
        );
        if let Some(err) = &report.last_error {
            println!("Last error: {}", err);
        }
        println!("Mirrored: {} file(s)", report.mirrored_files);
        println!(
            "Health:   {}",
            if report.healthy {
                "ok".to_string()
            } else {
                report.status.to_uppercase()
            }
        );
    }

    if !report.healthy {
        std::process::exit(1);
    }
    Ok(())
}

fn cmd_list(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, true)?;
    let client = client::FsClient::new(&config.api_url, config.api_token.as_deref())?;
    let snapshot = client.load_knowledge(&config.bank_id)?;
    let plan = format::plan_mirror(&snapshot);
    if plan.dirs.is_empty() && plan.files.is_empty() {
        println!("No knowledge base in bank \"{}\".", config.bank_id);
        return Ok(());
    }
    for dir in &plan.dirs {
        println!("{}/", dir);
    }
    for page in &plan.files {
        println!("{}", page.rel_path);
    }
    Ok(())
}

fn cmd_unmount(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, false)?;
    daemon::stop_daemon(&config.dir)?;

    let state = state::load_state(&config.dir, &config.bank_id, &config.api_url);
    let mut removed = 0;
    for entry in state.files.values() {
        let path = config.dir.join(&entry.file);
        make_writable(&path);
        if std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    let _ = std::fs::remove_dir_all(config.dir.join(paths::CONTROL_DIR));
    println!(
        "Unmounted {} — removed {} mirrored file(s) and control data.",
        config.dir.display(),
        removed
    );
    Ok(())
}

fn cmd_logs(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, false)?;
    match std::fs::read_to_string(daemon::log_path(&config.dir)) {
        Ok(log) => {
            let lines: Vec<&str> = log.lines().collect();
            let start = lines.len().saturating_sub(40);
            for line in &lines[start..] {
                println!("{}", line);
            }
        }
        Err(_) => println!("No logs for {}.", config.dir.display()),
    }
    Ok(())
}

/// Hidden entrypoint used by the detached daemon process.
fn cmd_run(common: &FsCommonArgs, baseline: &Baseline) -> Result<()> {
    let config = resolve_config(&overrides_from(common), baseline, true)?;
    run_loop(&config, |m| println!("[{}] {}", stamp(), m));
    Ok(())
}

#[cfg(unix)]
fn make_writable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644));
}
#[cfg(not(unix))]
fn make_writable(_path: &Path) {}
