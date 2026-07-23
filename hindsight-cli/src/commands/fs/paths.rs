//! Shared filesystem layout constants for a mount directory.

/// Hidden control sub-directory inside every mount (config, state, daemon).
pub const CONTROL_DIR: &str = ".hindsight-fs";

pub const STATE_FILE: &str = "state.json";
pub const PID_FILE: &str = "daemon.pid";
pub const LOG_FILE: &str = "daemon.log";
pub const INDEX_FILE: &str = "index.md";
