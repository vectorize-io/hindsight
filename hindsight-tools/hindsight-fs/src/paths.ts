/** Shared filesystem layout constants for a mount directory. */

/** Hidden control sub-directory inside every mount (config, state, daemon). */
export const CONTROL_DIR = ".hindsight-fs";

export const STATE_FILE = "state.json";
export const PID_FILE = "daemon.pid";
export const LOG_FILE = "daemon.log";
export const INDEX_FILE = "index.md";
