/**
 * hindsight-fs — mirror a Hindsight bank's mental models as a live local folder.
 *
 * Programmatic entry point. The CLI (`hindsight-fs`) is in cli.ts.
 */

export { runSync, type SyncResult } from "./sync.js";
export { runLoop, type RunLoopOptions, type LoopLogger } from "./loop.js";
export { resolveConfig, saveConfig, type MountConfig, type ConfigOverrides } from "./config.js";
export { HindsightFsClient, ApiError, type MentalModel } from "./client.js";
export { startDaemon, stopDaemon, daemonStatus, type DaemonStatus } from "./daemon.js";
export {
  computeHealth,
  type HealthReport,
  type HealthStatus,
  type HealthOptions,
} from "./health.js";
export { renderMentalModel, renderIndex, fileNameFor } from "./format.js";
export { stringifyFrontmatter, parseDocument, type Frontmatter } from "./frontmatter.js";
