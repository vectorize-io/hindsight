/**
 * hindsight-fs — mirror a Hindsight bank's knowledge base as a live local folder.
 *
 * Programmatic entry point. The CLI (`hindsight-fs`) is in cli.ts.
 */

export { runSync, type SyncResult } from "./sync.js";
export { runLoop, type RunLoopOptions, type LoopLogger } from "./loop.js";
export { resolveConfig, saveConfig, type MountConfig, type ConfigOverrides } from "./config.js";
export {
  HindsightFsClient,
  ApiError,
  type KnowledgeNode,
  type KnowledgeSnapshot,
} from "./client.js";
export { startDaemon, stopDaemon, daemonStatus, type DaemonStatus } from "./daemon.js";
export {
  computeHealth,
  type HealthReport,
  type HealthStatus,
  type HealthOptions,
} from "./health.js";
export { planMirror, renderIndex, slug, type MirrorPlan, type PageFile } from "./format.js";
export { stringifyFrontmatter, parseDocument, type Frontmatter } from "./frontmatter.js";
