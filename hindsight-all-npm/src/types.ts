import type { Logger } from './logger.js';

/**
 * Options for {@link HindsightServer}.
 *
 * The server is intentionally thin and pass-through: anything configurable
 * on the daemon side (env vars or CLI flags) can be set here without needing
 * a new dedicated option. Use {@link env} for `HINDSIGHT_*` / `OPENAI_API_KEY` /
 * custom provider settings, and the two `extra*` arrays to append raw CLI
 * args to `profile create` or `daemon start`.
 *
 * For talking to the daemon after `start()`, use `@vectorize-io/hindsight-client`
 * against `server.getBaseUrl()`. This package does not ship its own HTTP
 * client.
 */
export interface HindsightServerOptions {
  /** Profile name used for `--profile <name>` on every sub-command. Default: `"default"`. */
  profile?: string;
  /** TCP port the daemon listens on. Default: `8888`. */
  port?: number;
  /** Hostname the daemon binds to (for health checks). Default: `127.0.0.1`. */
  host?: string;
  /** Version of the underlying `hindsight-embed` PyPI package to run via `uvx`. Default: `"latest"`. */
  embedVersion?: string;
  /** Local path to a `hindsight-embed` checkout — takes precedence over `embedVersion`. */
  embedPackagePath?: string;
  /**
   * Environment variables passed to the daemon process AND written into the
   * profile via repeated `--env KEY=VALUE` flags. This is the preferred way
   * to surface any `HINDSIGHT_API_*` / `HINDSIGHT_EMBED_*` setting — adding a
   * new daemon env var never requires a wrapper update.
   *
   * Values of `undefined` are dropped (so you can spread conditionally).
   */
  env?: Record<string, string | undefined>;
  /** Extra args appended verbatim to `hindsight-embed profile create <name> --merge ...`. */
  extraProfileCreateArgs?: string[];
  /** Extra args appended verbatim to `hindsight-embed daemon --profile <name> start ...`. */
  extraDaemonStartArgs?: string[];
  /**
   * On macOS, automatically set
   * `HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=1` and
   * `HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=1` to avoid Metal/MPS crashes in
   * daemon mode. Default: `true` on `darwin`, ignored elsewhere. Any value set
   * explicitly in {@link env} wins over the auto-applied value.
   */
  platformCpuWorkaround?: boolean;
  /** Max time (ms) to wait for `/health` to return 200. Default: `30_000`. */
  readyTimeoutMs?: number;
  /** Polling interval (ms) while waiting for `/health`. Default: `1_000`. */
  readyPollIntervalMs?: number;
  /** Optional pluggable logger. Default: silent. */
  logger?: Logger;
}
