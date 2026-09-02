/**
 * Local daemon mode — memory with no server and no Cloud account.
 *
 * `serverMode: "daemon"` runs `hindsight-embed` on this machine and points the plugin at it.
 * Lifecycle is delegated to `@vectorize-io/hindsight-all`, which owns the uvx invocation, profile
 * creation, readiness polling and the macOS Metal/MPS workaround. This module only decides WHEN to
 * start it, and with what environment.
 *
 * Three connection modes, resolved in this order (the same priority the old per-agent Claude Code
 * plugin used in `scripts/lib/daemon.py`):
 *   1. an external API — Cloud or self-hosted; `serverMode` is not "daemon" and nothing here runs
 *   2. a healthy server already on the port — adopted as-is, never restarted, so a daemon started
 *      by an earlier session (or a server the user runs themselves) is reused
 *   3. start the daemon
 *
 * TIMING IS THE CONSTRAINT. A cold start pays for a uvx download plus model load and can take tens
 * of seconds, while the prompt hook has to return in time to not stall the user's turn. Only two
 * points therefore ensure a daemon: SessionStart, before the user has typed anything, and the Stop
 * hook, which has the longest budget and nothing waiting on its result. The prompt path does
 * nothing here at all.
 *
 * EVERY HARNESS NEEDS BOTH POINTS, not just the fresh-process hook ones. The persistent-plugin
 * hosts (opencode, Kilo, Cline, Prime Agent, dsh) run no hook binaries, so they reach these through
 * `RuntimeCore`: `seedIfCold` is their SessionStart and the write-back is their Stop. Missing that
 * is what #3524 reported — dsh in daemon mode never started a daemon, so every `hindsight_*` call
 * failed with ECONNREFUSED until the user ran `daemon-start.js` by hand. `daemon.test.ts` now fails
 * if a harness entrypoint builds a client without reaching one of the two.
 *
 * A daemon that is down is NOT special-cased anywhere downstream. Once the URL is resolved, every
 * mode uses the identical client and the identical error handling: an unreachable local daemon
 * surfaces as the same connection failure (and the same `retain_failed` diagnostic) as an
 * unreachable Cloud or self-hosted server. Skipping work because a local port was closed would have
 * made daemon mode behave differently from the other two for no benefit.
 *
 * There is no stop. One daemon serves every agent and repo on the machine, so ending one session
 * must not cut memory out from under another. Nothing retires it either: the server's idle timeout
 * was removed after it killed in-flight requests (#3903), so `daemonIdleTimeout` is inert.
 */
import { execFileSync, spawn as realSpawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Config } from "./config";
import { diag } from "./diag";
import { log } from "./log";

/**
 * How long each hook is willing to block waiting for a cold daemon. Both sit well inside the
 * corresponding hook timeouts declared in harness/hook-lifecycle.ts (SessionStart 30s, Stop 60s),
 * because overrunning those gets the hook killed by the host.
 */
export const DAEMON_WAIT_SESSION_START_MS = 12_000;
export const DAEMON_WAIT_RETAIN_MS = 40_000;

/** Probe a base URL's /health. Never throws — an unreachable server is just "not running". */
export async function isServerHealthy(baseUrl: string, timeoutMs = 2_000): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * macOS needs a CURRENT Rust toolchain to install the daemon.
 *
 * litellm — pulled in transitively by the API — publishes wheels only for manylinux and Windows
 * (verified against PyPI for 1.95.0: cp310-cp314, no macOS build for any of them). Every macOS
 * install therefore compiles it from the sdist through maturin, and its dependency tree pins a
 * recent rustc (1.94.1 at the time of writing; observed failing on 1.93.1). Only presence is
 * checked here — the required version comes from litellm's transitive crates and moves on its own
 * schedule, so pinning a number here would just go stale. A too-old toolchain still fails, but the
 * error carries the exact version needed.
 */
export function hasRustToolchain(): boolean {
  if (process.platform !== "darwin") return true; // wheels cover linux + windows
  try {
    execFileSync("cargo", ["--version"], {
      stdio: "pipe",
      timeout: 10_000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

/** Is `uv`/`uvx` on PATH? The daemon is fetched and run through it. */
export function hasUvx(): boolean {
  try {
    execFileSync("uvx", ["--version"], {
      stdio: "pipe",
      timeout: 10_000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

interface ProviderProbe {
  provider: string;
  keyEnv: string;
}

/**
 * Providers the daemon can use for local fact extraction, most-specific first. Mirrors the old
 * plugin's list. `claude-code` needs no key — it drives the Claude Code CLI the user already has,
 * which is why it is the last resort rather than an error.
 */
const PROVIDER_PROBES: ProviderProbe[] = [
  { provider: "openai", keyEnv: "OPENAI_API_KEY" },
  { provider: "anthropic", keyEnv: "ANTHROPIC_API_KEY" },
  { provider: "gemini", keyEnv: "GEMINI_API_KEY" },
  { provider: "groq", keyEnv: "GROQ_API_KEY" },
];

export interface LlmChoice {
  provider: string;
  /** Absent for providers that need no credential. */
  apiKey?: string;
  /** Where it came from, for the installer's output and diagnostics. */
  source: string;
}

/**
 * Decide which LLM the daemon extracts facts with.
 *
 * An explicit `HINDSIGHT_API_LLM_PROVIDER` always wins — including for providers this code knows
 * nothing about, so a local Ollama or a gateway can be pointed at without a code change here.
 */
export function detectLlm(env: NodeJS.ProcessEnv = process.env): LlmChoice | undefined {
  const explicit = env.HINDSIGHT_API_LLM_PROVIDER?.trim();
  if (explicit) {
    return {
      provider: explicit,
      apiKey: env.HINDSIGHT_API_LLM_API_KEY?.trim() || undefined,
      source: "HINDSIGHT_API_LLM_PROVIDER",
    };
  }
  for (const probe of PROVIDER_PROBES) {
    const key = env[probe.keyEnv]?.trim();
    if (key) return { provider: probe.provider, apiKey: key, source: probe.keyEnv };
  }
  if (onPath("claude")) {
    return { provider: "claude-code", source: "the Claude Code CLI on PATH (no API key needed)" };
  }
  return undefined;
}

function onPath(bin: string): boolean {
  try {
    execFileSync("which", [bin], {
      stdio: "pipe",
      timeout: 5_000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

/** Environment handed to the daemon: the LLM choice, the idle timeout, and any HINDSIGHT_API_* the
 *  user already exports (so provider-specific knobs pass through without being enumerated here). */
export function daemonEnv(
  cfg: Config,
  env: NodeJS.ProcessEnv = process.env
): Record<string, string | undefined> {
  const out: Record<string, string | undefined> = {};
  // Forwarded only when set, and inert either way: the server dropped its idle timeout in #3903
  // (it measured idleness from the request *start*, so a long retain killed its own daemon). Kept
  // so an existing config still starts — the daemon ignores the value.
  // It used to default to 300s here, which made the plugin the one thing on the machine opting a
  // SHARED daemon into an auto-exit; every other integration shipped 0.
  if (cfg.daemonIdleTimeout !== undefined) {
    out.HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT = String(cfg.daemonIdleTimeout);
  }
  for (const [key, value] of Object.entries(env)) {
    if (key.startsWith("HINDSIGHT_API_") && value) out[key] = value;
  }
  const llm = detectLlm(env);
  if (llm) {
    out.HINDSIGHT_API_LLM_PROVIDER = llm.provider;
    if (llm.apiKey) out.HINDSIGHT_API_LLM_API_KEY = llm.apiKey;
  }
  return out;
}

/**
 * Launch the starter as a DETACHED process, the way seeding and the codebase survey are launched.
 *
 * A cold start pays for a uvx download plus model load and routinely outlives any hook's timeout,
 * so it cannot be awaited inline: the harness would kill the hook mid-start and the daemon would
 * never come up. The child outlives this process and keeps going. Idempotent — the starter itself
 * re-checks health, so several sessions racing to start one daemon is harmless.
 */
export function startDaemonDetached(
  cfg: Config,
  harness: string,
  spawnFn: typeof realSpawn = realSpawn
): void {
  try {
    const starter = join(dirname(fileURLToPath(import.meta.url)), "daemon-start.js");
    const child = spawnFn("node", [starter, "--harness", harness], {
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    });
    // spawn() failures often surface ASYNCHRONOUSLY as an 'error' event; unhandled, that would
    // crash the hook.
    child.on("error", () => {});
    child.unref();
    diag(harness, "daemon_start_spawned", { apiUrl: cfg.apiUrl });
  } catch {
    /* best-effort: a failed spawn must not break the caller */
  }
}

/**
 * Make sure a daemon is serving `cfg.apiUrl`. A no-op in every other mode.
 *
 * Side effect only — callers proceed regardless of whether it came up, so that a request against a
 * down daemon fails through exactly the same path as a request against a down Cloud or self-hosted
 * server. Never throws: memory is best-effort and must not break the agent.
 *
 * `waitMs` bounds how long the caller is willing to block for a cold start — always well under the
 * calling hook's own timeout, so a slow start costs memory for one turn rather than a killed hook.
 */
export async function ensureDaemon(
  cfg: Config,
  harness: string,
  opts: { waitMs?: number; spawnFn?: typeof realSpawn } = {}
): Promise<void> {
  if (cfg.serverMode !== "daemon") return;
  if (await isServerHealthy(cfg.apiUrl)) return;
  if (!preflightDaemon(cfg, harness)) return;
  startDaemonDetached(cfg, harness, opts.spawnFn);
  await waitForHealth(cfg.apiUrl, opts.waitMs ?? 0);
}

/** Poll /health until it answers or the budget runs out. `budgetMs <= 0` means "don't wait". */
export async function waitForHealth(
  baseUrl: string,
  budgetMs: number,
  pollMs = 1_000
): Promise<boolean> {
  const deadline = Date.now() + budgetMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, pollMs));
    if (await isServerHealthy(baseUrl)) return true;
  }
  return false;
}

/** The two things daemon mode cannot work without. Reported, never thrown. */
export function preflightDaemon(cfg: Config, harness: string): boolean {
  if (!hasUvx()) {
    diag(harness, "daemon_uvx_missing", { apiUrl: cfg.apiUrl });
    log.warn(harness, "daemon mode needs `uv` on PATH — see https://docs.astral.sh/uv/");
    return false;
  }
  if (!hasRustToolchain()) {
    diag(harness, "daemon_rust_missing", { platform: process.platform });
    log.warn(
      harness,
      "daemon mode on macOS needs a current Rust toolchain (litellm ships no macOS wheel) — " +
        "install from https://rustup.rs, then `rustup default stable && rustup update`"
    );
    return false;
  }
  if (!detectLlm()) {
    diag(harness, "daemon_no_llm", {});
    log.warn(
      harness,
      "daemon mode needs an LLM for fact extraction — set OPENAI_API_KEY (or ANTHROPIC_API_KEY / " +
        "GEMINI_API_KEY / HINDSIGHT_API_LLM_PROVIDER), or install the Claude Code CLI"
    );
    return false;
  }
  return true;
}
