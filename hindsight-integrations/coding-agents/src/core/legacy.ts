/**
 * Carry the SERVER ENDPOINT over from the older per-agent Claude Code plugin.
 *
 * Only the endpoint — where memory is stored, and the credential to reach it. None of that
 * plugin's ~40 behavioural settings are translated: most (12 `recall*`, 7 `retain*`, the mission
 * pair) describe a pipeline this package replaced outright, and quietly reinterpreting the rest
 * would be worse than leaving them behind.
 *
 * Why it matters: someone running the old plugin against a self-hosted server or a local daemon
 * has already decided where their prompts and transcripts go. Installing this package and
 * defaulting to Cloud would silently start sending them somewhere else. Adopting the endpoint they
 * already chose is the conservative move, not the convenient one.
 *
 * Conversations are NOT carried here — they are re-imported from local transcripts
 * (`--import-conversations`, see core/history.ts). That path re-extracts them as new documents,
 * which is what makes it independent of the old bank: the old plugin's default was a single static
 * bank (`claude_code`, since `dynamicBankId` defaults to false) whose documents record only
 * `retained_at`, `message_count` and `session_id` — nothing identifying the repo. Splitting that
 * bank per repo would require the local transcripts anyway, so they are the only source used.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, relative, isAbsolute } from "node:path";
import { DEFAULT_DAEMON_PORT } from "./config";

/** The old per-agent Claude Code plugin's name in Claude's plugin registry (`<name>@<marketplace>`). */
const LEGACY_CLAUDE_PLUGIN = "hindsight-memory";

/**
 * The registry key (`hindsight-memory@<marketplace>`) of the old Claude Code plugin when it is still
 * installed AND active for `cwd`, else undefined.
 *
 * Running both is not harmless: each recalls into every prompt and retains every transcript, so the
 * agent sees two memory blocks and conversations land twice — the old plugin's copy in its single
 * static `claude_code` bank. The installer does not uninstall it (Claude owns its plugin registry),
 * so the session start is where the user finds out.
 *
 * Read from Claude's own files: `plugins/installed_plugins.json` for what is installed (user scope,
 * or a project/local scope whose `projectPath` contains `cwd`) and `settings.json` `enabledPlugins`
 * for an explicit disable. Any read failure means "not detected" — this only ever adds a warning.
 */
export function detectLegacyClaudePlugin(
  cwd: string,
  claudeDir: string = join(homedir(), ".claude")
): string | undefined {
  const registry = readJson(join(claudeDir, "plugins", "installed_plugins.json"));
  const plugins = registry?.plugins;
  if (!plugins || typeof plugins !== "object") return undefined;
  const enabled = readJson(join(claudeDir, "settings.json"))?.enabledPlugins as
    | Record<string, unknown>
    | undefined;
  for (const [key, raw] of Object.entries(plugins as Record<string, unknown>)) {
    if (key.split("@")[0] !== LEGACY_CLAUDE_PLUGIN) continue;
    if (enabled && enabled[key] === false) continue;
    // v2 registry: an array of installs; v1: a single install object.
    const installs = (Array.isArray(raw) ? raw : [raw]) as Array<Record<string, unknown> | null>;
    const active = installs.some((i) => {
      if (!i || typeof i !== "object") return false;
      if (typeof i.projectPath !== "string") return true; // user scope: every project
      const rel = relative(i.projectPath, cwd);
      return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
    });
    if (active) return key;
  }
  return undefined;
}

function readJson(path: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

/** The user-visible SessionStart warning for a still-active old plugin. */
export function legacyClaudePluginWarning(key: string): string {
  return (
    `⚠️ The old Hindsight Claude Code plugin (${key}) is still installed — it runs alongside this ` +
    `one, so memory is recalled and retained twice.\n` +
    `  ↳ remove it: claude plugin uninstall ${key}`
  );
}

/**
 * The old per-agent plugins that shipped a user config, and its filename.
 *
 * Only these two ever adopted the `~/.hindsight/<agent>.json` convention — the other superseded
 * integrations (Cursor CLI, Copilot CLI, opencode, Cline) have no such file, so there is no
 * endpoint of theirs to carry. Both use IDENTICAL key names, which is why one reader serves both.
 */
export const LEGACY_CONFIG_FILES: Record<string, string> = {
  "claude-code": "claude-code.json",
  codex: "codex.json",
};

/** Where an old plugin told users to put their config. */
export function legacyConfigPath(harness: string, home: string = homedir()): string {
  return join(home, ".hindsight", LEGACY_CONFIG_FILES[harness] ?? `${harness}.json`);
}

export interface LegacyEndpoint {
  /** Which old plugin it came from, for the installer's output. */
  harness: string;
  serverMode: "cloud" | "self-hosted" | "daemon";
  apiUrl?: string;
  apiToken?: string;
  /** Only when it differs from the default — in daemon mode the port IS the endpoint. */
  apiPort?: number;
  /** The file it came from, so the installer can say where it looked. */
  source: string;
}

const CLOUD_URL = "https://api.hindsight.vectorize.io";

/**
 * Read the old plugin's endpoint, or undefined when it was never configured.
 *
 * An ABSENT `hindsightApiUrl` is meaningful rather than missing: the old plugin treated an empty
 * URL as "use the local daemon on `apiPort`" (`daemon.py:get_api_url`), so a config file that
 * never set one describes daemon mode, not an unconfigured user.
 */
export function readLegacyEndpoint(
  home: string = homedir(),
  prefer: readonly string[] = []
): LegacyEndpoint | undefined {
  // Look at the agents being installed FIRST: someone wiring Codex should get Codex's endpoint even
  // if a stale claude-code.json is also lying around. Falls back to any known legacy config, since
  // the common case is one server shared by both.
  const order = [...prefer, ...Object.keys(LEGACY_CONFIG_FILES)].filter(
    (h, i, all) => h in LEGACY_CONFIG_FILES && all.indexOf(h) === i
  );
  for (const harness of order) {
    const found = readOne(harness, home);
    if (found) return found;
  }
  return undefined;
}

function readOne(harness: string, home: string): LegacyEndpoint | undefined {
  const source = legacyConfigPath(harness, home);
  if (!existsSync(source)) return undefined;
  let raw: Record<string, unknown>;
  try {
    raw = JSON.parse(readFileSync(source, "utf8")) as Record<string, unknown>;
  } catch {
    return undefined; // a config we cannot read is not a decision we can honour
  }
  if (!raw || typeof raw !== "object") return undefined;

  const apiToken = typeof raw.hindsightApiToken === "string" ? raw.hindsightApiToken : undefined;
  const url = typeof raw.hindsightApiUrl === "string" ? raw.hindsightApiUrl.trim() : "";

  if (url) {
    return {
      harness,
      serverMode: url.replace(/\/+$/, "") === CLOUD_URL ? "cloud" : "self-hosted",
      apiUrl: url,
      ...(apiToken ? { apiToken } : {}),
      source,
    };
  }
  const port = typeof raw.apiPort === "number" ? raw.apiPort : undefined;
  return {
    harness,
    serverMode: "daemon",
    ...(apiToken ? { apiToken } : {}),
    ...(port && port !== DEFAULT_DAEMON_PORT ? { apiPort: port } : {}),
    source,
  };
}
