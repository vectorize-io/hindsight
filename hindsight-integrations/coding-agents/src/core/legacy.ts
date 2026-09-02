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
import { join } from "node:path";
import { DEFAULT_DAEMON_PORT } from "./config";

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
