/**
 * Shared entrypoint factory for the PERSISTENT-PLUGIN harnesses (opencode and its forks).
 *
 * Unlike claude-code/cursor/codex — a fresh hook process per event — these harnesses load a plugin
 * once and keep it alive, so the whole v2 surface (recall+inject, cold seed, native tools,
 * write-back) is delivered through the host's plugin hooks. Kilo CLI is a fork of opencode: same
 * config schema, same `plugin` array, and @kilocode/plugin re-exports opencode's Plugin/Hooks
 * types, so both hosts drive the identical runtime and differ only in which harness name they
 * report. That name is NOT config-chosen — the host that loaded us determines it — and it selects
 * the `harnesses.<name>` config section and feeds `{harness}` bank templating.
 */
import type { Plugin } from "@opencode-ai/plugin";
import { posix, win32 } from "node:path";
import { deriveBankId } from "../core/bank";
import { applyBankConfig, loadConfig } from "../core/config";
import { log } from "../core/log";
import { HindsightClient } from "../core/hindsight";
import { RuntimeCore } from "../core/runtime";
import { readOpencodeMessages, type OcMessage } from "../core/transcript-opencode";
import { opencodeAdapter } from "./opencode";

/** Hold toasts until this long after init — see the notifier comment below. */
const TOAST_MOUNT_GRACE_MS = 3000;

function isFilesystemRoot(path: string): boolean {
  return posix.parse(path).root === path || win32.parse(path).root === path;
}

export function resolveProjectDirectory(input?: {
  worktree?: string;
  directory?: string;
}): string | undefined {
  const worktree = input?.worktree;
  return worktree && !isFilesystemRoot(worktree) ? worktree : input?.directory;
}

/**
 * Build the default export for a persistent-plugin host. `harness` is the name the host is known
 * by ("opencode", "kilo"), used for config lookup, bank derivation and log scoping.
 */
export function createPluginEntry(harness: string): Plugin {
  return async (input) => {
    const projectDir = resolveProjectDirectory(input);
    let cfg = loadConfig({ harness });
    if (cfg.disabled) return {}; // inert: same agent, no memory (baseline parity)

    const resolved = applyBankConfig(
      cfg,
      deriveBankId(cfg, projectDir || process.cwd(), harness),
      projectDir || process.cwd()
    );
    cfg = resolved.cfg;
    const bankId = resolved.bankId;
    if (cfg.disabled) return {}; // per-bank opt-out (banks.<id> override)
    const client = new HindsightClient({
      apiUrl: cfg.apiUrl,
      apiToken: cfg.apiToken,
      bank: bankId,
      maxParallelRetains: cfg.maxParallelRetains,
      observationScopes: cfg.observationScopes,
    });
    const core = new RuntimeCore(client, bankId, cfg, harness, projectDir || process.cwd());
    // Visible presence via the host's own notice API (POST /tui/show-toast) — never stderr, which
    // these TUIs render at the cursor, wedging text against the input bar.
    const oc = (input as { client?: { tui?: { showToast?: (o: unknown) => Promise<unknown> } } })
      ?.client;
    // The host injects the v1 SDK client: showToast takes {body: {...}} and RESOLVES with
    // {data|error} (openapi-fetch style) instead of rejecting — inspect the result, not .catch.
    log.debug(harness, "toast channel", { wired: Boolean(oc?.tui?.showToast) });
    if (oc?.tui?.showToast) {
      // The toast event is not durable: one published before the TUI mounts and subscribes to the
      // event stream is silently lost. A warm bank makes the seed banner fire <1s after plugin
      // init, well before mount — so hold any toast until ~3s past init (later ones go immediately).
      const initAt = Date.now();
      core.setNotifier((title, message) => {
        const wait = Math.max(0, TOAST_MOUNT_GRACE_MS - (Date.now() - initAt));
        setTimeout(() => {
          void oc.tui!.showToast!({ body: { title, message, variant: "info", duration: 6000 } })
            .then((r: unknown) => {
              const err = (r as { error?: unknown })?.error;
              if (err)
                log.debug(harness, "toast rejected", { error: JSON.stringify(err).slice(0, 200) });
            })
            .catch((e: unknown) =>
              log.debug(harness, "toast failed", { error: String(e).slice(0, 120) })
            );
        }, wait).unref?.();
      });
    }

    // Let the runtime READ the transcript back on session.idle. The messages.transform hook only
    // ever sees the list as it stood BEFORE the reply, so the completed exchange has to be
    // refetched; `GET /session/{id}/message` returns `{info, parts}` items, which is exactly the
    // shape the live-transcript normalizer already consumes.
    const sessionApi = (
      oc as {
        session?: {
          messages?: (o: {
            path: { id: string };
          }) => Promise<{ data?: OcMessage[]; error?: unknown }>;
        };
      }
    )?.session;
    if (sessionApi?.messages) {
      core.setTranscriptSource(async (sessionId) => {
        const r = await sessionApi.messages!({ path: { id: sessionId } });
        if (r?.error) throw new Error(JSON.stringify(r.error).slice(0, 200));
        return readOpencodeMessages(r?.data ?? []);
      });
    }
    log.debug(harness, "transcript source", { wired: Boolean(sessionApi?.messages) });

    const runtime = opencodeAdapter.createRuntime(core) as Awaited<ReturnType<Plugin>>;

    // SessionStart-equivalent: cold-check the bank, kick off the background engine, compute the
    // knowledge preamble. Fire-and-forget — these hosts BLOCK THEIR BOOT on plugin init, so this
    // must never gate startup on a network round-trip (a stalled server froze the whole TUI).
    // onPrompt already tolerates an empty preamble until this resolves.
    void core.seedIfCold(projectDir);

    // Keeping the bank current needs no separate path: seedIfCold fired the deepen engine, and the
    // engine's idempotent git pass (cfg.gitIngest) ingests whatever is new — syncing IS re-seeding.
    return runtime;
  };
}
