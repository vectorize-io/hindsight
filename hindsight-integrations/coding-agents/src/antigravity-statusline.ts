#!/usr/bin/env node
/**
 * Antigravity CLI custom status-line command. Unlike hook output, its stdout is rendered directly
 * by the TUI, so this is the host-native, non-conversational way to show that Hindsight is active.
 * It deliberately does not call the Hindsight API: Antigravity invokes this command on UI state
 * changes, and a network request here would make rendering depend on server latency.
 */
import { applyBankConfig, loadConfig, type Config } from "./core/config";
import { deriveBankIdOrSkip } from "./core/bank";
import { brandWord } from "./core/brand";

export interface AntigravityStatusLineState {
  cwd?: string;
  workspace?: { current_dir?: string };
}

/** Build the compact, user-visible Hindsight indicator from Antigravity's documented state payload. */
export function buildAntigravityStatusLine(state: AntigravityStatusLineState, cfg: Config): string {
  const cwd = state.cwd || state.workspace?.current_dir;
  if (!cwd) return brandWord();

  // No session root here: the documented state payload carries no conversation id, so there is no
  // key to look one up by (see core/session-cache.ts). Outside a git repo this can therefore name
  // the directory the agent navigated to while the hooks keep writing to the session's own bank
  // (#3563). It is a display-only divergence — this command never calls the API, so nothing is
  // retained or created under the name shown.
  const bankId = deriveBankIdOrSkip(cfg, cwd, "antigravity-cli");
  if (bankId === null) return brandWord(); // unidentifiable repo: name no bank rather than the wrong one
  const resolved = applyBankConfig(cfg, bankId, cwd);
  return resolved.cfg.disabled ? "" : `${brandWord()} · ${resolved.bankId}`;
}

/* c8 ignore start */
async function main(): Promise<void> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  try {
    const state = JSON.parse(Buffer.concat(chunks).toString("utf8")) as AntigravityStatusLineState;
    const cfg = loadConfig({ harness: "antigravity-cli" });
    process.stdout.write(buildAntigravityStatusLine(state, cfg));
  } catch {
    // A malformed TUI payload must leave the status line blank, never surface an error in the UI.
  }
}

if (process.argv[1]?.endsWith("antigravity-statusline.js")) void main();
/* c8 ignore stop */
