#!/usr/bin/env node
/**
 * hindsight-claude-hook — the Claude Code entry point (a `UserPromptSubmit` hook).
 *
 * Claude Code has no persistent plugin runtime, so memory arrives through its hook mechanism:
 * on each user prompt this executable reflects the project's memory over the prompt and returns
 * the synthesized root-cause context via `additionalContext`. Reflect runs ONCE per Claude
 * session (the first prompt); later prompts re-emit the cached answer so the decision stays in
 * context. A failed reflect never breaks the agent — but it is always recorded in the diagnostic
 * file, so a silently memory-less session can't masquerade as a memory session.
 *
 * Install (Claude Code settings.json):
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "hindsight-claude-hook" } ] } ] } }
 *
 * Config: the same layered files as every entry point (~/.hindsight/coding-agent.json plus an
 * optional <project>/.hindsight/coding-agent.json), harness name "claude-code" — so its
 * `harnesses."claude-code"` section can point Claude at a different bank than opencode.
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "./core/config";
import { HindsightClient } from "./core/hindsight";
import { buildSystemInjection } from "./core/inject";

interface HookEvent {
  session_id?: string;
  prompt?: string;
  cwd?: string;
}

function diag(event: string, extra: Record<string, unknown> = {}): void {
  try {
    appendFileSync(process.env.HINDSIGHT_DIAG_FILE || "/tmp/hindsight-plugin.log",
      JSON.stringify({ ts: new Date().toISOString(), harness: "claude-code", event, ...extra }) + "\n");
  } catch {
    /* diagnostics must not break the agent */
  }
}

function emit(context: string): void {
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context },
  }));
}

async function main(): Promise<void> {
  let ev: HookEvent = {};
  try {
    ev = JSON.parse(readFileSync(0, "utf8")) as HookEvent;
  } catch {
    return; // no/invalid event: stay silent
  }
  const prompt = (ev.prompt || "").trim();
  if (!prompt) return;

  const cfg = loadConfig({ harness: "claude-code", projectDir: ev.cwd });
  if (cfg.disabled) return;

  // once-per-session reflect, cached across hook invocations (each invocation is a fresh process)
  const cacheDir = join(tmpdir(), "hindsight-claude");
  const cacheFile = join(cacheDir, `${ev.session_id || "no-session"}.json`);
  try {
    const cached = JSON.parse(readFileSync(cacheFile, "utf8")) as { answer?: string };
    if (typeof cached.answer === "string") {
      if (cached.answer) emit(buildSystemInjection(cached.answer));
      return; // reflect already ran for this session ("" = it ran and found nothing)
    }
  } catch {
    /* no cache yet — first prompt of the session */
  }

  const client = new HindsightClient({ apiUrl: cfg.apiUrl, apiToken: cfg.apiToken, bank: cfg.bankId });
  const t0 = Date.now();
  let answer = "";
  try {
    answer = await client.reflect(prompt, { budget: "high", timeoutMs: cfg.reflectTimeoutMs });
    diag(answer ? "reflect_ok" : "reflect_empty",
         { ms: Date.now() - t0, chars: answer.length, query: prompt.slice(0, 80) });
  } catch (e) {
    diag("reflect_failed", { ms: Date.now() - t0,
      error: String((e as Error)?.message || e).slice(0, 200), query: prompt.slice(0, 80) });
  }
  try {
    mkdirSync(cacheDir, { recursive: true });
    writeFileSync(cacheFile, JSON.stringify({ answer }));
  } catch {
    /* cache is best-effort; worst case we reflect again next prompt */
  }
  if (answer) emit(buildSystemInjection(answer));
}

void main();
