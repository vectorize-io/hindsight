/**
 * Shared runtime for HOOK-based harnesses (Claude Code, Cursor CLI, ...).
 *
 * Persistent-plugin harnesses (opencode) get a long-lived RuntimeCore; hook harnesses invoke a
 * fresh process per prompt, so the reflect-once-per-session behavior lives here instead:
 * reflect on the session's first prompt, cache the answer in tmp, re-emit it on later prompts.
 * A failed reflect never breaks the agent — and is always recorded in the diagnostic file, so a
 * silently memory-less session can't masquerade as a memory session.
 *
 * A harness plugs in with a HookSpec: its name, how to read (prompt, cwd, sessionId) from its
 * stdin event, and how to wrap injected context in its native output schema.
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deriveBankId } from "./bank";
import { loadConfig } from "./config";
import { HindsightClient } from "./hindsight";
import { buildSystemInjection } from "./inject";

export interface HookEventFields {
  prompt?: string;
  cwd?: string;
  sessionId?: string;
}

export interface HookSpec {
  /** Harness name — config `harnesses.<name>` section, {harness} template field, diag records. */
  harness: string;
  /** Read the fields out of the harness's stdin event (shapes differ per harness). */
  parse(event: Record<string, unknown>): HookEventFields;
  /** Wrap injected context in the harness's native hook-output schema. */
  emit(context: string): unknown;
}

function diag(harness: string, event: string, extra: Record<string, unknown> = {}): void {
  try {
    appendFileSync(
      process.env.HINDSIGHT_DIAG_FILE || "/tmp/hindsight-plugin.log",
      JSON.stringify({ ts: new Date().toISOString(), harness, event, ...extra }) + "\n"
    );
  } catch {
    /* diagnostics must not break the agent */
  }
}

/** Run one hook invocation: stdin event in, (maybe) an injection object on stdout. */
export async function runHook(spec: HookSpec): Promise<void> {
  let ev: Record<string, unknown> = {};
  try {
    ev = JSON.parse(readFileSync(0, "utf8")) as Record<string, unknown>;
  } catch {
    return; // no/invalid event: stay silent
  }
  const { prompt: rawPrompt, cwd, sessionId } = spec.parse(ev);
  const prompt = (rawPrompt || "").trim();
  if (!prompt) return;

  const cfg = loadConfig({ harness: spec.harness, projectDir: cwd });
  if (cfg.disabled) return;

  const out = (context: string) => process.stdout.write(JSON.stringify(spec.emit(context)));

  // once-per-session reflect, cached across hook invocations (each invocation is a fresh process)
  const cacheDir = join(tmpdir(), `hindsight-${spec.harness}`);
  const cacheFile = join(cacheDir, `${sessionId || "no-session"}.json`);
  try {
    const cached = JSON.parse(readFileSync(cacheFile, "utf8")) as { answer?: string };
    if (typeof cached.answer === "string") {
      if (cached.answer) out(buildSystemInjection(cached.answer));
      return; // reflect already ran for this session ("" = it ran and found nothing)
    }
  } catch {
    /* no cache yet — first prompt of the session */
  }

  const client = new HindsightClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: deriveBankId(cfg, cwd || process.cwd(), spec.harness),
  });
  const t0 = Date.now();
  let answer = "";
  try {
    answer = await client.reflect(prompt, { budget: "high", timeoutMs: cfg.reflectTimeoutMs });
    diag(spec.harness, answer ? "reflect_ok" : "reflect_empty", {
      ms: Date.now() - t0,
      chars: answer.length,
      query: prompt.slice(0, 80),
    });
  } catch (e) {
    diag(spec.harness, "reflect_failed", {
      ms: Date.now() - t0,
      error: String((e as Error)?.message || e).slice(0, 200),
      query: prompt.slice(0, 80),
    });
  }
  try {
    mkdirSync(cacheDir, { recursive: true });
    writeFileSync(cacheFile, JSON.stringify({ answer }));
  } catch {
    /* cache is best-effort; worst case we reflect again next prompt */
  }
  if (answer) out(buildSystemInjection(answer));
}
