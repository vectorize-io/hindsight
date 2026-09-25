/**
 * Grok Build runs the hooks in ~/.claude/settings.json as well as its own, so on a machine wired
 * for both hosts every Grok event reaches Claude Code's entry points too. These drive the real
 * lifecycle declarations with Grok's envelope and assert which ones do any work.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runHook } from "./core/hook";
import { runRetainHook } from "./core/retain-hook";
import { runSessionStartHook } from "./core/session-start";
import type { RawConfig } from "./core/config";
import { HOOK_HARNESSES } from "./harness/hook-lifecycle";

let stdin = "";
vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    readFileSync: (target: unknown, ...rest: unknown[]) =>
      target === 0 ? stdin : (actual.readFileSync as (...a: unknown[]) => unknown)(target, ...rest),
  };
});

let rawConfig: RawConfig = {};
vi.mock("./core/config", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./core/config")>();
  return { ...actual, loadConfig: () => actual.resolveConfig(rawConfig) };
});

let root: string;
let sessionId: string;

/** Grok 1.0.41's envelope: its camelCase fields plus Claude-named aliases. */
const grokEvent = (hookEventName: string, hook_event_name: string, extra = {}) =>
  JSON.stringify({
    hookEventName,
    hook_event_name,
    sessionId,
    session_id: sessionId,
    cwd: root,
    workspaceRoot: `${root}/`,
    permissionMode: "auto",
    ...extra,
  });

const promptClient = () =>
  vi.fn(() => ({
    reflect: async () => "",
    listPages: async () => ({ items: [] }),
    searchKnowledgePages: async () => [],
    recallObservations: async () => [],
    knowledgePagesSupported: false,
  }));

const quietly = async (run: () => Promise<void>) => {
  const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
  try {
    await run();
  } finally {
    write.mockRestore();
  }
};

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-grok-compat-"));
  execFileSync("git", ["init", "-q", root]);
  sessionId = `sess-grok-${basename(root)}`;
  rawConfig = { autoSeed: false, autoUpdate: false };
  vi.stubEnv("HINDSIGHT_DIAG_FILE", join(root, "diag.log"));
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

describe("Claude Code hooks run by Grok", () => {
  it("the prompt hook recalls under Claude Code", async () => {
    const makeClient = promptClient();
    stdin = grokEvent("user_prompt_submit", "UserPromptSubmit", { prompt: "which parser?" });
    await quietly(() => runHook(HOOK_HARNESSES["claude-code"].prompt, makeClient as never));
    expect(makeClient).toHaveBeenCalled();
  });

  it("the prompt hook stays silent when Grok's hook runner started it", async () => {
    vi.stubEnv("GROK_HOOK_EVENT", "user_prompt_submit");
    const makeClient = promptClient();
    stdin = grokEvent("user_prompt_submit", "UserPromptSubmit", { prompt: "which parser?" });
    await quietly(() => runHook(HOOK_HARNESSES["claude-code"].prompt, makeClient as never));
    expect(makeClient).not.toHaveBeenCalled();
  });

  it("the session-start hook stays silent when Grok's hook runner started it", async () => {
    vi.stubEnv("GROK_HOOK_EVENT", "session_start");
    const makeClient = vi.fn();
    stdin = grokEvent("session_start", "SessionStart", { source: "new" });
    await quietly(() =>
      runSessionStartHook(HOOK_HARNESSES["claude-code"].sessionStart, makeClient)
    );
    expect(makeClient).not.toHaveBeenCalled();
  });

  it("the stop hook stays silent when Grok's hook runner started it", async () => {
    vi.stubEnv("GROK_HOOK_EVENT", "stop");
    const makeClient = vi.fn();
    stdin = grokEvent("stop", "Stop", { reason: "end_turn", transcript_path: join(root, "t") });
    await runRetainHook(HOOK_HARNESSES["claude-code"].retain, makeClient);
    expect(makeClient).not.toHaveBeenCalled();
  });

  it("leave Grok's own prompt hook running", async () => {
    vi.stubEnv("GROK_HOOK_EVENT", "user_prompt_submit");
    const makeClient = promptClient();
    stdin = grokEvent("user_prompt_submit", "UserPromptSubmit", { prompt: "which parser?" });
    await quietly(() => runHook(HOOK_HARNESSES["grok-build"].prompt, makeClient as never));
    expect(makeClient).toHaveBeenCalled();
  });
});
