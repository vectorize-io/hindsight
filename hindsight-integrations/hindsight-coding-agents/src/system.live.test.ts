/**
 * LIVE system test — the real integration, end to end, verified with a real LLM.
 *
 * Requires a running Hindsight server (with its LLM configured):
 *   HINDSIGHT_LIVE_E2E=1 HINDSIGHT_API_URL=http://localhost:8899 npm run test:live
 *
 * Flow: build a real git repo with a decision recorded in a commit + a conversation, run the REAL
 * backfill CLI (extraction happens with the server's LLM), then invoke the built hook binaries as
 * real subprocesses and assert the decision's literal values come back in the injected context —
 * i.e. retrieval is verified semantically, not by mocking. Skipped entirely unless opted in.
 */
import { execFileSync, execSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const LIVE = process.env.HINDSIGHT_LIVE_E2E === "1";
const API_URL = process.env.HINDSIGHT_API_URL || "http://localhost:8899";
const DIST = join(__dirname, "..", "dist");

// The planted decision. The bug report never mentions these literals; only memory can supply them.
const DECISION_STATUSES = ["429", "408"];
const SYMPTOM =
  "Our HTTP client keeps retrying requests that will never succeed and we hammer the auth " +
  "endpoint after failures. Which failures are actually safe to retry here?";

let repo: string;
let diagFile: string;
const RUN = `${Date.now()}`; // session ids must be unique per run: the hooks cache per session id in tmp

function runHookBin(bin: string, event: Record<string, unknown>): string {
  return execFileSync("node", [join(DIST, bin)], {
    input: JSON.stringify(event),
    encoding: "utf-8",
    env: { ...process.env, HINDSIGHT_DIAG_FILE: diagFile },
    timeout: 180_000,
  });
}

function diagLines(): { harness: string; event: string }[] {
  try {
    return readFileSync(diagFile, "utf-8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l));
  } catch {
    return [];
  }
}

describe.runIf(LIVE)("live system: backfill -> reflect -> hook injection", () => {
  beforeAll(() => {
    repo = mkdtempSync(join(tmpdir(), "hs-live-"));
    diagFile = join(repo, "diag.log");

    // a real repo whose history carries the decision's rationale
    execSync(
      `git init -q && git -c user.email=t@t -c user.name=t commit -q --allow-empty ` +
        `-m "fix: retry only transient failures" ` +
        `-m "During the incident the client retried every failed call including 401s and locked accounts. ` +
        `Policy decided with platform: retry any 5xx plus EXACTLY 429 and 408 from the 4xx range; every ` +
        `other 4xx is permanent and must fail fast."`,
      { cwd: repo }
    );
    // project-local config: this repo's own bank on the live server (unique per test run)
    mkdirSync(join(repo, ".hindsight"), { recursive: true });
    writeFileSync(
      join(repo, ".hindsight", "coding-agent.json"),
      JSON.stringify({ apiUrl: API_URL, bankIdTemplate: `live-e2e-{gitProject}` })
    );
    // a past conversation restating the decision (the normalized interchange format)
    const conv = [
      {
        id: "retry-policy",
        turns: [
          { role: "user", text: "The client retried a 401 storm last night and locked accounts. What retry policy do we want?" },
          { role: "assistant", text: "Decided: retry any 5xx, plus exactly 429 and 408 from the 4xx range; every other 4xx fails fast." },
        ],
      },
    ];
    const convFile = join(repo, "conv.json");
    writeFileSync(convFile, JSON.stringify(conv));

    // the REAL backfill CLI: ingests git + chats, drains extraction (server-side real LLM)
    execFileSync(
      "node",
      [join(DIST, "backfill.js"), "--repo", repo, "--bank", `live-e2e-${basename(repo)}`,
       "--api-url", API_URL, "--reset", "--conversations", convFile, "--no-pages"],
      { encoding: "utf-8", timeout: 600_000 }
    );
  }, 660_000);

  afterAll(() => {
    if (repo) rmSync(repo, { recursive: true, force: true });
  });

  it("claude hook injects the decided literals for a symptom prompt", () => {
    const out = runHookBin("claude-hook.js", {
      session_id: `live-claude-${RUN}`,
      cwd: repo,
      prompt: SYMPTOM,
    });
    const parsed = JSON.parse(out);
    const ctx: string = parsed.hookSpecificOutput.additionalContext;
    expect(parsed.hookSpecificOutput.hookEventName).toBe("UserPromptSubmit");
    for (const literal of DECISION_STATUSES) expect(ctx).toContain(literal);
    expect(diagLines().at(-1)).toMatchObject({ harness: "claude-code", event: "reflect_ok" });
  }, 200_000);

  it("second prompt in the same session is served from cache (no new reflect)", () => {
    const before = diagLines().length;
    const out = runHookBin("claude-hook.js", {
      session_id: `live-claude-${RUN}`,
      cwd: repo,
      prompt: "another question, same session",
    });
    expect(JSON.parse(out).hookSpecificOutput.additionalContext).toContain("429");
    expect(diagLines().length).toBe(before); // no new reflect recorded
  }, 30_000);

  it("codex hook shares the protocol shape and the bank", () => {
    const out = runHookBin("codex-hook.js", {
      session_id: `live-codex-${RUN}`,
      cwd: repo,
      user_prompt: SYMPTOM, // codex may send user_prompt instead of prompt
    });
    const parsed = JSON.parse(out);
    const ctx: string = parsed.hookSpecificOutput.additionalContext;
    for (const literal of DECISION_STATUSES) expect(ctx).toContain(literal);
    expect(diagLines().at(-1)).toMatchObject({ harness: "codex", event: "reflect_ok" });
  }, 200_000);

  it("cursor hook speaks its own protocol against the same bank", () => {
    const out = runHookBin("cursor-hook.js", {
      conversation_id: `live-cursor-${RUN}`,
      workspace_roots: [repo],
      prompt: SYMPTOM,
    });
    const parsed = JSON.parse(out);
    expect(parsed.continue).toBe(true);
    for (const literal of DECISION_STATUSES) expect(parsed.additional_context).toContain(literal);
    expect(diagLines().at(-1)).toMatchObject({ harness: "cursor-cli", event: "reflect_ok" });
  }, 200_000);
});

describe.runIf(!LIVE)("live system tests", () => {
  it.skip("skipped — set HINDSIGHT_LIVE_E2E=1 with a running Hindsight server to run", () => {});
});
