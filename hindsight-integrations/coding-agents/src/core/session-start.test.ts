import { afterEach, describe, expect, it, vi } from "vitest";
import { buildSessionStartContext, runSessionStartHook } from "./session-start";
import { resolveConfig } from "./config";
import { HOOK_HARNESSES } from "../harness/hook-lifecycle";

/** Default roster the mock client returns; asserted on by name below. */
const listPagesOk = async () => ({ items: [{ id: "p1", name: "Component map" }] });

describe("buildSessionStartContext", () => {
  it("cold git repo + autoSeed on -> seeds + surveys, note in systemMessage (user-visible) + roster in additionalContext (model)", async () => {
    const client = { listDocumentIds: async () => new Set<string>(), listPages: listPagesOk };
    const startSeed = vi.fn();
    const startSurvey = vi.fn();
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      hasGit: () => true,
      startSeed,
      startSurvey,
    });
    expect(startSeed).toHaveBeenCalledWith("/repo/dir", { limit: 300, harness: "claude-code" });
    expect(startSurvey).toHaveBeenCalledWith("/repo/dir", {
      harness: "claude-code",
      model: "haiku",
      budgetUsd: 2,
    });
    // The learning note is USER-VISIBLE (systemMessage), not buried in model context.
    expect(out.systemMessage).toContain("is learning");
    expect(out.systemMessage).toContain("bank-1");
    // The knowledge preamble is model context, lists live pages, and drops the old static mission.
    expect(out.additionalContext).toContain("<hindsight_knowledge>");
    expect(out.additionalContext).toContain("- Component map (p1)");
    expect(out.additionalContext).not.toContain("agent_knowledge_list_pages");
    // The banner must NOT be duplicated into model context. (The tool guide legitimately
    // contains a "🧠 From Hindsight memory" attribution example, so match on banner text.)
    expect(out.additionalContext).not.toContain("memory bank");
    expect(out.additionalContext).not.toContain("is learning this repo");
    expect(out.deferInitialReflect).toBe(true);
  });

  it("threads the ASKING harness to the background seed (not the config loader's default) — #3247", async () => {
    // Regression: the seed used to fire without a harness, so deepen.js fell back to the config
    // loader's "opencode" default and misfiled a non-opencode session's survey + git history into
    // an `opencode::<project>` bank. The seed must receive the harness that asked.
    const client = { listDocumentIds: async () => new Set<string>(), listPages: listPagesOk };
    const startSeed = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      harness: "codex",
      hasGit: () => true,
      startSeed,
      startSurvey: vi.fn(),
    });
    expect(startSeed).toHaveBeenCalledWith("/repo/dir", { limit: 300, harness: "codex" });
  });

  it("cold git repo + codebaseSurvey:false -> starts the seed but NOT the survey", async () => {
    const client = { listDocumentIds: async () => new Set<string>(), listPages: listPagesOk };
    const startSeed = vi.fn();
    const startSurvey = vi.fn();
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig({ codebaseSurvey: false }),
      client,
      hasGit: () => true,
      startSeed,
      startSurvey,
    });
    expect(startSeed).toHaveBeenCalledWith("/repo/dir", { limit: 300, harness: "claude-code" });
    expect(startSurvey).not.toHaveBeenCalled();
    expect(out.systemMessage).toContain("is learning");
  });

  it("non-git dir -> no seed, listDocumentIds not called, roster preamble only (no learning note)", async () => {
    const startSeed = vi.fn();
    let called = false;
    const client = {
      listDocumentIds: async () => {
        called = true;
        return new Set<string>();
      },
      listPages: listPagesOk,
    };
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      hasGit: () => false,
      startSeed,
    });
    expect(startSeed).not.toHaveBeenCalled();
    expect(called).toBe(false);
    expect(out.additionalContext).toContain("- Component map (p1)");
    // banner shows on EVERY session now; non-cold paths use the "remembering" wording
    expect(out.systemMessage).toContain("is tracking the decisions");
    expect(out.deferInitialReflect).toBe(false);
  });

  it("an empty but reachable page roster defers first reflect even when git documents already exist", async () => {
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client: {
        listDocumentIds: async () => new Set(["git:abc"]),
        listPages: async () => ({ items: [] }),
      },
      hasGit: () => true,
      startSeed: vi.fn(),
    });
    expect(out.deferInitialReflect).toBe(true);
  });

  // The old "declined state -> no seed" test is gone with the seed-state file itself: the live
  // bank is the ONLY state now, so there is no client-side declined flag to consult. Opting a
  // repo out of memory is `disabled` in project config.

  it("cold-check-wins: EMPTY live bank -> (re)seeds — the bank is the only state", async () => {
    const startSeed = vi.fn();
    let called = false;
    const client = {
      listDocumentIds: async () => {
        called = true;
        return new Set<string>(); // bank is empty (fresh, or user cleared it)
      },
      listPages: listPagesOk,
    };
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      hasGit: () => true,
      startSeed,
    });
    // The live bank is consulted, and an empty bank seeds — no client-side flag can contradict it.
    expect(called).toBe(true);
    expect(startSeed).toHaveBeenCalledWith("/repo/dir", { limit: 300, harness: "claude-code" });
    expect(out.systemMessage).toContain("is learning");
  });

  it("warm bank (non-empty doc set) -> deepen engine fires, but no survey/note", async () => {
    const startSeed = vi.fn();
    const startSurvey = vi.fn();
    const client = { listDocumentIds: async () => new Set(["git:abc"]), listPages: listPagesOk };
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      hasGit: () => true,
      startSeed,
      startSurvey,
    });
    // The engine is idempotent, so every warm session start re-fires it to pick up missing work.
    expect(startSeed).toHaveBeenCalledWith("/repo/dir", { limit: 300, harness: "claude-code" });
    // The cold-only extras stay off: no survey, no user-facing learning note.
    expect(startSurvey).not.toHaveBeenCalled();
    expect(out.additionalContext).toContain("- Component map (p1)");
    // banner shows on EVERY session now; non-cold paths use the "remembering" wording
    expect(out.systemMessage).toContain("is tracking the decisions");
  });

  it("listDocumentIds throws (server unreachable) -> no seed, roster preamble only", async () => {
    const startSeed = vi.fn();
    const client = {
      listDocumentIds: async () => {
        throw new Error("network down");
      },
      listPages: listPagesOk,
    };
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      hasGit: () => true,
      startSeed,
    });
    expect(startSeed).not.toHaveBeenCalled();
    expect(out.additionalContext).toContain("- Component map (p1)");
    // banner shows on EVERY session now; non-cold paths use the "remembering" wording
    expect(out.systemMessage).toContain("is tracking the decisions");
  });

  it("listPages rejects -> fail-open: empty-state preamble, seed still starts, note still visible (cold repo)", async () => {
    const startSeed = vi.fn();
    const client = {
      listDocumentIds: async () => new Set<string>(),
      listPages: async () => {
        throw new Error("pages endpoint down");
      },
    };
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client,
      hasGit: () => true,
      startSeed,
    });
    // Seeding is unaffected by a listPages failure.
    expect(startSeed).toHaveBeenCalledWith("/repo/dir", { limit: 300, harness: "claude-code" });
    // Empty-state roster preamble still renders (no page names, no throw).
    expect(out.additionalContext).toContain("<hindsight_knowledge>");
    expect(out.additionalContext).toContain("No knowledge pages yet");
    expect(out.additionalContext).not.toContain("(p1)");
    // The background-learning note is still user-visible.
    expect(out.systemMessage).toContain("is learning");
  });

  it("autoSeed:false -> skips the whole seed branch (no listDocumentIds call), roster preamble only", async () => {
    const startSeed = vi.fn();
    let called = false;
    const client = {
      listDocumentIds: async () => {
        called = true;
        return new Set<string>();
      },
      listPages: listPagesOk,
    };
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig({ autoSeed: false }),
      client,
      hasGit: () => true,
      startSeed,
    });
    expect(startSeed).not.toHaveBeenCalled();
    expect(called).toBe(false);
    expect(out.additionalContext).toContain("- Component map (p1)");
    // banner shows on EVERY session now; non-cold paths use the "remembering" wording
    expect(out.systemMessage).toContain("is tracking the decisions");
  });
});

describe("runSessionStartHook anti-recursion guard", () => {
  const ORIGINAL = process.env.HINDSIGHT_DISABLE_HOOKS;

  afterEach(() => {
    if (ORIGINAL === undefined) delete process.env.HINDSIGHT_DISABLE_HOOKS;
    else process.env.HINDSIGHT_DISABLE_HOOKS = ORIGINAL;
  });

  it("HINDSIGHT_DISABLE_HOOKS set -> returns immediately, never reads stdin or builds a client", async () => {
    process.env.HINDSIGHT_DISABLE_HOOKS = "1";
    const makeClient = vi.fn();
    // No stdin is provided/mocked here — if the guard didn't return before `readFileSync(0, ...)`,
    // this call would attempt to read the real process stdin. Resolving without calling makeClient
    // proves the guard fired first.
    await runSessionStartHook(HOOK_HARNESSES["claude-code"].sessionStart, makeClient);
    expect(makeClient).not.toHaveBeenCalled();
  });
});

// Periodic re-survey (Option A): the survey baseline lives in the bank as `survey-baseline:<sha>`
// marker docs. Warm sessions re-survey when the MIN reachable `<sha>..HEAD` count >= threshold.
describe("buildSessionStartContext — periodic re-survey (bank-stored commit count)", () => {
  // A warm bank: source:git non-empty; source:survey-baseline returns the given marker ids.
  // findings present by default so the crashed-survey retry path stays quiet in cadence tests.
  const warmClient = (
    baselineIds: string[],
    retain = vi.fn(),
    uploads: string[] = ["repository-component-map"]
  ) => ({
    listDocumentIds: async (tag: string) =>
      tag === "source:survey-baseline"
        ? new Set(baselineIds)
        : tag === "source:upload"
          ? new Set(uploads)
          : new Set(["git:abc"]),
    listPages: listPagesOk,
    retain,
  });
  const marker = (sha: string) => [
    expect.stringContaining(sha.slice(0, 12)), // human-readable content carrying the sha
    expect.any(String),
    `survey-baseline:${sha}`,
    ["source:survey-baseline"],
    "survey", // survey-lifecycle strategy (marker rule: zero extraction)
    // Opts are always passed now; with no retainMetadata configured the stamp is empty, and
    // `retain` only sets metadata when truthy, so nothing reaches the API.
    { metadata: undefined },
  ];

  it(">= threshold since the latest reachable baseline -> re-surveys + records a new baseline", async () => {
    const startSurvey = vi.fn();
    const retain = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 20 }),
      client: warmClient(["survey-baseline:oldsha"], retain),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "newsha",
      commitsSince: () => 25,
    });
    expect(startSurvey).toHaveBeenCalledWith(
      "/repo",
      expect.objectContaining({ harness: "claude-code" })
    );
    expect(retain).toHaveBeenCalledWith(...marker("newsha"));
  });

  it("applies retain attribution to survey baseline markers", async () => {
    const retain = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({
        surveyRefreshCommits: 20,
        retainTags: ["project:{project}"],
        retainMetadata: { bank: "{bankId}" },
      }),
      client: warmClient(["survey-baseline:oldsha"], retain),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey: vi.fn(),
      headSha: () => "newsha",
      commitsSince: () => 25,
    });

    expect(retain).toHaveBeenCalledWith(
      expect.any(String),
      expect.any(String),
      "survey-baseline:newsha",
      ["project:repo", "source:survey-baseline"],
      "survey",
      { metadata: { bank: "bank-1" } }
    );
  });

  it("< threshold -> no re-survey, no new baseline", async () => {
    const startSurvey = vi.fn();
    const retain = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 20 }),
      client: warmClient(["survey-baseline:oldsha"], retain),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "newsha",
      commitsSince: () => 5,
    });
    expect(startSurvey).not.toHaveBeenCalled();
    expect(retain).not.toHaveBeenCalled();
  });

  it("no baseline yet (upgrade from a pre-feature bank) -> records HEAD as baseline, does NOT survey", async () => {
    const startSurvey = vi.fn();
    const retain = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 20 }),
      client: warmClient([], retain),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "headnow",
      commitsSince: () => 999, // must not fire on the very first (baseline) encounter
    });
    expect(startSurvey).not.toHaveBeenCalled();
    expect(retain).toHaveBeenCalledWith(...marker("headnow"));
  });

  it("all markers unreachable (rebase/gc) -> re-baselines to HEAD, does NOT survey", async () => {
    const startSurvey = vi.fn();
    const retain = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 20 }),
      client: warmClient(["survey-baseline:gone1", "survey-baseline:gone2"], retain),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "newhead",
      commitsSince: () => null, // none reachable from HEAD
    });
    expect(startSurvey).not.toHaveBeenCalled();
    expect(retain).toHaveBeenCalledWith(...marker("newhead"));
  });

  it("takes the MIN reachable count (newest survey), ignoring older + dead-branch markers", async () => {
    const startSurvey = vi.fn();
    const counts: Record<string, number | null> = { old1: 50, old2: 10, dead: null };
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 20 }),
      client: warmClient(["survey-baseline:old1", "survey-baseline:old2", "survey-baseline:dead"]),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "head",
      commitsSince: (_d: string, sha: string) => counts[sha] ?? null,
    });
    expect(startSurvey).not.toHaveBeenCalled(); // min reachable (old2 = 10) < 20
  });

  it("surveyRefreshCommits=0 disables re-survey even far past threshold", async () => {
    const startSurvey = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 0 }),
      client: warmClient(["survey-baseline:old"]),
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "newsha",
      commitsSince: () => 999,
    });
    expect(startSurvey).not.toHaveBeenCalled();
  });

  it("cold seed records the survey baseline", async () => {
    const startSurvey = vi.fn();
    const retain = vi.fn();
    await buildSessionStartContext({
      cwd: "/repo",
      bankId: "bank-1",
      cfg: resolveConfig(),
      client: { listDocumentIds: async () => new Set<string>(), listPages: listPagesOk, retain },
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "seedhead",
      commitsSince: () => 0,
    });
    expect(startSurvey).toHaveBeenCalled();
    expect(retain).toHaveBeenCalledWith(...marker("seedhead"));
  });
});

describe("buildSessionStartContext — crashed-survey retry (baseline without findings)", () => {
  it("re-fires the survey when a baseline exists but NO findings docs ever arrived", async () => {
    const startSurvey = vi.fn();
    const retain = vi.fn();
    const client = {
      listDocumentIds: async (tag: string) =>
        tag === "source:survey-baseline"
          ? new Set(["survey-baseline:aaa"])
          : tag === "source:upload"
            ? new Set<string>() // survey died before ingesting anything
            : new Set(["git:abc"]),
      listPages: async () => ({ items: [] }),
      retain,
    };
    const out = await buildSessionStartContext({
      cwd: "/tmp/x",
      bankId: "bank-1",
      cfg: resolveConfig({ surveyRefreshCommits: 50 }),
      client,
      hasGit: () => true,
      startSeed: vi.fn(),
      startSurvey,
      headSha: () => "bbb",
      commitsSince: () => 1, // far below the cadence threshold — retry must fire anyway
    });
    expect(startSurvey).toHaveBeenCalledTimes(1);
    expect(out).toBeTruthy();
  });
});
