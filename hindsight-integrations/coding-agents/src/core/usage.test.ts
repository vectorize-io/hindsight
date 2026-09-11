import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TransportTurn } from "./chat";
import {
  formatUsageReport,
  memoryUsageCursorStore,
  readUsage,
  recordUsage,
  summarizeTurns,
} from "./usage";

const t = (role: string, content: string): TransportTurn => ({ role, content });

let root: string;
let usageFile: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-usage-"));
  usageFile = join(root, "usage.jsonl");
  vi.stubEnv("HINDSIGHT_USAGE_FILE", usageFile);
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

const lines = () =>
  readFileSync(usageFile, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));

describe("summarizeTurns", () => {
  it("counts only Hindsight tools, whatever prefix the host gives them", () => {
    const usage = summarizeTurns([
      t("system", "REF-ID: x"), // before the first user turn: belongs to no turn
      t("user", "how do we round?"),
      t("action", "mcp__hindsight__hindsight_search_knowledge_pages rounding"),
      t("action", "Read src/round.ts"),
      t("action", "hindsight.hindsight_reflect why half-up"),
      t("assistant", "> 🧠 **From Hindsight memory (Conventions)** — half up"),
      t("user", "thanks"),
      t("action", "hindsight_read_knowledge_page kp-1"),
      t("assistant", "done"),
    ]);
    expect(usage).toEqual([
      {
        turn: 1,
        calls: ["hindsight_search_knowledge_pages", "hindsight_reflect"],
        credited: true,
      },
      { turn: 2, calls: ["hindsight_read_knowledge_page"], credited: false },
    ]);
  });

  it("reads credit from Gemini-shaped `model` turns, and never from the user's own words", () => {
    const usage = summarizeTurns([
      t("user", "did it come From Hindsight memory?"),
      t("user", "second"),
      t("model", "🧠 From Hindsight memory — yes"),
    ]);
    expect(usage.map((u) => u.credited)).toEqual([false, true]);
  });
});

describe("recordUsage", () => {
  const base = { harness: "claude-code", sessionId: "s1", bankId: "bank-1" };

  it("records each turn once across repeated Stops over the growing transcript", () => {
    const cursors = memoryUsageCursorStore();
    const first = [t("user", "a"), t("action", "mcp__hindsight__hindsight_reflect q")];
    recordUsage({ ...base, turns: first, cursors, lastTurnComplete: true });
    recordUsage({ ...base, turns: first, cursors, lastTurnComplete: true }); // Stop re-delivered
    recordUsage({
      ...base,
      turns: [...first, t("user", "b"), t("assistant", "ok")],
      cursors,
      lastTurnComplete: true,
    });

    expect(lines()).toMatchObject([
      {
        harness: "claude-code",
        session: "s1",
        bank: "bank-1",
        turn: 1,
        calls: ["hindsight_reflect"],
      },
      { turn: 2, calls: [], credited: false },
    ]);
  });

  it("holds back an unanswered last turn instead of recording it empty", () => {
    const cursors = memoryUsageCursorStore();
    recordUsage({ ...base, turns: [t("user", "a")], cursors, lastTurnComplete: false });
    expect(() => readFileSync(usageFile)).toThrow(); // nothing finished yet

    recordUsage({
      ...base,
      turns: [t("user", "a"), t("action", "hindsight_reflect q"), t("assistant", "done")],
      cursors,
      lastTurnComplete: true,
    });
    expect(lines()).toMatchObject([{ turn: 1, calls: ["hindsight_reflect"] }]);
  });
});

describe("readUsage", () => {
  it("aggregates per harness, dedupes re-recorded turns and includes the rotated generation", () => {
    const line = (o: object) => JSON.stringify({ ts: "x", bank: "b", ...o }) + "\n";
    writeFileSync(
      `${usageFile}.1`,
      line({
        harness: "codex",
        session: "a",
        turn: 1,
        calls: ["hindsight_reflect"],
        credited: true,
      })
    );
    writeFileSync(
      usageFile,
      line({
        harness: "codex",
        session: "a",
        turn: 1,
        calls: ["hindsight_reflect"],
        credited: true,
      }) +
        line({ harness: "codex", session: "a", turn: 2, calls: [], credited: false }) +
        // A write-only turn: counts as a call, never against the credit rate.
        line({
          harness: "codex",
          session: "a",
          turn: 3,
          calls: ["hindsight_ingest_document"],
          credited: false,
        }) +
        line({
          harness: "codex",
          session: "b",
          turn: 1,
          calls: ["hindsight_reflect", "hindsight_search_knowledge_pages"],
          credited: false,
        }) +
        "not json\n"
    );

    const usage = readUsage(usageFile);
    expect(usage).toEqual([
      {
        harness: "codex",
        turns: 4,
        turnsWithCalls: 3,
        calls: 4,
        turnsWithRetrieval: 2,
        creditedTurnsWithRetrieval: 1,
        byTool: {
          hindsight_reflect: 2,
          hindsight_search_knowledge_pages: 1,
          hindsight_ingest_document: 1,
        },
      },
    ]);
    expect(formatUsageReport(usage, usageFile)).toContain(
      "codex: 4 turns · 3 with a Hindsight call (75%) · 1.00 calls/turn · " +
        "credited after retrieval 1/2 (50%)"
    );
  });

  it("says so when nothing is recorded", () => {
    expect(formatUsageReport(readUsage(usageFile), usageFile)).toBe(
      `no Hindsight usage recorded yet (${usageFile})`
    );
  });
});
