import { describe, expect, it } from "vitest";
import type { SessionEvent } from "@/lib/api";
import {
  buildTimeline,
  checkSpans,
  formatClock,
  decisionPath,
  filterRows,
  isActive,
  isDelivered,
  nextPollDelay,
  noteLatencyMs,
  parseDecisionDetail,
  replayEvents,
  replayOffsets,
  toolTarget,
} from "@/lib/session-timeline";

let clock = Date.parse("2026-10-19T19:00:00Z");
function ev(
  seq: number,
  source: "client" | "hindsight",
  kind: string,
  payload: Record<string, unknown>,
  stepMs = 1000
): SessionEvent {
  clock += stepMs;
  return {
    seq,
    event_id: `id${seq}`,
    source,
    kind,
    created_at: new Date(clock).toISOString(),
    payload,
  };
}

// A small session: request, a Read, an Edit whose result failed, a check on the
// request that spoke, its note, and a silent check on the edit.
function session() {
  clock = Date.parse("2026-10-19T19:00:00Z");
  return [
    ev(1, "client", "message", { role: "user", text: "Fix bucket" }),
    ev(2, "client", "tool_call", {
      role: "assistant",
      tool: "Read",
      call_id: "a",
      input: { file_path: "/work/flagcore/bucket.py" },
    }),
    ev(3, "client", "tool_result", { role: "tool", call_id: "a", text: "def bucket" }),
    ev(4, "hindsight", "check", { trigger: "task", moment_seq: 1, outcome: "note" }, 4000),
    ev(5, "hindsight", "note", { text: "Hash user_id + flag", check_seq: 4 }, 0),
    ev(6, "client", "tool_call", {
      role: "assistant",
      tool: "Edit",
      call_id: "b",
      input: { file_path: "/work/flagcore/bucket.py" },
    }),
    ev(7, "client", "tool_result", { role: "tool", call_id: "b", text: "boom", is_error: true }),
    ev(8, "hindsight", "check", { trigger: "edit", moment_seq: 7, outcome: "silent" }),
    ev(9, "hindsight", "check", { trigger: "plan", moment_seq: 99, outcome: "error", error: "x" }),
  ];
}

describe("buildTimeline", () => {
  it("folds each tool result under its call", () => {
    const { rows } = buildTimeline(session());
    const tools = rows.filter((r) => r.kind === "tool");
    expect(tools).toHaveLength(2);
    expect(tools[0]).toMatchObject({
      tool: "Read",
      target: "flagcore/bucket.py",
      family: "read",
      isError: false,
    });
    expect(tools[0].kind === "tool" && tools[0].result?.seq).toBe(3);
    expect(tools[1]).toMatchObject({ tool: "Edit", family: "edit", isError: true });
  });

  it("attaches a check to the row of its moment, even when the moment is a folded result", () => {
    const { rows, rowKeyBySeq } = buildTimeline(session());
    const user = rows.find((r) => r.kind === "message")!;
    expect(user.checks.map((c) => c.seq)).toEqual([4]);
    const edit = rows.find((r) => r.kind === "tool" && r.tool === "Edit")!;
    expect(edit.checks.map((c) => c.seq)).toEqual([8]);
    expect(rowKeyBySeq.get(8)).toBe(edit.key);
  });

  it("keeps a check whose moment is not in the log as its own row", () => {
    const { rows } = buildTimeline(session());
    expect(rows.filter((r) => r.kind === "check").map((r) => r.seq)).toEqual([9]);
  });

  it("joins a result that arrives before its call", () => {
    const { rows } = buildTimeline([
      ev(1, "client", "tool_result", { call_id: "z", text: "out" }),
      ev(2, "client", "tool_call", {
        tool: "Bash",
        call_id: "z",
        input: { command: "pytest -q\nmore" },
      }),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ tool: "Bash", target: "pytest -q ⏎ more", family: "command" });
  });
});

describe("filterRows", () => {
  const base = { tools: true, silent: true, hindsightOnly: false, query: "" };

  it("hides quiet checks but keeps the loud ones", () => {
    const rows = filterRows(buildTimeline(session()).rows, { ...base, silent: false });
    const edit = rows.find((r) => r.kind === "tool" && r.tool === "Edit")!;
    expect(edit.checks).toHaveLength(0);
    expect(rows.find((r) => r.kind === "message")!.checks).toHaveLength(1);
  });

  it("hiding tool calls keeps a tool row that hosts a check", () => {
    const rows = filterRows(buildTimeline(session()).rows, { ...base, tools: false });
    expect(rows.filter((r) => r.kind === "tool").map((r) => r.kind === "tool" && r.tool)).toEqual([
      "Edit",
    ]);
  });

  it("hindsight only keeps notes, checks and the moments they judged", () => {
    const rows = filterRows(buildTimeline(session()).rows, { ...base, hindsightOnly: true });
    expect(rows.map((r) => r.seq)).toEqual([1, 5, 6, 9]);
  });

  it("searches row text and the checks attached to it", () => {
    const rows = filterRows(buildTimeline(session()).rows, { ...base, query: "user_id" });
    expect(rows.map((r) => r.kind)).toEqual(["note"]);
  });
});

describe("delivery and latency", () => {
  it("a note is delivered once acked_seq reaches it", () => {
    expect(isDelivered(5, 4)).toBe(false);
    expect(isDelivered(5, 5)).toBe(true);
  });

  it("measures from the moment's event to the note", () => {
    const events = session();
    const { eventBySeq } = buildTimeline(events);
    expect(noteLatencyMs(events[4], eventBySeq)).toBe(6000);
    expect(noteLatencyMs(ev(10, "hindsight", "note", { check_seq: 42 }), eventBySeq)).toBeNull();
  });
});

describe("decision path", () => {
  it("parses each route the decision model can take", () => {
    expect(
      parseDecisionDetail("none=0.02, kept 1 of 8 offered: settled: the writer words the answer")
    ).toEqual({
      none: 0.02,
      kept: 1,
      offered: 8,
      route: "writer",
    });
    expect(
      parseDecisionDetail("none=0.87, kept 1 of 8 offered: unsure: the reflect LLM judges").route
    ).toBe("llm");
    expect(parseDecisionDetail("none=0.94 of 8 offered: silent, no LLM")).toEqual({
      none: 0.94,
      kept: null,
      offered: 8,
      route: "silent",
    });
    expect(parseDecisionDetail("nothing found: silent, no LLM").route).toBe("nothing");
  });

  it("splits the pick from the calls that follow it", () => {
    const path = decisionPath([
      {
        scope: "decision_pick",
        duration_ms: 406,
        detail: "none=0.00, kept 1 of 8 offered: settled: x",
      },
      { scope: "abstain_answer", duration_ms: 1106 },
    ]);
    expect(path.pick).toMatchObject({ route: "writer", ms: 406 });
    expect(path.after.map((c) => c.scope)).toEqual(["abstain_answer"]);
    expect(decisionPath(undefined)).toEqual({ pick: null, after: [] });
  });
});

describe("tool families", () => {
  it("knows a tool by the words of its name, whichever agent named it", () => {
    const familyOf = (tool: string) =>
      buildTimeline([
        {
          seq: 1,
          event_id: "c",
          source: "client",
          kind: "tool_call",
          created_at: "2026-10-19T19:00:00Z",
          payload: { role: "assistant", tool, call_id: "c", input: {} },
        },
      ]).rows[0];
    const cases: [string, string][] = [
      ["Edit", "edit"],
      ["apply_patch", "edit"],
      ["write_file", "edit"],
      ["patch", "edit"],
      ["Bash", "command"],
      ["terminal", "command"],
      ["exec_command", "command"],
      ["Read", "read"],
      ["read_file", "read"],
      ["search_files", "read"],
      ["TodoWrite", "read"],
      ["dispatch_task", "other"],
    ];
    for (const [tool, family] of cases) expect(familyOf(tool)).toMatchObject({ family });
  });
});

describe("live rules", () => {
  it("is live only when recently written, otherwise idle", () => {
    const now = Date.parse("2026-10-19T19:10:00Z");
    expect(isActive({ last_event_at: "2026-10-19T19:09:00Z" }, now)).toBe(true);
    expect(isActive({ last_event_at: "2026-10-19T19:00:00Z" }, now)).toBe(false);
    expect(isActive({ last_event_at: null }, now)).toBe(false);
  });

  it("polls fast while busy, backs off when idle or hidden", () => {
    const p = { visible: true, checking: false, gotNew: false, idleMs: 0 };
    expect(nextPollDelay(p)).toBe(1000);
    expect(nextPollDelay({ ...p, idleMs: 5 * 60_000 })).toBe(3000);
    expect(nextPollDelay({ ...p, idleMs: 60 * 60_000 })).toBe(10_000);
    expect(nextPollDelay({ ...p, idleMs: 60 * 60_000, checking: true })).toBe(1000);
    expect(nextPollDelay({ ...p, visible: false })).toBe(15_000);
  });

  it("replays at real pace with dead gaps capped", () => {
    clock = 0;
    const events = [
      ev(1, "client", "x", {}, 0),
      ev(2, "client", "x", {}, 2000),
      ev(3, "client", "x", {}, 60_000),
    ];
    expect(replayOffsets(events, 10_000)).toEqual([0, 2000, 12_000]);
  });
});

describe("toolTarget", () => {
  it("picks the main argument", () => {
    expect(toolTarget({ pattern: "crc32" })).toBe("crc32");
    expect(toolTarget({ command: ["git", "status"] })).toBe("git status");
    expect(toolTarget(undefined)).toBe("");
  });
});

describe("replay shows checks while they run", () => {
  // The task check ends 6s after the prompt and took 5s: it started 1s after it.
  function log() {
    clock = Date.parse("2026-10-19T19:00:00Z");
    return [
      ev(1, "client", "message", { role: "user", text: "Fix bucket" }),
      ev(2, "client", "tool_call", { tool: "Read", call_id: "a", input: {} }, 3000),
      ev(
        3,
        "hindsight",
        "check",
        { trigger: "task", moment_seq: 1, outcome: "note", ms: 5000, answer: "x" },
        3000
      ),
      ev(4, "hindsight", "note", { text: "x", check_seq: 3 }, 0),
    ];
  }

  it("places a check from its start to when it was logged, never before its moment", () => {
    const events = log();
    const offsets = replayOffsets(events);
    expect(checkSpans(events, offsets)).toEqual([
      { seq: 3, start: 1000, end: 6000, outcome: "note", trigger: "task" },
    ]);
    events[2].payload.ms = 60_000; // longer than the replay clock gives it
    expect(checkSpans(events, offsets)[0].start).toBe(0);
  });

  it("shows the running check on its moment without its outcome, then the real one", () => {
    const events = log();
    const offsets = replayOffsets(events);
    const spans = checkSpans(events, offsets);
    const mid = buildTimeline(replayEvents(events, offsets, spans, 4000));
    const running = mid.rows[0].checks[0].check;
    expect(running.running_ms).toBe(3000);
    expect(running.outcome).toBeUndefined();
    expect(running.answer).toBeUndefined();
    expect(mid.rows.some((r) => r.kind === "note")).toBe(false);
    // The source log is untouched.
    expect(events[2].payload.outcome).toBe("note");

    const done = buildTimeline(replayEvents(events, offsets, spans, 6000));
    expect(done.rows[0].checks[0].check.outcome).toBe("note");
    expect(done.rows[0].checks[0].check.running_ms).toBeUndefined();
    expect(done.rows.some((r) => r.kind === "note")).toBe(true);
  });
});

describe("handoffs", () => {
  it("draws the server's handoff entry where it was logged, with the row it went back with", () => {
    const events = [
      ...session(),
      ev(10, "client", "message", { role: "assistant", text: "Looking." }),
      ev(11, "hindsight", "handoff", { note_seq: 5, with_seqs: [10] }),
    ];
    const rows = buildTimeline(events).rows;
    const h = rows[rows.length - 1];
    expect(h.kind).toBe("handoff");
    if (h.kind !== "handoff") return;
    expect(h.note.seq).toBe(5);
    expect(h.host?.kind === "message" && h.host.text).toBe("Looking.");
  });

  it("has no host when a poll fetched the note", () => {
    const rows = buildTimeline([
      ...session(),
      ev(10, "hindsight", "handoff", { note_seq: 5, with_seqs: [] }),
    ]).rows;
    const h = rows[rows.length - 1];
    expect(h.kind === "handoff" && h.host).toBeNull();
  });
});

describe("formatClock", () => {
  it("reads the same on both sides: minutes and seconds since the start", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(41_900)).toBe("0:41");
    expect(formatClock(725_000)).toBe("12:05");
    expect(formatClock(3_729_000)).toBe("1:02:09");
  });
});
