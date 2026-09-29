import { writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readDroidTranscript } from "./transcript-droid";

/** Build one Droid transcript line the way the CLI writes it. */
const message = (
  role: "user" | "assistant",
  content: unknown,
  opts: { visibility?: string; timestamp?: string } = {}
) => ({
  type: "message",
  ...(opts.timestamp ? { timestamp: opts.timestamp } : {}),
  message: {
    role,
    ...(opts.visibility ? { visibility: opts.visibility } : {}),
    content,
  },
});

describe("readDroidTranscript", () => {
  it("normalizes Droid messages and compacts tool calls into action turns", () => {
    const path = "/tmp/hindsight-droid-transcript-test.jsonl";
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "session_start",
          id: "s1",
          cwd: "/repo",
        }),
        JSON.stringify(
          message("user", [{ type: "text", text: "Fix the retry policy" }], {
            timestamp: "2026-09-01T10:00:00.000Z",
          })
        ),
        JSON.stringify(
          message(
            "assistant",
            [
              { type: "text", text: "I will inspect the policy." },
              { type: "tool_use", name: "Read", input: { file_path: "/repo/src/retry.ts" } },
            ],
            { timestamp: "2026-09-01T10:00:01.000Z" }
          )
        ),
        JSON.stringify({ type: "message", message: { role: "user", content: [] } }),
        JSON.stringify({ type: "agent_turn_outcome", outcome: "done" }),
        "not json",
        JSON.stringify(null),
      ].join("\n")
    );

    const turns = readDroidTranscript(path);
    expect(turns.map((t) => [t.role, t.timestamp])).toEqual([
      ["user", "2026-09-01T10:00:00.000Z"],
      ["assistant", "2026-09-01T10:00:01.000Z"],
      ["action", "2026-09-01T10:00:01.000Z"],
    ]);
    expect(turns[0]?.content).toBe("Fix the retry policy");
    expect(turns[1]?.content).toBe("I will inspect the policy.");
    // The action turn names the tool and its primary target, no arguments.
    expect(turns[2]?.content).toContain("Read");
    expect(turns[2]?.content).toContain("retry.ts");
    expect(turns[2]?.content).not.toContain("file_path");
  });

  it("skips messages visible to only the user or model", () => {
    const path = "/tmp/hindsight-droid-transcript-hooklines.jsonl";
    writeFileSync(
      path,
      [
        JSON.stringify(
          message(
            "user",
            [{ type: "text", text: "<system-reminder>hook context</system-reminder>" }],
            {
              visibility: "user_only",
            }
          )
        ),
        JSON.stringify(
          message("assistant", [{ type: "text", text: "visible reply" }], {
            visibility: "user_only",
          })
        ),
        // Model-only context is not necessarily wrapped in a removable tag. It still must not be
        // retained as something the user said.
        JSON.stringify(
          message("user", [{ type: "text", text: "plain model-only system context" }], {
            visibility: "llm_only",
          })
        ),
        JSON.stringify(message("user", [{ type: "text", text: "real prompt" }])),
        JSON.stringify(
          message("assistant", [{ type: "text", text: "real reply" }], {
            visibility: "both",
          })
        ),
      ].join("\n")
    );

    expect(readDroidTranscript(path)).toEqual([
      { role: "user", content: "real prompt" },
      { role: "assistant", content: "real reply" },
    ]);
  });

  it("strips injected memory blocks so recall output never feeds back into the bank", () => {
    const path = "/tmp/hindsight-droid-transcript-injected.jsonl";
    writeFileSync(
      path,
      [
        JSON.stringify(
          message("user", [
            {
              type: "text",
              text: "before <hindsight_memories>recalled facts</hindsight_memories> after",
            },
          ])
        ),
      ].join("\n")
    );

    expect(readDroidTranscript(path)).toEqual([{ role: "user", content: "before  after" }]);
  });

  it("fails open for a missing or malformed transcript", () => {
    expect(readDroidTranscript("/tmp/does-not-exist-droid-transcript.jsonl")).toEqual([]);
    const path = "/tmp/hindsight-droid-transcript-garbage.jsonl";
    writeFileSync(path, "{broken\n[1,2]\n");
    expect(readDroidTranscript(path)).toEqual([]);
  });
});
