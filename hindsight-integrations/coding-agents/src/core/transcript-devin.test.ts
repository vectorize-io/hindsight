import { execFileSync } from "node:child_process";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { parseDevinMessages, readDevinTranscript } from "./transcript-devin";

vi.mock("node:child_process", () => ({ execFileSync: vi.fn() }));

const execFile = vi.mocked(execFileSync);

beforeEach(() => {
  execFile.mockReset();
});

describe("parseDevinMessages", () => {
  it("uses the final streamed assistant update", () => {
    expect(
      parseDevinMessages([
        {
          node_id: 1,
          chat_message: JSON.stringify({
            message_id: "u",
            role: "user",
            content: "Explain this repo",
          }),
        },
        {
          node_id: 2,
          chat_message: JSON.stringify({ message_id: "a", role: "assistant", content: "Partial" }),
        },
        {
          node_id: 3,
          chat_message: JSON.stringify({
            message_id: "a",
            role: "assistant",
            content: "Complete answer",
          }),
        },
      ])
    ).toEqual([
      { role: "user", content: "Explain this repo" },
      { role: "assistant", content: "Complete answer" },
    ]);
  });
});

describe("readDevinTranscript", () => {
  it("returns an empty transcript when sqlite3 successfully finds no rows", () => {
    execFile.mockReturnValue("");

    expect(readDevinTranscript("session-empty")).toEqual([]);
  });

  it("parses a successful sqlite3 query", () => {
    execFile.mockReturnValue(
      JSON.stringify([
        {
          node_id: 1,
          chat_message: JSON.stringify({
            message_id: "u",
            role: "user",
            content: "Remember the release process",
          }),
        },
      ])
    );

    expect(readDevinTranscript("session-ok")).toEqual([
      { role: "user", content: "Remember the release process" },
    ]);
    expect(execFile).toHaveBeenCalledWith(
      "sqlite3",
      expect.arrayContaining(["-json", expect.stringContaining("sessions.db")]),
      expect.objectContaining({ stdio: ["ignore", "pipe", "pipe"] })
    );
  });

  it.each([
    new Error("spawnSync sqlite3 ENOENT"),
    new Error("Error: database disk image is malformed"),
  ])("propagates sqlite3 failures for the retain diagnostic boundary", (error) => {
    execFile.mockImplementation(() => {
      throw error;
    });

    expect(() => readDevinTranscript("session-broken")).toThrow(error.message);
  });
});
