import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readZcodeTranscript, zcodeAssistantText } from "./transcript-zcode";

let root: string;
let file: string;

/** ZCode's ephemeral Stop transcript: a bare `{message}` per line, with NO top-level `type`. */
const line = (role: string, text: string, extra: Record<string, unknown> = {}) =>
  JSON.stringify({ ...extra, message: { role, content: [{ type: "text", text }] } });

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-zcode-transcript-"));
  file = join(root, "ephemeral.jsonl");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("readZcodeTranscript", () => {
  it("reads the ephemeral shape, whose records carry no top-level type", () => {
    writeFileSync(file, [line("assistant", "first"), line("assistant", "second")].join("\n"));
    expect(readZcodeTranscript(file).map((t) => [t.role, t.content])).toEqual([
      ["assistant", "first"],
      ["assistant", "second"],
    ]);
  });

  it("reads the typed envelope too, and keeps its timestamp", () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "assistant",
        timestamp: "2026-01-01T00:00:00Z",
        message: { role: "assistant", content: [{ type: "text", text: "hi" }] },
      })
    );
    expect(readZcodeTranscript(file)).toEqual([
      { role: "assistant", content: "hi", timestamp: "2026-01-01T00:00:00Z" },
    ]);
  });

  it("accepts a plain string content as well as a block list", () => {
    writeFileSync(file, JSON.stringify({ message: { role: "assistant", content: "plain" } }));
    expect(readZcodeTranscript(file)[0].content).toBe("plain");
  });

  it("joins the text blocks and ignores the non-text ones", () => {
    writeFileSync(
      file,
      JSON.stringify({
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "before" },
            { type: "tool_use", name: "shell", input: { cmd: "ls" } },
            { type: "text", text: "after" },
          ],
        },
      })
    );
    expect(readZcodeTranscript(file)[0].content).toBe("before\nafter");
  });

  it("strips memory this integration injected, so a reply never re-ingests it", () => {
    writeFileSync(
      file,
      line("assistant", "done <hindsight_memories>recalled fact</hindsight_memories>")
    );
    expect(readZcodeTranscript(file)[0].content).toBe("done");
  });

  it("drops records that are neither user nor assistant, and empty ones", () => {
    writeFileSync(
      file,
      [
        line("system", "telemetry"),
        line("assistant", "   "),
        JSON.stringify({ message: { role: "assistant" } }),
        line("assistant", "kept"),
      ].join("\n")
    );
    expect(readZcodeTranscript(file).map((t) => t.content)).toEqual(["kept"]);
  });

  /** A Stop hook that throws loses the very turn it exists to retain. */
  it("fails open on a missing file, a blank path, and malformed lines", () => {
    expect(readZcodeTranscript(join(root, "absent.jsonl"))).toEqual([]);
    expect(readZcodeTranscript("")).toEqual([]);
    writeFileSync(file, ["null", "7", "{not json", "", line("assistant", "kept")].join("\n"));
    expect(readZcodeTranscript(file).map((t) => t.content)).toEqual(["kept"]);
  });
});

describe("zcodeAssistantText", () => {
  it("returns the LAST assistant reply — the one this Stop is closing", () => {
    writeFileSync(
      file,
      [line("assistant", "earlier"), line("user", "and then?"), line("assistant", "latest")].join(
        "\n"
      )
    );
    expect(zcodeAssistantText(file)).toBe("latest");
  });

  it("returns an empty string when there is no transcript or no reply in it", () => {
    expect(zcodeAssistantText(undefined)).toBe("");
    expect(zcodeAssistantText(join(root, "absent.jsonl"))).toBe("");
    writeFileSync(file, line("user", "only a prompt"));
    expect(zcodeAssistantText(file)).toBe("");
  });
});
