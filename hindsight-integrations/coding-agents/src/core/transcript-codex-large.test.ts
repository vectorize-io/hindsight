import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    readFileSync: vi.fn(() => {
      const error = new Error("Cannot create a string longer than 0x1fffffe8 characters");
      Object.assign(error, { code: "ERR_STRING_TOO_LONG" });
      throw error;
    }),
  };
});

import { readCodexTranscript } from "./transcript-codex";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-codex-large-transcript-"));
  file = join(root, "rollout.jsonl");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("readCodexTranscript oversized rollout handling", () => {
  it("parses JSONL without reading the complete file into one string", () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "response_item",
        payload: {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "keep streaming" }],
        },
      })
    );

    expect(readCodexTranscript(file)).toEqual([{ role: "user", content: "keep streaming" }]);
  });
});
