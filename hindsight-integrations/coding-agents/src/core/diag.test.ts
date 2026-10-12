import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseHookEvent } from "./diag";

describe("parseHookEvent", () => {
  let dir: string;
  let diagFile: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hs-diag-"));
    diagFile = join(dir, "diag.jsonl");
    vi.stubEnv("HINDSIGHT_DIAG_FILE", diagFile);
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    rmSync(dir, { recursive: true, force: true });
  });

  // #5366: Cursor on Windows pipes the event through PowerShell, which prefixes a BOM — sometimes
  // two, and sometimes decoded through Windows-1252 into "ï»¿".
  it.each(["﻿", "﻿﻿", "ï»¿", "﻿ï»¿"])("strips a leading %j BOM", (bom) => {
    expect(parseHookEvent("cursor-cli", `${bom}{"conversation_id":"c1"}`)).toEqual({
      conversation_id: "c1",
    });
    expect(existsSync(diagFile)).toBe(false);
  });

  it("records invalid input in the diag file instead of failing silently", () => {
    expect(parseHookEvent("cursor-cli", "not json")).toBeNull();
    expect(parseHookEvent("cursor-cli", "[1]")).toBeNull();
    const events = readFileSync(diagFile, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as { harness: string; event: string });
    expect(events.map((e) => [e.harness, e.event])).toEqual([
      ["cursor-cli", "hook_input_invalid"],
      ["cursor-cli", "hook_input_invalid"],
    ]);
  });

  it("treats empty input as no event, without a diag record", () => {
    expect(parseHookEvent("cursor-cli", "")).toBeNull();
    expect(existsSync(diagFile)).toBe(false);
  });
});
