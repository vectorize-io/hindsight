import { mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

// The config path is read once at import, so point it at a scratch file before dsh.ts loads.
const { configFile } = vi.hoisted(() => {
  const { mkdtempSync } = require("node:fs") as typeof import("node:fs");
  const { tmpdir } = require("node:os") as typeof import("node:os");
  const { join } = require("node:path") as typeof import("node:path");
  const file = join(mkdtempSync(join(tmpdir(), "hs-dsh-cfg-")), "coding-agent.json");
  process.env.HINDSIGHT_CONFIG = file;
  return { configFile: file };
});

const { toolTemplate } = await import("./dsh");

// Real path: macOS tmpdir is a symlink, and bank resolution compares resolved directories.
const scratch = realpathSync(mkdtempSync(join(tmpdir(), "hs-dsh-tools-")));
const dir = (name: string) => {
  const d = join(scratch, name);
  mkdirSync(d, { recursive: true });
  return d;
};

describe("dsh tool registry template (#5817)", () => {
  it("uses an opted-in project when the host runs from a directory that is not opted in", () => {
    const profile = dir("profiles/desktop");
    const repo = dir("repo");
    writeFileSync(
      configFile,
      JSON.stringify({ apiUrl: "http://127.0.0.1:9", optInOnly: true, optInPaths: [repo] })
    );
    expect(toolTemplate(profile)?.root).toBe(repo);
  });

  it("registers nothing when memory is on for no configured directory", () => {
    const profile = dir("profiles/web");
    writeFileSync(
      configFile,
      JSON.stringify({ apiUrl: "http://127.0.0.1:9", optInOnly: true, optInPaths: [] })
    );
    expect(toolTemplate(profile)).toBeUndefined();
  });
});
