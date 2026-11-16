import { mkdtempSync, readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { buildMemoryTreeContext, memoryTreeDir, writeMemoryTree } from "./memory-tree";

const tree = {
  path: "",
  files: [
    { path: "index.md", kind: "index" as const, content: "# Memory\n- prices.md" },
    { path: "../escape.md", kind: "topic" as const, content: "no" },
  ],
  folders: [
    {
      path: "db",
      files: [{ path: "db/prices.md", kind: "topic" as const, content: "prc is cents" }],
      folders: [],
    },
  ],
};

describe("memory tree", () => {
  it("replaces the folder with the tree's files and returns the root index", () => {
    const dir = join(mkdtempSync(join(tmpdir(), "mt-")), "memory");
    mkdirSync(dir);
    writeFileSync(join(dir, "stale.md"), "old");
    expect(writeMemoryTree(dir, tree)).toBe("# Memory\n- prices.md");
    expect(readFileSync(join(dir, "db/prices.md"), "utf8")).toBe("prc is cents");
    expect(existsSync(join(dir, "stale.md"))).toBe(false);
    expect(existsSync(join(dir, "..", "escape.md"))).toBe(false);
  });

  it("keeps the folder name `memory`, so the agent's notes there are retained", () => {
    expect(memoryTreeDir("claude-code::my/repo", "/h")).toBe(
      "/h/.hindsight/memory-tree/claude-code__my_repo/memory"
    );
  });

  it("points at the folder and caps the inlined index", () => {
    const ctx = buildMemoryTreeContext("/m", `${"line\n".repeat(2000)}`);
    expect(ctx).toContain("in /m, as markdown files");
    expect(ctx).toContain("(… rest in index.md)");
    expect(ctx.length).toBeLessThan(7000);
    expect(buildMemoryTreeContext("/m", "")).toContain("The folder is empty");
  });
});
