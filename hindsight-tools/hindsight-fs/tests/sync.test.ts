import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { runSync } from "../src/sync.js";
import { parseDocument } from "../src/frontmatter.js";
import type { KnowledgeNode } from "../src/client.js";
import type { MountConfig } from "../src/config.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResp(obj: unknown) {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(obj),
    text: () => Promise.resolve(JSON.stringify(obj)),
  });
}

/** Route the two knowledge-base fetches (tree + export) for a sync pass. */
function setKb(roots: KnowledgeNode[], content: Record<string, string>) {
  mockFetch.mockImplementation((url: string) => {
    if (url.includes("/knowledge-base/tree")) return jsonResp({ roots });
    if (url.includes("/knowledge-base/export")) {
      const files = Object.entries(content).map(([id, c]) => ({ path: `${id}.md`, content: c }));
      return jsonResp({ files });
    }
    return jsonResp({});
  });
}

function page(id: string, name: string, parent_id: string | null = null): KnowledgeNode {
  return { id, kind: "page", name, parent_id, children: [] };
}
function folder(
  id: string,
  name: string,
  children: KnowledgeNode[],
  parent_id: string | null = null
): KnowledgeNode {
  return { id, kind: "folder", name, parent_id, mission: `mission ${name}`, children };
}

let dir: string;

function config(overrides: Partial<MountConfig> = {}): MountConfig {
  return {
    dir,
    apiUrl: "http://localhost:8000",
    bankId: "acme",
    intervalSeconds: 30,
    writable: false,
    ...overrides,
  };
}

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "hsfs-test-"));
  mockFetch.mockReset();
});
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true });
});

describe("runSync", () => {
  it("mirrors the folder/page tree as nested directories + .md files", async () => {
    setKb([folder("f1", "Policies", [page("p1", "Billing", "f1")]), page("p2", "Glossary")], {
      p1: "---\nid: p1\ntitle: Billing\n---\n\nNet-30.\n",
      p2: "---\nid: p2\n---\n\nTerms.\n",
    });

    const result = await runSync(config());
    expect(result.total).toBe(2);
    expect(result.folders).toBe(1);

    const billing = await fs.readFile(path.join(dir, "policies", "billing.md"), "utf8");
    expect(parseDocument(billing).body.trim()).toBe("Net-30.");
    expect(await exists(path.join(dir, "glossary.md"))).toBe(true);

    const index = await fs.readFile(path.join(dir, ".hindsight-fs", "index.md"), "utf8");
    expect(index).toContain("**Policies/**");
    expect(index).toContain("policies/billing.md");
  });

  it("does not rewrite unchanged files but updates changed ones", async () => {
    setKb([page("prefs", "Preferences")], { prefs: "v1\n" });
    await runSync(config());
    const firstStat = await fs.stat(path.join(dir, "preferences.md"));

    setKb([page("prefs", "Preferences")], { prefs: "v1\n" });
    const second = await runSync(config());
    expect(second.unchanged).toBe(1);
    expect(second.written).toBe(0);
    expect((await fs.stat(path.join(dir, "preferences.md"))).mtimeMs).toBe(firstStat.mtimeMs);

    setKb([page("prefs", "Preferences")], { prefs: "v2\n" });
    const third = await runSync(config());
    expect(third.written).toBe(1);
    expect((await fs.readFile(path.join(dir, "preferences.md"), "utf8")).trim()).toBe("v2");
  });

  it("prunes a removed page within a folder, then prunes the folder when emptied", async () => {
    setKb([folder("f", "Stuff", [page("a", "A", "f"), page("b", "B", "f")])], {
      a: "a\n",
      b: "b\n",
    });
    await runSync(config());
    expect(await exists(path.join(dir, "stuff", "b.md"))).toBe(true);

    // Remove b but keep a in the folder → exactly one file pruned; folder stays.
    setKb([folder("f", "Stuff", [page("a", "A", "f")])], { a: "a\n" });
    const r2 = await runSync(config());
    expect(r2.removed).toBe(1);
    expect(await exists(path.join(dir, "stuff", "b.md"))).toBe(false);
    expect(await exists(path.join(dir, "stuff", "a.md"))).toBe(true);

    // Remove the folder entirely (move a to the root) → the folder dir is pruned.
    setKb([page("a", "A")], { a: "a\n" });
    await runSync(config());
    expect(await exists(path.join(dir, "stuff"))).toBe(false);
    expect(await exists(path.join(dir, "a.md"))).toBe(true);
  });

  it("does not wipe the mirror on a fetch error", async () => {
    setKb([page("a", "A")], { a: "a\n" });
    await runSync(config());

    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({}),
        text: () => Promise.resolve("boom"),
      })
    );
    await expect(runSync(config())).rejects.toThrow();
    expect(await exists(path.join(dir, "a.md"))).toBe(true);
  });

  it("recreates a file the user deleted even if the page is unchanged", async () => {
    setKb([page("a", "A")], { a: "a\n" });
    await runSync(config());
    await fs.unlink(path.join(dir, "a.md"));

    setKb([page("a", "A")], { a: "a\n" });
    const result = await runSync(config());
    expect(result.written).toBe(1);
    expect(await exists(path.join(dir, "a.md"))).toBe(true);
  });

  it("writes read-only files by default and editable files with writable", async () => {
    setKb([page("a", "A")], { a: "a\n" });
    await runSync(config());
    expect((await fs.stat(path.join(dir, "a.md"))).mode & 0o222).toBe(0);

    setKb([page("a", "A")], { a: "a\n" });
    await runSync(config({ writable: true }));
    expect((await fs.stat(path.join(dir, "a.md"))).mode & 0o200).toBe(0o200);
  });

  it("reverts a tampered file even when the page is unchanged server-side", async () => {
    setKb([page("a", "A")], { a: "original\n" });
    await runSync(config());

    const file = path.join(dir, "a.md");
    await fs.chmod(file, 0o644);
    await fs.writeFile(file, "HIJACKED", "utf8");

    setKb([page("a", "A")], { a: "original\n" });
    const result = await runSync(config());
    expect(result.reverted).toBe(1);
    expect(result.written).toBe(1);
    expect((await fs.readFile(file, "utf8")).trim()).toBe("original");
    expect((await fs.stat(file)).mode & 0o222).toBe(0);
  });
});

async function exists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}
