import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { runSync } from "../src/sync.js";
import { parseDocument } from "../src/frontmatter.js";
import type { MentalModel } from "../src/client.js";
import type { MountConfig } from "../src/config.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function listResponse(items: MentalModel[]) {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({ items }),
    text: () => Promise.resolve(JSON.stringify({ items })),
  });
}

let dir: string;

function config(overrides: Partial<MountConfig> = {}): MountConfig {
  return {
    dir,
    apiUrl: "http://localhost:8000",
    bankId: "acme",
    intervalSeconds: 30,
    detail: "content",
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
  it("writes a markdown file per mental model", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([
        {
          id: "prefs",
          bank_id: "acme",
          name: "Preferences",
          content: "Likes dark mode.",
          tags: ["ui"],
        },
        { id: "comms", bank_id: "acme", name: "Comms", content: "Async first." },
      ])
    );

    const result = await runSync(config());
    expect(result.total).toBe(2);
    expect(result.written).toBe(2);

    const prefs = await fs.readFile(path.join(dir, "prefs.md"), "utf8");
    const parsed = parseDocument(prefs);
    expect(parsed.frontmatter.id).toBe("prefs");
    expect(parsed.frontmatter.tags).toEqual(["ui"]);
    expect(parsed.body.trim()).toBe("Likes dark mode.");

    // index written under control dir
    const index = await fs.readFile(path.join(dir, ".hindsight-fs", "index.md"), "utf8");
    expect(index).toContain("prefs.md");
  });

  it("does not rewrite unchanged files but updates changed ones", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "prefs", bank_id: "acme", name: "Preferences", content: "v1" }])
    );
    await runSync(config());
    const firstStat = await fs.stat(path.join(dir, "prefs.md"));

    // Same content again → unchanged, file not rewritten.
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "prefs", bank_id: "acme", name: "Preferences", content: "v1" }])
    );
    const second = await runSync(config());
    expect(second.unchanged).toBe(1);
    expect(second.written).toBe(0);
    const secondStat = await fs.stat(path.join(dir, "prefs.md"));
    expect(secondStat.mtimeMs).toBe(firstStat.mtimeMs);

    // Changed content → rewritten.
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "prefs", bank_id: "acme", name: "Preferences", content: "v2" }])
    );
    const third = await runSync(config());
    expect(third.written).toBe(1);
    const body = parseDocument(await fs.readFile(path.join(dir, "prefs.md"), "utf8")).body;
    expect(body.trim()).toBe("v2");
  });

  it("prunes files for deleted models", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([
        { id: "a", bank_id: "acme", name: "A", content: "a" },
        { id: "b", bank_id: "acme", name: "B", content: "b" },
      ])
    );
    await runSync(config());
    expect(await exists(path.join(dir, "b.md"))).toBe(true);

    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "a" }])
    );
    const result = await runSync(config());
    expect(result.removed).toBe(1);
    expect(await exists(path.join(dir, "b.md"))).toBe(false);
    expect(await exists(path.join(dir, "a.md"))).toBe(true);
  });

  it("does not wipe the mirror on a fetch error", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "a" }])
    );
    await runSync(config());

    mockFetch.mockReturnValueOnce(
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

  it("recreates a file the user deleted even if the model is unchanged", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "a" }])
    );
    await runSync(config());
    await fs.unlink(path.join(dir, "a.md"));

    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "a" }])
    );
    const result = await runSync(config());
    expect(result.written).toBe(1);
    expect(await exists(path.join(dir, "a.md"))).toBe(true);
  });

  it("writes read-only files by default and editable files with writable", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "a" }])
    );
    await runSync(config());
    const ro = await fs.stat(path.join(dir, "a.md"));
    expect(ro.mode & 0o222).toBe(0); // no write bits

    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "a" }])
    );
    await runSync(config({ writable: true }));
    const rw = await fs.stat(path.join(dir, "a.md"));
    expect(rw.mode & 0o200).toBe(0o200); // owner write bit set
  });

  it("reverts a tampered file even when the model is unchanged server-side", async () => {
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "original" }])
    );
    await runSync(config());

    // Simulate an agent forcing the file writable and editing it.
    const file = path.join(dir, "a.md");
    await fs.chmod(file, 0o644);
    await fs.writeFile(file, "HIJACKED", "utf8");

    // Same model content from the API → must still be reverted from disk.
    mockFetch.mockReturnValueOnce(
      listResponse([{ id: "a", bank_id: "acme", name: "A", content: "original" }])
    );
    const result = await runSync(config());
    expect(result.reverted).toBe(1);
    expect(result.written).toBe(1);
    const body = parseDocument(await fs.readFile(file, "utf8")).body;
    expect(body.trim()).toBe("original");
    const stat = await fs.stat(file);
    expect(stat.mode & 0o222).toBe(0); // back to read-only
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
