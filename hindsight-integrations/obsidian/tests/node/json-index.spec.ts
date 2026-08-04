import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SyncIndex } from "../../src/sync";
import { defaultIndexPath, loadIndex, makePersist } from "../../src/node/json-index";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "hs-idx-"));
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

const INDEX: SyncIndex = { "a.md": { hash: "sha256:1", mtime: 5, syncedAt: "T0" } };

describe("json-index", () => {
  it("returns an empty index when the file does not exist", async () => {
    expect(await loadIndex(join(dir, "missing.json"))).toEqual({});
  });

  it("round-trips an index through persist + load", async () => {
    const path = join(dir, "sub", "idx.json"); // parent created by persist
    await makePersist(path, () => "T1")(INDEX);
    expect(await loadIndex(path)).toEqual(INDEX);
  });

  it("stamps lastSyncAt and nests the index under syncIndex", async () => {
    const path = join(dir, "idx.json");
    await makePersist(path, () => "2026-08-04T00:00:00.000Z")(INDEX);
    const raw = JSON.parse(await readFile(path, "utf8"));
    expect(raw).toEqual({ syncIndex: INDEX, lastSyncAt: "2026-08-04T00:00:00.000Z" });
  });

  it("writes atomically (no leftover .tmp file)", async () => {
    const path = join(dir, "idx.json");
    await makePersist(path, () => "T1")(INDEX);
    const entries = await readdir(dir);
    expect(entries).toContain("idx.json");
    expect(entries.some((e) => e.endsWith(".tmp"))).toBe(false);
  });

  it("treats a corrupt index as empty and warns", async () => {
    const path = join(dir, "corrupt.json");
    await writeFile(path, "{ not json");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await loadIndex(path)).toEqual({});
    expect(warn).toHaveBeenCalledOnce();
  });

  it("defaultIndexPath sanitizes the vault name and lives under ~/.hindsight/obsidian", () => {
    const p = defaultIndexPath("My Vault/2026");
    expect(p).toMatch(/[/\\]\.hindsight[/\\]obsidian[/\\]My_Vault_2026\.json$/);
  });
});
