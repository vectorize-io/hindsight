import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SyncIndex } from "../../src/sync";
import {
  canonicalApiOrigin,
  defaultIndexPath,
  identityFingerprint,
  type IndexIdentity,
  IndexIdentityError,
  loadIndex,
  makePersist,
} from "../../src/node/json-index";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "hs-idx-"));
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

const INDEX: SyncIndex = { "a.md": { hash: "sha256:1", mtime: 5, syncedAt: "T0" } };

const IDENTITY: IndexIdentity = {
  apiOrigin: "http://127.0.0.1:8888",
  bankId: "bank-a",
  vaultPath: "/tmp/test-vault",
  vaultName: "test-vault",
  prefixDocId: false,
};

/** IDENTITY with one field overridden, for mismatch tests. */
const withField = (patch: Partial<IndexIdentity>): IndexIdentity => ({ ...IDENTITY, ...patch });

describe("json-index", () => {
  it("returns an empty index when the file does not exist", async () => {
    expect(await loadIndex(join(dir, "missing.json"), IDENTITY)).toEqual({});
  });

  it("round-trips an index through persist + load", async () => {
    const path = join(dir, "sub", "idx.json"); // parent created by persist
    await makePersist(path, IDENTITY, () => "T1")(INDEX);
    expect(await loadIndex(path, IDENTITY)).toEqual(INDEX);
  });

  it("stamps lastSyncAt + identity and nests the index under syncIndex", async () => {
    const path = join(dir, "idx.json");
    await makePersist(path, IDENTITY, () => "2026-08-04T00:00:00.000Z")(INDEX);
    const raw = JSON.parse(await readFile(path, "utf8"));
    expect(raw).toEqual({
      version: 1,
      identity: IDENTITY,
      syncIndex: INDEX,
      lastSyncAt: "2026-08-04T00:00:00.000Z",
    });
  });

  it("writes atomically (no leftover .tmp file)", async () => {
    const path = join(dir, "idx.json");
    await makePersist(path, IDENTITY, () => "T1")(INDEX);
    const entries = await readdir(dir);
    expect(entries).toContain("idx.json");
    expect(entries.some((e) => e.endsWith(".tmp"))).toBe(false);
  });

  it("treats a corrupt index as empty and warns", async () => {
    const path = join(dir, "corrupt.json");
    await writeFile(path, "{ not json");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await loadIndex(path, IDENTITY)).toEqual({});
    expect(warn).toHaveBeenCalledOnce();
  });

  describe("target binding (issue #3257)", () => {
    it("reuses the index when the identity is unchanged", async () => {
      const path = join(dir, "idx.json");
      await makePersist(path, IDENTITY, () => "T1")(INDEX);
      expect(await loadIndex(path, IDENTITY)).toEqual(INDEX);
    });

    it.each<[string, IndexIdentity]>([
      ["bank", withField({ bankId: "bank-b" })],
      ["API origin", withField({ apiOrigin: "http://other:8888" })],
      ["vault path", withField({ vaultPath: "/tmp/other-vault" })],
      ["vault name", withField({ vaultName: "renamed" })],
      ["prefixDocId", withField({ prefixDocId: true })],
    ])("refuses to reuse an index bound to a different %s", async (_label, other) => {
      const path = join(dir, "idx.json");
      await makePersist(path, IDENTITY, () => "T1")(INDEX);
      await expect(loadIndex(path, other)).rejects.toBeInstanceOf(IndexIdentityError);
    });

    it("reuses the index across an include/exclude change (scope is not bound)", async () => {
      // Same destination, narrowed scope → the engine legitimately reuses the
      // index (and would prune newly-excluded docs it owns on that bank).
      const path = join(dir, "idx.json");
      await makePersist(path, IDENTITY, () => "T1")(INDEX);
      expect(await loadIndex(path, IDENTITY)).toEqual(INDEX);
    });

    it("names the changed field in the refusal message", async () => {
      const path = join(dir, "idx.json");
      await makePersist(path, IDENTITY, () => "T1")(INDEX);
      await expect(loadIndex(path, withField({ bankId: "bank-b" }))).rejects.toThrow(
        /bankId.*bank-a.*bank-b/s
      );
    });

    it("refuses a legacy index that has no identity metadata", async () => {
      const path = join(dir, "legacy.json");
      // The pre-3257 envelope: syncIndex + lastSyncAt, no identity.
      await writeFile(path, JSON.stringify({ syncIndex: INDEX, lastSyncAt: "T0" }));
      await expect(loadIndex(path, IDENTITY)).rejects.toThrow(/predates target binding/);
    });
  });

  describe("defaultIndexPath", () => {
    it("lives under ~/.hindsight/obsidian and encodes vault + bank + fingerprint", () => {
      const p = defaultIndexPath(withField({ vaultName: "My Vault/2026", bankId: "bank a" }));
      expect(p).toMatch(
        /[/\\]\.hindsight[/\\]obsidian[/\\]My_Vault_2026-bank_a-[0-9a-f]{12}\.json$/
      );
    });

    it("gives different targets different default paths (same vault name)", () => {
      const a = defaultIndexPath(withField({ bankId: "bank-a" }));
      const b = defaultIndexPath(withField({ bankId: "bank-b" }));
      expect(a).not.toBe(b);
    });

    it("is stable for the same destination regardless of scope", () => {
      // Scope isn't part of the identity, so it can't shift the default path.
      const p1 = defaultIndexPath(IDENTITY);
      const p2 = defaultIndexPath({ ...IDENTITY });
      expect(identityFingerprint(IDENTITY)).toBe(identityFingerprint({ ...IDENTITY }));
      expect(p1).toBe(p2);
    });
  });

  describe("canonicalApiOrigin", () => {
    it("drops path, query, and credentials", () => {
      expect(canonicalApiOrigin("https://user:pw@host:8888/v1/default?x=1")).toBe(
        "https://host:8888"
      );
    });

    it("normalizes trailing slashes on an unparseable value", () => {
      expect(canonicalApiOrigin("not a url///")).toBe("not a url");
    });
  });
});
