/**
 * Full-stack reconcile tests: FsVault (real temp dir) + JSON index + the shared
 * SyncEngine, with a fake client capturing the API calls. Exercises the same
 * behaviors the plugin relies on (create/update/skip/delete/rename/exclude/
 * prefix/prune-ownership) plus a parity check that the filesystem frontend
 * produces byte-identical requests to an in-memory (plugin-style) vault.
 */

import { mkdir, rm, mkdtemp, rename, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "../../src/client";
import { FsVault } from "../../src/node/fs-vault";
import { type IndexIdentity, loadIndex, makePersist } from "../../src/node/json-index";
import {
  SyncEngine,
  type SyncConfig,
  type SyncFile,
  type SyncIndex,
  type SyncVault,
} from "../../src/sync";

let root: string;
let indexPath: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "hs-recon-"));
  indexPath = join(root, ".idx.json");
});
afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

type RetainFn = (
  bank: string,
  docId: string,
  content: string,
  opts: { tags: string[]; metadata: Record<string, string> }
) => Promise<void>;

function fakeClient() {
  return {
    retain: vi.fn<RetainFn>(async () => {}),
    deleteDocument: vi.fn<(bank: string, docId: string) => Promise<void>>(async () => {}),
  };
}

async function writeNote(rel: string, content: string): Promise<void> {
  const abs = join(root, rel);
  await mkdir(join(abs, ".."), { recursive: true });
  await writeFile(abs, content);
}

async function makeEngine(config: Partial<SyncConfig> = {}) {
  const client = fakeClient();
  const cfg: SyncConfig = {
    bankId: "bank",
    includeFolders: [],
    excludeFolders: [],
    vaultName: "Vault",
    prefixDocId: false,
    ...config,
  };
  // Destination identity for the index (scope is intentionally not bound, so
  // narrowing include/exclude across reconciles reuses the same index).
  const identity: IndexIdentity = {
    apiOrigin: "http://test",
    bankId: cfg.bankId,
    vaultPath: root,
    vaultName: cfg.vaultName,
    prefixDocId: cfg.prefixDocId,
  };
  const index = await loadIndex(indexPath, identity);
  const engine = new SyncEngine(
    client as unknown as HindsightClient,
    new FsVault(root),
    cfg,
    index,
    makePersist(indexPath, identity, () => "T0"),
    () => "T0"
  );
  return { client, engine };
}

describe("reconcile over a filesystem vault", () => {
  it("creates a document with scope tags + path metadata, then skips unchanged", async () => {
    await writeNote("Work/Clients/acme.md", "# Acme\nrenewal in Q3");

    const first = await makeEngine();
    const summary = await first.engine.reconcile();
    expect(summary).toMatchObject({ added: 1, updated: 0, deleted: 0 });

    const [bank, docId, body, opts] = first.client.retain.mock.calls[0];
    expect(bank).toBe("bank");
    expect(docId).toBe("Work/Clients/acme.md");
    expect(body).toContain("renewal in Q3");
    expect(opts.tags).toEqual(
      expect.arrayContaining(["vault:Vault", "folder:Work", "folder:Work/Clients"])
    );
    expect(opts.tags.some((t) => /^created:\d{4}$/.test(t))).toBe(true);
    expect(opts.metadata.path).toBe("Work/Clients/acme.md");

    // A second engine (fresh, but loading the persisted index) must skip.
    const second = await makeEngine();
    await second.engine.reconcile();
    expect(second.client.retain).not.toHaveBeenCalled();
  });

  it("re-ingests when content changes but skips an mtime-only touch", async () => {
    await writeNote("a.md", "original");
    await (await makeEngine()).engine.reconcile();

    // Content change → update.
    await writeNote("a.md", "edited body");
    const edit = await makeEngine();
    expect((await edit.engine.reconcile()).updated).toBe(1);
    expect(edit.client.retain).toHaveBeenCalledOnce();

    // mtime moves but content (hash) is identical → skip, no re-ingest.
    const future = new Date(Date.now() + 60_000);
    await utimes(join(root, "a.md"), future, future);
    const touch = await makeEngine();
    const summary = await touch.engine.reconcile();
    expect(summary).toMatchObject({ added: 0, updated: 0, unchanged: 1 });
    expect(touch.client.retain).not.toHaveBeenCalled();
  });

  it("prunes a document when its note is deleted from disk", async () => {
    await writeNote("gone.md", "temporary");
    await (await makeEngine()).engine.reconcile();

    await rm(join(root, "gone.md"));
    const after = await makeEngine();
    const summary = await after.engine.reconcile();
    expect(summary.deleted).toBe(1);
    expect(after.client.deleteDocument).toHaveBeenCalledWith("bank", "gone.md");
  });

  it("handles a rename as delete-old + create-new", async () => {
    await writeNote("old.md", "movable content");
    await (await makeEngine()).engine.reconcile();

    await rename(join(root, "old.md"), join(root, "new.md"));
    const after = await makeEngine();
    const summary = await after.engine.reconcile();
    expect(summary).toMatchObject({ added: 1, deleted: 1 });
    expect(after.client.deleteDocument).toHaveBeenCalledWith("bank", "old.md");
    const created = after.client.retain.mock.calls.map((c) => c[1]);
    expect(created).toContain("new.md");
  });

  it("excludes a folder and prunes notes that become excluded", async () => {
    await writeNote("Keep/a.md", "keep me");
    await writeNote("Archive/b.md", "archive me");

    // First sync with no excludes → both ingested.
    const initial = await makeEngine();
    expect((await initial.engine.reconcile()).added).toBe(2);

    // Re-sync excluding Archive → its previously-synced doc is pruned.
    const excluded = await makeEngine({ excludeFolders: ["Archive"] });
    const summary = await excluded.engine.reconcile();
    expect(summary.deleted).toBe(1);
    expect(excluded.client.deleteDocument).toHaveBeenCalledWith("bank", "Archive/b.md");
    expect(excluded.client.retain).not.toHaveBeenCalled(); // Keep/a.md unchanged
  });

  it("prefixes document ids with the vault name when configured", async () => {
    await writeNote("note.md", "prefixed");
    const { client, engine } = await makeEngine({ prefixDocId: true, vaultName: "Brain" });
    await engine.reconcile();
    expect(client.retain.mock.calls[0][1]).toBe("Brain/note.md");
  });

  it("only prunes documents tracked in its own index (prune ownership)", async () => {
    await writeNote("mine.md", "owned");
    await (await makeEngine()).engine.reconcile();

    // Delete the owned note; the index also names a doc we never had on disk —
    // reconcile must prune only what the index tracks, nothing invented.
    await rm(join(root, "mine.md"));
    const after = await makeEngine();
    await after.engine.reconcile();
    expect(after.client.deleteDocument).toHaveBeenCalledExactlyOnceWith("bank", "mine.md");
  });

  it("carries frontmatter tags and uses the created date as the timestamp", async () => {
    await writeNote(
      "Projects/x.md",
      "---\ntags: [alpha, beta]\ncreated: 2025-11-02\n---\n# X\nthe body"
    );
    const { client, engine } = await makeEngine();
    await engine.reconcile();

    const [, , , opts] = client.retain.mock.calls[0] as unknown as [
      string,
      string,
      string,
      { tags: string[]; metadata: Record<string, string>; timestamp?: string },
    ];
    // Frontmatter tags flow through alongside the auto-scope tags.
    expect(opts.tags).toEqual(
      expect.arrayContaining(["alpha", "beta", "vault:Vault", "folder:Projects"])
    );
    // `created:` in frontmatter wins as the document timestamp.
    expect(opts.timestamp).toContain("2025-11-02");
    expect(opts.metadata.vault).toBe("Vault");
  });

  it("skips a note whose body is empty after frontmatter (nothing to ground on)", async () => {
    await writeNote("empty.md", "---\ntags: [x]\n---\n   \n");
    const { client, engine } = await makeEngine();
    const summary = await engine.reconcile();
    expect(client.retain).not.toHaveBeenCalled();
    expect(summary.added).toBe(0);
  });

  it("with includeFolders set, syncs only the included folder", async () => {
    await writeNote("Work/keep.md", "included");
    await writeNote("Personal/skip.md", "excluded");
    const { client, engine } = await makeEngine({ includeFolders: ["Work"] });
    await engine.reconcile();

    const docIds = client.retain.mock.calls.map((c) => c[1]);
    expect(docIds).toEqual(["Work/keep.md"]);
  });
});

// An in-memory vault mirroring how the Obsidian plugin feeds the same engine.
function memoryVault(
  files: Record<string, { content: string; mtime: number; ctime: number }>
): SyncVault {
  return {
    getMarkdownFiles: (): SyncFile[] =>
      Object.keys(files).map((path) => ({
        path,
        stat: { mtime: files[path].mtime, ctime: files[path].ctime },
      })),
    read: async (f) => files[f.path].content,
  };
}

describe("dual-ingester parity", () => {
  it("produces identical retain requests from the filesystem and the plugin vault", async () => {
    // Both notes carry a `created:` date in frontmatter so the created-date tags
    // come from the note, not from filesystem ctime/birthtime — utimes can only
    // set mtime, and birthtime behaves differently across OSes (macOS clamps it
    // to a past mtime, Linux does not), which would otherwise make this compare
    // platform-dependent.
    const noteA = "---\ntags: [work]\ncreated: 2026-03-15\n---\n# A\nalpha";
    const noteB = "---\ncreated: 2026-03-15\n---\n# B\nbeta";
    await writeNote("Work/a.md", noteA);
    await writeNote("b.md", noteB);

    const config: SyncConfig = {
      bankId: "bank",
      includeFolders: [],
      excludeFolders: [],
      vaultName: "Vault",
      prefixDocId: false,
    };

    // Filesystem frontend (CLI) — pin mtimes so timestamps match the memory vault.
    const fixed = new Date("2026-03-15T00:00:00Z");
    await utimes(join(root, "Work/a.md"), fixed, fixed);
    await utimes(join(root, "b.md"), fixed, fixed);
    const fsClient = fakeClient();
    const fsIndex: SyncIndex = {};
    await new SyncEngine(
      fsClient as unknown as HindsightClient,
      new FsVault(root),
      config,
      fsIndex,
      async () => {},
      () => "T0"
    ).reconcile();

    // Plugin-style in-memory frontend, same files/mtimes/config.
    const memClient = fakeClient();
    await new SyncEngine(
      memClient as unknown as HindsightClient,
      memoryVault({
        "Work/a.md": { content: noteA, mtime: fixed.getTime(), ctime: fixed.getTime() },
        "b.md": { content: noteB, mtime: fixed.getTime(), ctime: fixed.getTime() },
      }),
      config,
      {},
      async () => {},
      () => "T0"
    ).reconcile();

    const normalize = (calls: unknown[][]) =>
      calls
        .map((c) => ({
          docId: c[1],
          body: c[2],
          tags: [...(c[3] as { tags: string[] }).tags].sort(),
          meta: c[3],
        }))
        .sort((x, y) => String(x.docId).localeCompare(String(y.docId)));

    expect(normalize(fsClient.retain.mock.calls)).toEqual(normalize(memClient.retain.mock.calls));
  });
});
