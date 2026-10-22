import { beforeEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "../src/client";
import { SyncEngine, type SyncConfig, type SyncIndex, type SyncVault } from "../src/sync";

interface FileSpec {
  content: string;
  mtime: number;
  ctime: number;
}

function fakeClient() {
  return {
    retain: vi.fn(async (_bank: string, _docId: string, _content: string, _opts?: unknown) => {}),
    deleteDocument: vi.fn(async (_bank: string, _docId: string) => {}),
  };
}

function fakeVault(files: Record<string, FileSpec>): SyncVault {
  return {
    getMarkdownFiles: () =>
      Object.keys(files).map((path) => ({
        path,
        stat: { mtime: files[path].mtime, ctime: files[path].ctime },
      })),
    read: async (file) => files[file.path].content,
  };
}

const BASE_CONFIG: SyncConfig = {
  bankId: "bank",
  includeFolders: [],
  excludeFolders: [],
  vaultName: "Vault",
  prefixDocId: false,
};

function makeEngine(
  files: Record<string, FileSpec>,
  index: SyncIndex = {},
  config: Partial<SyncConfig> = {}
) {
  const client = fakeClient();
  const persist = vi.fn(async () => {});
  const engine = new SyncEngine(
    client as unknown as HindsightClient,
    fakeVault(files),
    { ...BASE_CONFIG, ...config },
    index,
    persist,
    () => "T0"
  );
  return { client, engine, index, persist, vault: fakeVault(files) };
}

describe("SyncEngine", () => {
  beforeEach(() => vi.clearAllMocks());

  it("creates a document on first ingest, then skips when unchanged", async () => {
    const files = { "a.md": { content: "# A\nhello world", mtime: 1, ctime: 0 } };
    const { engine, client, vault } = makeEngine(files);
    const file = vault.getMarkdownFiles()[0];

    expect(await engine.ingestFile(file)).toBe("created");
    expect(client.retain).toHaveBeenCalledTimes(1);
    expect(client.retain).toHaveBeenCalledWith(
      "bank",
      "a.md",
      "# A\nhello world",
      expect.objectContaining({ updateMode: "replace" })
    );

    // Same mtime → skipped without a read or retain.
    expect(await engine.ingestFile(file)).toBe("skipped");
    expect(client.retain).toHaveBeenCalledTimes(1);
  });

  it("attaches auto-scope tags (vault, folder ancestors, date buckets) and vault-prefixed id", async () => {
    const created = Date.UTC(2026, 2, 15); // 2026-03
    const updated = Date.UTC(2026, 5, 20); // 2026-06
    const files = {
      "Work/Clients/acme.md": { content: "deal notes", mtime: updated, ctime: created },
    };
    const { engine, client } = makeEngine(files, {}, { vaultName: "Personal", prefixDocId: true });

    await engine.reconcile();

    const [, docId, , opts] = client.retain.mock.calls[0] as [
      string,
      string,
      string,
      { tags: string[]; metadata: Record<string, string> },
    ];
    expect(docId).toBe("Personal/Work/Clients/acme.md");
    expect(opts.tags).toEqual(
      expect.arrayContaining([
        "vault:Personal",
        "folder:Work",
        "folder:Work/Clients",
        "created:2026",
        "created:2026-03",
        "updated:2026",
        "updated:2026-06",
      ])
    );
    expect(opts.metadata.path).toBe("Work/Clients/acme.md");
    expect(opts.metadata.vault).toBe("Personal");
  });

  it("passes observation scope through without changing note provenance", async () => {
    const created = Date.UTC(2026, 2, 15);
    const updated = Date.UTC(2026, 5, 20);
    const files = {
      "Work/note.md": {
        content: "---\ntags: [planning]\nsource: meeting\n---\n# Decision\nkeep it simple",
        mtime: updated,
        ctime: created,
      },
    };
    const { engine, client } = makeEngine(files, {}, { observationScopes: "shared" });

    await engine.reconcile();

    const [, docId, content, opts] = client.retain.mock.calls[0] as [
      string,
      string,
      string,
      {
        tags: string[];
        metadata: Record<string, string>;
        timestamp: string;
        updateMode: string;
        observationScopes?: string;
      },
    ];
    expect(docId).toBe("Work/note.md");
    expect(content).toBe("# Decision\nkeep it simple");
    expect(opts).toMatchObject({
      observationScopes: "shared",
      updateMode: "replace",
      timestamp: new Date(created).toISOString(),
      metadata: { source: "meeting", vault: "Vault", path: "Work/note.md" },
    });
    expect(opts.tags).toEqual(
      expect.arrayContaining(["planning", "vault:Vault", "folder:Work", "created:2026-03"])
    );
  });

  it("omits observation scope from retain options when unset", async () => {
    const files = { "a.md": { content: "body", mtime: 1, ctime: 0 } };
    const { engine, client } = makeEngine(files);
    await engine.reconcile();
    const opts = client.retain.mock.calls[0][3] as Record<string, unknown>;
    expect(opts.observationScopes).toBeUndefined();
  });

  it("re-ingests (updated) when content changes", async () => {
    const index: SyncIndex = {};
    const filesV1 = { "a.md": { content: "v1", mtime: 1, ctime: 0 } };
    const { engine: e1, vault: v1vault } = makeEngine(filesV1, index);
    await e1.ingestFile(v1vault.getMarkdownFiles()[0]);

    const filesV2 = { "a.md": { content: "v2 changed", mtime: 2, ctime: 0 } };
    const client = fakeClient();
    const engine = new SyncEngine(
      client as unknown as HindsightClient,
      fakeVault(filesV2),
      BASE_CONFIG,
      index,
      vi.fn(async () => {}),
      () => "T1"
    );
    const file = fakeVault(filesV2).getMarkdownFiles()[0];
    expect(await engine.ingestFile(file)).toBe("updated");
    expect(client.retain).toHaveBeenCalledTimes(1);
  });

  it("hash-gate: skips re-ingest when mtime moved but content is identical", async () => {
    const index: SyncIndex = {};
    const v1 = { "a.md": { content: "same", mtime: 1, ctime: 0 } };
    const { engine: e1, vault: v1vault } = makeEngine(v1, index);
    await e1.ingestFile(v1vault.getMarkdownFiles()[0]);

    const v2 = { "a.md": { content: "same", mtime: 999, ctime: 0 } };
    const client = fakeClient();
    const engine = new SyncEngine(
      client as unknown as HindsightClient,
      fakeVault(v2),
      BASE_CONFIG,
      index,
      vi.fn(async () => {}),
      () => "T1"
    );
    expect(await engine.ingestFile(fakeVault(v2).getMarkdownFiles()[0])).toBe("skipped");
    expect(client.retain).not.toHaveBeenCalled();
  });

  it("deletes a document on note delete (only if previously synced)", async () => {
    const index: SyncIndex = { "a.md": { hash: "h", mtime: 1, syncedAt: "T0" } };
    const { engine, client } = makeEngine({}, index);

    await engine.handleDelete("a.md");
    expect(client.deleteDocument).toHaveBeenCalledWith("bank", "a.md");
    expect(index["a.md"]).toBeUndefined();

    // Unknown path → no-op.
    await engine.handleDelete("never-synced.md");
    expect(client.deleteDocument).toHaveBeenCalledTimes(1);
  });

  it("rename = delete old document + ingest new path", async () => {
    const index: SyncIndex = { "old.md": { hash: "h", mtime: 1, syncedAt: "T0" } };
    const files = { "new.md": { content: "moved", mtime: 2, ctime: 0 } };
    const { engine, client } = makeEngine(files, index);
    const file = fakeVault(files).getMarkdownFiles()[0];

    await engine.handleRename(file, "old.md");
    expect(client.deleteDocument).toHaveBeenCalledWith("bank", "old.md");
    expect(client.retain).toHaveBeenCalledWith(
      "bank",
      "new.md",
      "moved",
      expect.objectContaining({ updateMode: "replace" })
    );
  });

  it("reconcile ingests live notes and prunes orphaned documents", async () => {
    const index: SyncIndex = { "gone.md": { hash: "h", mtime: 1, syncedAt: "T0" } };
    const files = { "kept.md": { content: "kept", mtime: 1, ctime: 0 } };
    const { engine, client } = makeEngine(files, index);

    const summary = await engine.reconcile();
    expect(summary.added).toBe(1);
    expect(summary.deleted).toBe(1);
    expect(client.deleteDocument).toHaveBeenCalledWith("bank", "gone.md");
    expect(index["gone.md"]).toBeUndefined();
  });

  it("respects exclude folders and vault-prefixed document ids", async () => {
    const files = {
      "Private/secret.md": { content: "secret", mtime: 1, ctime: 0 },
      "Notes/keep.md": { content: "keep", mtime: 1, ctime: 0 },
    };
    const { engine, client } = makeEngine(
      files,
      {},
      {
        excludeFolders: ["Private"],
        prefixDocId: true,
      }
    );

    await engine.reconcile();
    const docIds = client.retain.mock.calls.map((c) => c[1]);
    expect(docIds).toContain("Vault/Notes/keep.md");
    expect(docIds).not.toContain("Vault/Private/secret.md");
  });

  it("dates facts by creation time by default", async () => {
    const created = Date.UTC(2026, 0, 11);
    const edited = Date.UTC(2026, 9, 1);
    const files = { "a.md": { content: "body", mtime: edited, ctime: created } };
    const { engine, client, vault } = makeEngine(files);
    await engine.ingestFile(vault.getMarkdownFiles()[0]);
    expect(client.retain.mock.calls[0][3]).toMatchObject({
      timestamp: new Date(created).toISOString(),
    });
  });

  it("timestampFrom=updated dates facts by the last edit (#5712)", async () => {
    const created = Date.UTC(2026, 0, 11);
    const edited = Date.UTC(2026, 9, 1);
    const files = {
      // frontmatter updated/modified wins over mtime; created is ignored.
      "fm.md": {
        content: "---\ncreated: 2026-01-11\nmodified: 2026-10-05\n---\nbody",
        mtime: edited,
        ctime: created,
      },
      // no frontmatter edit date → file mtime, even with a created date.
      "plain.md": { content: "---\ncreated: 2026-01-11\n---\nbody", mtime: edited, ctime: created },
    };
    const { engine, client } = makeEngine(files, {}, { timestampFrom: "updated" });
    await engine.reconcile();
    const byDoc = Object.fromEntries(
      client.retain.mock.calls.map((c) => [c[1], (c[3] as { timestamp: string }).timestamp])
    );
    expect(byDoc["fm.md"]).toBe("2026-10-05");
    expect(byDoc["plain.md"]).toBe(new Date(edited).toISOString());
  });
});

// Real client over a fake transport: the exact repro from #5067.
describe("SyncEngine cleanup when the server document is already gone (#5067)", () => {
  const gone = {
    status: 404,
    text: JSON.stringify({ detail: "Document not found" }),
    json: { detail: "Document not found" },
  };

  function engineWith(files: Record<string, FileSpec>, index: SyncIndex, status = gone) {
    const transport = vi.fn(async (req: { method: string }) =>
      req.method === "DELETE" ? status : { status: 200, text: "{}", json: {} }
    );
    const persist = vi.fn(async () => {});
    const client = new HindsightClient("https://h", undefined, transport);
    const engine = new SyncEngine(
      client,
      fakeVault(files),
      BASE_CONFIG,
      index,
      persist,
      () => "T0"
    );
    return { engine, index, persist, transport };
  }

  const stale = (): SyncIndex => ({ "Notes/old.md": { hash: "h", mtime: 1, syncedAt: "T" } });

  it("reconcile drops and persists the stale entry, then converges", async () => {
    const { engine, index, persist } = engineWith({}, stale());
    expect((await engine.reconcile()).deleted).toBe(1);
    expect(index).toEqual({});
    expect(persist).toHaveBeenCalled();
    expect((await engine.reconcile()).deleted).toBe(0);
  });

  it("handleDelete drops the stale entry", async () => {
    const { engine, index } = engineWith({}, stale());
    await engine.handleDelete("Notes/old.md");
    expect(index).toEqual({});
  });

  it("handleRename drops the old entry and still ingests the new path", async () => {
    const files = { "Notes/new.md": { content: "body", mtime: 2, ctime: 0 } };
    const { engine, index, transport } = engineWith(files, stale());
    await engine.handleRename(fakeVault(files).getMarkdownFiles()[0], "Notes/old.md");
    expect(Object.keys(index)).toEqual(["Notes/new.md"]);
    expect(transport.mock.calls.some(([r]) => r.method === "POST")).toBe(true);
  });

  it("a missing bank still fails and keeps the entry", async () => {
    const bankGone = {
      status: 404,
      text: JSON.stringify({ detail: "Bank 'bank' not found" }),
      json: { detail: "Bank 'bank' not found" },
    };
    const { engine, index } = engineWith({}, stale(), bankGone);
    await expect(engine.reconcile()).rejects.toThrow("HTTP 404");
    expect(Object.keys(index)).toEqual(["Notes/old.md"]);
  });
});
