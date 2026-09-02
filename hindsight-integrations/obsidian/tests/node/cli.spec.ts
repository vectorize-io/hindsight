import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  HelpRequested,
  UsageError,
  buildConfig,
  createWatchHandlers,
  parseCliArgs,
  runCli,
} from "../../src/node/cli";
import type { FsVault } from "../../src/node/fs-vault";
import type { SyncEngine, SyncFile } from "../../src/sync";

describe("parseCliArgs", () => {
  beforeEach(() => {
    vi.stubEnv("HINDSIGHT_API_URL", "");
    vi.stubEnv("HINDSIGHT_API_TOKEN", "");
  });
  afterEach(() => vi.unstubAllEnvs());

  const base = ["--vault", "/v", "--bank", "b", "--api-url", "https://h"];

  it("resolves required flags and sensible defaults", () => {
    const o = parseCliArgs(base);
    expect(o.bank).toBe("b");
    expect(o.apiUrl).toBe("https://h");
    expect(o.vault).toMatch(/[/\\]v$/); // resolved to absolute
    expect(o.vaultName).toBe("v"); // basename of the vault dir
    expect(o.include).toEqual([]);
    expect(o.exclude).toEqual([]);
    expect(o.prefixDocId).toBe(false);
    expect(o.watch).toBe(false);
    // Default index path is target-scoped: <vault>-<bank>-<fingerprint>.json.
    expect(o.indexPath).toMatch(/\.hindsight[/\\]obsidian[/\\]v-b-[0-9a-f]{12}\.json$/);
  });

  it("binds the index identity to the resolved destination (not scope)", () => {
    const o = parseCliArgs([...base, "--exclude", "Archive", "--prefix-doc-id"]);
    expect(o.identity).toEqual({
      apiOrigin: "https://h",
      bankId: "b",
      vaultPath: o.vault,
      vaultName: "v",
      prefixDocId: true,
    });
  });

  it("routes different banks to different default index files", () => {
    const a = parseCliArgs(["--vault", "/v", "--bank", "a", "--api-url", "https://h"]);
    const b = parseCliArgs(["--vault", "/v", "--bank", "b", "--api-url", "https://h"]);
    expect(a.indexPath).not.toBe(b.indexPath);
  });

  it("collects repeatable --include/--exclude and flags", () => {
    const o = parseCliArgs([
      ...base,
      "--include",
      "Work",
      "--include",
      "Personal",
      "--exclude",
      "Archive",
      "--prefix-doc-id",
      "--vault-name",
      "Brain",
      "--index",
      "/tmp/i.json",
    ]);
    expect(o.include).toEqual(["Work", "Personal"]);
    expect(o.exclude).toEqual(["Archive"]);
    expect(o.prefixDocId).toBe(true);
    expect(o.vaultName).toBe("Brain");
    expect(o.indexPath).toBe("/tmp/i.json");
  });

  it("falls back to env for api url/token", () => {
    vi.stubEnv("HINDSIGHT_API_URL", "https://env");
    vi.stubEnv("HINDSIGHT_API_TOKEN", "envtok");
    const o = parseCliArgs(["--vault", "/v", "--bank", "b"]);
    expect(o.apiUrl).toBe("https://env");
    expect(o.apiToken).toBe("envtok");
  });

  it("requires --vault, --bank and an api url", () => {
    expect(() => parseCliArgs(["--bank", "b", "--api-url", "h"])).toThrow(UsageError);
    expect(() => parseCliArgs(["--vault", "/v", "--api-url", "h"])).toThrow(UsageError);
    expect(() => parseCliArgs(["--vault", "/v", "--bank", "b"])).toThrow(/api-url/);
  });

  it("rejects an unknown subcommand and surfaces --help", () => {
    expect(() => parseCliArgs(["frobnicate", ...base])).toThrow(UsageError);
    expect(() => parseCliArgs(["--help"])).toThrow(HelpRequested);
  });
});

describe("buildConfig", () => {
  it("maps options onto a SyncConfig", () => {
    const cfg = buildConfig(
      parseCliArgs(["--vault", "/v", "--bank", "b", "--api-url", "h", "--prefix-doc-id"])
    );
    expect(cfg).toEqual({
      bankId: "b",
      includeFolders: [],
      excludeFolders: [],
      vaultName: "v",
      prefixDocId: true,
    });
  });
});

describe("createWatchHandlers", () => {
  it("upserts a live file and deletes on unlink", async () => {
    const ingestFile = vi.fn(async () => "created" as const);
    const handleDelete = vi.fn(async () => {});
    const engine = { ingestFile, handleDelete } as unknown as SyncEngine;
    const file: SyncFile = { path: "a.md", stat: { mtime: 1, ctime: 0 } };
    const vault = {
      syncFileFor: (p: string) => (p === "a.md" ? file : null),
    } as unknown as FsVault;

    const h = createWatchHandlers(engine, vault);
    await h.onUpsert("a.md");
    await h.onUpsert("gone.md"); // vanished before we could stat it → no-op
    await h.onUnlink("a.md");

    expect(ingestFile).toHaveBeenCalledExactlyOnceWith(file, { force: true });
    expect(handleDelete).toHaveBeenCalledExactlyOnceWith("a.md");
  });
});

describe("runCli", () => {
  let root: string;
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "hs-cli-"));
  });
  afterEach(async () => {
    await rm(root, { recursive: true, force: true });
    vi.unstubAllGlobals();
  });

  it("reconciles a real vault end-to-end (fetch mocked), printing a summary", async () => {
    await mkdir(join(root, "Work"), { recursive: true });
    await writeFile(join(root, "Work", "note.md"), "# Note\nremember this");

    const fetchSpy = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(
      async () => ({ status: 200, text: async () => "{}" }) as unknown as Response
    );
    vi.stubGlobal("fetch", fetchSpy);
    const out: string[] = [];

    const code = await runCli(
      ["--vault", root, "--bank", "b", "--api-url", "https://h", "--index", join(root, "idx.json")],
      (m) => out.push(m)
    );

    expect(code).toBe(0);
    expect(out.join("\n")).toMatch(/reconcile: \+1 added/);
    // One retain POST to the memories endpoint.
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("https://h/v1/default/banks/b/memories");
    expect(init?.method).toBe("POST");
  });

  it("prints help and exits 0", async () => {
    const out: string[] = [];
    expect(await runCli(["--help"], (m) => out.push(m))).toBe(0);
    expect(out.join("\n")).toMatch(/Usage:/);
  });

  it("returns exit code 2 on a usage error", async () => {
    const err: string[] = [];
    expect(
      await runCli(
        ["--bank", "b"],
        () => {},
        (m) => err.push(m)
      )
    ).toBe(2);
    expect(err.join("\n")).toMatch(/--vault is required/);
  });

  it("threads --exclude through to the engine (excluded folder never POSTed)", async () => {
    await mkdir(join(root, "Keep"), { recursive: true });
    await mkdir(join(root, "Archive"), { recursive: true });
    await writeFile(join(root, "Keep", "a.md"), "keep");
    await writeFile(join(root, "Archive", "b.md"), "archive");

    const posted: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (body.items) posted.push(body.items[0].document_id);
        return { status: 200, text: async () => "{}" } as unknown as Response;
      })
    );

    const code = await runCli(
      [
        "--vault",
        root,
        "--bank",
        "b",
        "--api-url",
        "https://h",
        "--exclude",
        "Archive",
        "--index",
        join(root, "i.json"),
      ],
      () => {}
    );
    expect(code).toBe(0);
    expect(posted).toEqual(["Keep/a.md"]);
  });

  it("refuses to reuse one --index file across two banks (issue #3257)", async () => {
    await writeFile(join(root, "note.md"), "# Note\nremember this");
    const posts: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (init?.method === "POST") posts.push(url);
        return { status: 200, text: async () => "{}" } as unknown as Response;
      })
    );
    const index = join(root, "shared.json");
    const argsFor = (bank: string) => [
      "--vault",
      root,
      "--bank",
      bank,
      "--api-url",
      "https://h",
      "--index",
      index,
    ];

    // First target populates the shared index.
    expect(await runCli(argsFor("bank-a"), () => {})).toBe(0);
    expect(posts).toEqual(["https://h/v1/default/banks/bank-a/memories"]);

    // Reusing it against bank-b must fail closed — never silently skip note.md,
    // and never retain against bank-b off the back of bank-a's index.
    const err: string[] = [];
    const code = await runCli(
      argsFor("bank-b"),
      () => {},
      (m) => err.push(m)
    );
    expect(code).toBe(1);
    expect(err.join("\n")).toMatch(/different target|bankId/);
    // No POST landed on bank-b.
    expect(posts).toEqual(["https://h/v1/default/banks/bank-a/memories"]);
  });

  it("refuses a stale --index whose scope changed, preventing cross-target deletes", async () => {
    await writeFile(join(root, "note.md"), "# Note\nkeep");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ status: 200, text: async () => "{}" }) as unknown as Response)
    );
    const index = join(root, "shared.json");
    const at = (url: string) => [
      "--vault",
      root,
      "--bank",
      "b",
      "--api-url",
      url,
      "--index",
      index,
    ];

    expect(await runCli(at("https://a"), () => {})).toBe(0);

    // Same bank + vault, different API origin → the index no longer owns the target.
    const err: string[] = [];
    expect(
      await runCli(
        at("https://b"),
        () => {},
        (m) => err.push(m)
      )
    ).toBe(1);
    expect(err.join("\n")).toMatch(/apiOrigin|different target/);
  });
});
