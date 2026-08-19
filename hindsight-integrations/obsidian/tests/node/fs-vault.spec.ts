import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FsVault } from "../../src/node/fs-vault";

let root: string;

async function write(rel: string, content: string): Promise<void> {
  const abs = join(root, rel);
  await mkdir(join(abs, ".."), { recursive: true });
  await writeFile(abs, content);
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "hs-vault-"));
});
afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

describe("FsVault", () => {
  it("lists only markdown files, recursively, with POSIX-relative paths", async () => {
    await write("root.md", "a");
    await write("Work/Clients/acme.md", "b");
    await write("Work/notes.txt", "not markdown");
    await write("image.png", "binary");

    const paths = new FsVault(root)
      .getMarkdownFiles()
      .map((f) => f.path)
      .sort();

    expect(paths).toEqual(["Work/Clients/acme.md", "root.md"]);
  });

  it("skips dotfolders like .obsidian, .trash and .git", async () => {
    await write("keep.md", "a");
    await write(".obsidian/plugins/x/data.md", "config");
    await write(".trash/deleted.md", "trash");
    await write(".git/notes.md", "git");

    const paths = new FsVault(root).getMarkdownFiles().map((f) => f.path);
    expect(paths).toEqual(["keep.md"]);
  });

  it("populates stat.mtime and stat.ctime as epoch milliseconds", async () => {
    await write("a.md", "hello");
    const [file] = new FsVault(root).getMarkdownFiles();
    expect(file.stat.mtime).toBeGreaterThan(0);
    expect(file.stat.ctime).toBeGreaterThan(0);
    // millisecond scale, not seconds
    expect(file.stat.mtime).toBeGreaterThan(1_000_000_000_000);
  });

  it("reads file contents", async () => {
    await write("Notes/deep.md", "# Title\nbody");
    const vault = new FsVault(root);
    const file = vault.getMarkdownFiles().find((f) => f.path === "Notes/deep.md")!;
    expect(await vault.read(file)).toBe("# Title\nbody");
  });

  it("syncFileFor returns a SyncFile for a live path and null for a missing one", async () => {
    await write("here.md", "x");
    const vault = new FsVault(root);
    expect(vault.syncFileFor("here.md")?.path).toBe("here.md");
    expect(vault.syncFileFor("gone.md")).toBeNull();
  });

  it("handles unicode and spaces in names", async () => {
    await write("Área/notações café.md", "x");
    const paths = new FsVault(root).getMarkdownFiles().map((f) => f.path);
    expect(paths).toEqual(["Área/notações café.md"]);
  });
});
