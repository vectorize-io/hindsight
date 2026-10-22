import { execFileSync } from "node:child_process";
import { copyFile, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";

const PLUGIN_DIR = join(__dirname, "..", "..");
let dir: string;

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

// The release script bumps package.json and then runs exactly this `npm version`; the
// package's `version` hook must carry the bump into the files Obsidian reads (#5969).
it("npm version bumps manifest.json and records the version in versions.json", async () => {
  dir = await mkdtemp(join(tmpdir(), "obsidian-version-"));
  for (const f of ["package.json", "manifest.json", "versions.json", "version-bump.mjs"]) {
    await copyFile(join(PLUGIN_DIR, f), join(dir, f));
  }
  const before = JSON.parse(await readFile(join(dir, "versions.json"), "utf8"));

  execFileSync("npm", ["version", "9.9.9", "--no-git-tag-version", "--allow-same-version"], {
    cwd: dir,
    stdio: "ignore",
  });

  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8"));
  const versions = JSON.parse(await readFile(join(dir, "versions.json"), "utf8"));
  expect(manifest.version).toBe("9.9.9");
  expect(versions).toEqual({ ...before, "9.9.9": manifest.minAppVersion });
});
