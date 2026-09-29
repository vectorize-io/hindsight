#!/usr/bin/env node
/**
 * Keep devDependencies out of the published package.json.
 *
 * OpenClaw installs a plugin by running a full `npm install` inside the
 * extracted package directory, so the plugin's own devDependencies are
 * resolved on the user's machine — unlike an ordinary npm dependency, where
 * they are ignored. That makes our test toolchain part of every user's
 * install graph, and npm 10's arborist crashes resolving vitest's optional
 * peer set ("Cannot read properties of null (reading 'edgesOut')") now that
 * vitest 5 is published. `--omit=dev` does not help: arborist builds the full
 * ideal tree before pruning. Shipping a manifest with no devDependencies does.
 *
 * Runs as prepack (strip) / postpack (restore) so the working tree is only
 * ever modified for the duration of `npm pack` / `npm publish`.
 */
import { copyFileSync, existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const packageDir = dirname(dirname(fileURLToPath(import.meta.url)));
const manifestPath = join(packageDir, "package.json");
const backupPath = join(packageDir, "package.json.pack-backup");

/** Fields that only matter while developing this package. */
const DEV_ONLY_FIELDS = ["devDependencies", "overrides"];

function strip() {
  if (existsSync(backupPath)) {
    throw new Error(
      `${backupPath} already exists — a previous pack did not restore package.json. ` +
        `Restore it manually before packing again.`
    );
  }
  const original = readFileSync(manifestPath, "utf8");
  const manifest = JSON.parse(original);
  const removed = DEV_ONLY_FIELDS.filter((field) => field in manifest);
  if (removed.length === 0) return;
  copyFileSync(manifestPath, backupPath);
  for (const field of removed) delete manifest[field];
  // Trailing newline keeps the file prettier-clean if a restore ever fails.
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`[pack-manifest] stripped from published manifest: ${removed.join(", ")}`);
}

function restore() {
  if (!existsSync(backupPath)) return;
  copyFileSync(backupPath, manifestPath);
  rmSync(backupPath);
  console.log("[pack-manifest] restored package.json");
}

const mode = process.argv[2];
if (mode === "strip") strip();
else if (mode === "restore") restore();
else {
  console.error("usage: pack-manifest.mjs strip|restore");
  process.exit(1);
}
