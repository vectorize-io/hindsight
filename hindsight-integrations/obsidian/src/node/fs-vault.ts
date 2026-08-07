/**
 * Filesystem {@link SyncVault}: enumerates a vault's Markdown files from disk so
 * the shared {@link SyncEngine} can run headless, with no Obsidian runtime.
 *
 * Semantics are kept identical to Obsidian's `Vault`:
 * - paths are vault-relative and POSIX-separated (so document ids/tags match the
 *   plugin's regardless of host OS);
 * - `stat.mtime`/`stat.ctime` are epoch milliseconds, matching `TFile.stat`;
 * - dotfolders (`.obsidian`, `.trash`, `.git`, …) are skipped — they hold config
 *   and history, never vault notes. Note-level include/exclude scoping is the
 *   engine's job (`SyncConfig.includeFolders`/`excludeFolders`), not the vault's.
 */

import { readdirSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { SyncFile, SyncVault } from "../sync";

export class FsVault implements SyncVault {
  constructor(private readonly root: string) {}

  getMarkdownFiles(): SyncFile[] {
    const out: SyncFile[] = [];
    this.walk("", out);
    return out;
  }

  private walk(rel: string, out: SyncFile[]): void {
    const abs = rel ? join(this.root, rel) : this.root;
    for (const entry of readdirSync(abs, { withFileTypes: true })) {
      // Skip dotfiles/dotfolders (.obsidian, .trash, .git, …) — config, not notes.
      if (entry.name.startsWith(".")) continue;
      const childRel = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        this.walk(childRel, out);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
        const file = this.syncFileFor(childRel);
        if (file) out.push(file);
      }
    }
  }

  /**
   * Build a {@link SyncFile} for a single vault-relative path, or null if it is
   * gone. Used by watch mode to turn an fs event into an ingest.
   */
  syncFileFor(path: string): SyncFile | null {
    try {
      const st = statSync(join(this.root, ...path.split("/")));
      // birthtime is unreliable on some Linux filesystems (reported as 0); fall
      // back to ctime (inode change) so created-date tags still get a value.
      return { path, stat: { mtime: st.mtimeMs, ctime: st.birthtimeMs || st.ctimeMs } };
    } catch {
      return null;
    }
  }

  async read(file: SyncFile): Promise<string> {
    return readFile(join(this.root, ...file.path.split("/")), "utf8");
  }
}
