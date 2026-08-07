/** Filesystem helper: load .txt/.md documents from a file or directory into memory. */

import { promises as fs } from "node:fs";
import path from "node:path";

/** A document loaded from disk at apply-time (documents are not stored in the project). */
export interface LoadedDocument {
  name: string;
  content: string;
}

export async function collectDocuments(target: string): Promise<LoadedDocument[]> {
  const resolved = path.resolve(target);
  const stat = await fs.stat(resolved).catch(() => {
    throw new Error(`Path not found: ${target}`);
  });

  const files: string[] = [];
  if (stat.isFile()) {
    files.push(resolved);
  } else if (stat.isDirectory()) {
    const walk = async (dir: string): Promise<void> => {
      for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) await walk(full);
        else if (/\.(txt|md)$/i.test(entry.name)) files.push(full);
      }
    };
    await walk(resolved);
  }

  files.sort();
  return Promise.all(
    files.map(async (f) => ({ name: path.basename(f), content: await fs.readFile(f, "utf8") }))
  );
}
