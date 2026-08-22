/**
 * Load a .env file into process.env so the tool can reuse the Hindsight deployment's config
 * (LLM provider/model/key, API key). Resolution order:
 *   1. MISSION_SANDBOX_ENV_FILE if set (explicit path)
 *   2. the nearest .env walking up from the current working directory
 *
 * Uses Node's built-in loader, which does NOT overwrite already-set process.env values, so an
 * explicitly exported variable still wins over the file.
 */

import { existsSync } from "node:fs";
import path from "node:path";

export function loadProjectEnv(): string | null {
  const candidates: string[] = [];
  if (process.env.MISSION_SANDBOX_ENV_FILE) {
    candidates.push(path.resolve(process.env.MISSION_SANDBOX_ENV_FILE));
  }
  let dir = process.cwd();
  for (let i = 0; i < 8; i++) {
    candidates.push(path.join(dir, ".env"));
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  for (const file of candidates) {
    if (!existsSync(file)) continue;
    try {
      process.loadEnvFile(file);
      return file;
    } catch {
      // Unreadable/malformed — try the next candidate.
    }
  }
  return null;
}
