import { existsSync, statSync } from "node:fs";

/** Whether `p` exists and is a regular file (mirrors Python's `Path.is_file()`). */
export function isFile(p) {
  try {
    return existsSync(p) && statSync(p).isFile();
  } catch {
    return false;
  }
}
