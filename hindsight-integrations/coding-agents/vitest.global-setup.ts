import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * One fresh temp dir per test run, removed when it ends. Retain cursors are temp-dir files for
 * every harness, persistent plugins included (#6559): this keeps tests from writing into the
 * developer's real one, and one run's cursors from turning the next run's first write-back into an
 * append. Set here, before the workers start, so every worker inherits it.
 */
export default function setup(): () => void {
  const dir = mkdtempSync(join(tmpdir(), "hs-coding-agents-test-"));
  // tmpdir() reads TMPDIR on POSIX and TEMP/TMP on Windows.
  process.env.TMPDIR = process.env.TMP = process.env.TEMP = dir;
  return () => rmSync(dir, { recursive: true, force: true });
}
