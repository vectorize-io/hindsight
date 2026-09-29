#!/usr/bin/env node
/**
 * hindsight-seed — MANUAL control command (escape hatch): `seed` forces a background backfill of a
 * repo's bank now; `decline` opts a repo out of auto-seeding. Normal seeding is automatic and
 * deterministic via the Claude SessionStart hook (core/session-start.ts) — this CLI is not part of
 * that automatic path; it's for forcing/opting-out by hand.
 *
 * Claude-Code-only for now: the harness is hardcoded to "claude-code" below, since this CLI is only
 * ever invoked from the Claude Code SessionStart flow (Task 10b). If it's ever driven from another
 * harness's context — especially one with a non-default `bankIdTemplate` or a `harnesses.<name>`
 * override in the config — the hardcoded harness would resolve the wrong bank. Generalizing would
 * need a real `--harness` flag, not added here.
 */
import { loadConfig } from "./core/config";
import { deriveBankIdOrSkip } from "./core/bank";
import { seedControl } from "./core/seed";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : undefined;
}

const command = process.argv[2] || "";
const repo = arg("repo") || process.cwd();
const cfg = loadConfig({ harness: "claude-code" });
const bankId = deriveBankIdOrSkip(cfg, repo, "claude-code");
if (bankId === null) {
  console.error(
    `seed: cannot identify the repository at ${repo} — refusing to seed a guessed bank`
  );
  process.exit(1);
}
const result = seedControl(command, { repo, bankId });
console.log(result.message);
process.exit(result.ok ? 0 : 1);
