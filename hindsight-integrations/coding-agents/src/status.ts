#!/usr/bin/env node
/**
 * status — print the bank's sync status as one JSON line (see core/status.ts). Harness plumbing,
 * not a user CLI: the benchmark and e2e suites poll this until `synced` before measuring; users
 * ask the agent via the `hindsight_sync_status` tool instead.
 *
 * usage: node status.js --repo <path> [--bank <id>] [--harness <name>] [--api-url U] [--api-token X] [--config path]
 */
import { deriveBankIdOrSkip } from "./core/bank";
import { applyBankConfig, loadConfig } from "./core/config";
import { HindsightClient } from "./core/hindsight";
import { syncStatus } from "./core/status";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  return undefined;
}

const REPO = arg("repo");
const cfg0 = loadConfig({ harness: arg("harness") ?? undefined, path: arg("config") });
const BANK =
  arg("bank") ??
  (REPO ? deriveBankIdOrSkip(cfg0, REPO, arg("harness") ?? cfg0.harness) : cfg0.bankId) ??
  undefined;
if (!BANK) {
  console.error("usage: node status.js --repo <path> [--bank <id>] [--api-url U]");
  process.exit(1);
}

const resolved0 = BANK ? applyBankConfig(cfg0, BANK) : { cfg: cfg0, bankId: BANK };
const cfg = resolved0.cfg;
const FINAL_BANK = resolved0.bankId ?? BANK;
const client = new HindsightClient({
  apiUrl: arg("api-url") ?? cfg.apiUrl,
  apiToken: arg("api-token") ?? cfg.apiToken,
  bank: FINAL_BANK!,
  maxParallelRetains: cfg.maxParallelRetains,
  observationScopes: cfg.observationScopes,
});

syncStatus(client, FINAL_BANK!, REPO)
  .then((s) => console.log(JSON.stringify(s)))
  .catch((e) => {
    console.error("status failed:", (e as Error).message || e);
    process.exit(1);
  });
