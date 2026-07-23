#!/usr/bin/env node
/**
 * hindsight-coding-backfill — one-shot setup + ingest of a repo's history into a Hindsight bank,
 * for the reflect-only coding-agents plugin.
 *
 * It (1) configures the bank (reflect mission, observations on, `git`+`chat` retain strategies),
 * (2) ingests EVERY git commit (full message + full diff) under the `git` strategy, (3) reads the
 * chosen HARNESS's past sessions and ingests each RAW as a JSON transcript under the `chat` strategy,
 * and (4) synthesizes generic codebase knowledge pages. Every item carries a REF-ID tracer.
 *
 * The only harness-specific step is (3): --harness selects how past sessions are read. Git ingest,
 * strategies, missions, and pages are identical across agents.
 *
 * Shared connection/bank settings (api-url, api-token, bank, harness) come from the JSON config
 * ~/.hindsight/coding-agent.json — the SAME file the runtime plugin reads; the matching --flags below
 * override it. Operation flags (--repo, --conversations, --limit, --reset, --no-pages, --concurrency)
 * are CLI-only.
 *
 * Usage:
 *   hindsight-coding-backfill --repo <path> [--bank <id>] [--harness opencode] \
 *       [--conversations <sessions.json>] [--api-url http://localhost:8888] [--api-token X] \
 *       [--config <path>] [--limit N] [--reset] [--no-pages] [--concurrency 8]
 */
import { deriveBankId } from "./core/bank";
import { loadConfig, CONFIG_PATH } from "./core/config";
import { HindsightClient } from "./core/hindsight";
import { ingestGit } from "./core/git";
import { ingestChats } from "./core/chat";
import { getHarness, HARNESS_NAMES } from "./harness/registry";

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  return process.argv.includes(`--${name}`) ? "true" : def;
}

// Shared settings: config file provides the base; the matching --flag overrides per-run.
const cfg = loadConfig(arg("config") ?? CONFIG_PATH);
const REPO = arg("repo");
// --bank wins; else the SAME per-repo resolution the runtime uses on --repo, so
// `hindsight-coding-backfill --repo .` fills exactly the bank the agents will read.
const BANK =
  arg("bank") ?? (REPO ? deriveBankId(cfg, REPO, arg("harness") ?? cfg.harness) : cfg.bankId);
const HARNESS = arg("harness") ?? cfg.harness;
const API_URL = arg("api-url") ?? cfg.apiUrl;
const API_TOKEN = arg("api-token") ?? cfg.apiToken;
// Operation-only flags (never in config):
const CONV = arg("conversations");
const LIMIT = arg("limit") ? Number(arg("limit")) : undefined;
const RESET = process.argv.includes("--reset");
const NO_PAGES = process.argv.includes("--no-pages");
const CONCURRENCY = Number(arg("concurrency", "8"));

if (!REPO || !BANK) {
  console.error(
    "usage: hindsight-coding-backfill --repo <path> [--bank <id>] [--harness <name>] " +
      "[--conversations f.json] [--api-url U] [--config path] [--limit N] [--reset] [--no-pages] [--concurrency N]\n" +
      "shared settings default from ~/.hindsight/coding-agent.json; --bank is required there or via --bank.\n" +
      `harnesses: ${HARNESS_NAMES.join(", ")}`
  );
  process.exit(1);
}

const log = (m: string) => console.log(m);

async function main() {
  const harness = getHarness(HARNESS); // throws with the available list on an unknown name
  const client = new HindsightClient({ apiUrl: API_URL, apiToken: API_TOKEN, bank: BANK!, log });
  console.log(`hindsight-coding-backfill -> ${client.apiUrl} bank=${BANK} harness=${harness.name}`);

  await client.configureBank({ reset: RESET });

  // chats FIRST: they're few and carry the decisions that make memory necessary — ingesting them
  // before the (large) git flood keeps them from being starved in the server's extraction queue.
  const sessions = await harness.chatReader.read({ conversations: CONV, repo: REPO });
  const chatFails = await ingestChats(client, sessions, { concurrency: CONCURRENCY, log });
  const gitFails = await ingestGit(client, REPO!, { limit: LIMIT, concurrency: CONCURRENCY, log });

  await client.drain(client.opIds, "extraction");

  // knowledge pages are synthesized from the extracted facts, so create them AFTER the drain.
  if (NO_PAGES) console.log("[pages] skipped (--no-pages)");
  else await client.createPages();

  const failures = chatFails + gitFails;
  console.log(
    `\n✅ backfill complete${failures ? ` (${failures} items failed to enqueue)` : ""}. ` +
      `Point the plugin at this bank by setting "bankId": "${BANK}" in ~/.hindsight/coding-agent.json.`
  );
}

main().catch((e) => {
  console.error("backfill failed:", e.message || e);
  process.exit(1);
});
