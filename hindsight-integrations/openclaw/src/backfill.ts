#!/usr/bin/env node
import { existsSync, readFileSync, realpathSync } from "fs";
import { dirname, join, resolve } from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { HindsightServer } from "@vectorize-io/hindsight-all";
import { HindsightClient } from "@vectorize-io/hindsight-client";

function loadPackageVersion(): string {
  try {
    const pkgPath = join(dirname(fileURLToPath(import.meta.url)), "..", "package.json");
    const pkg = JSON.parse(readFileSync(pkgPath, "utf8")) as { version?: string };
    return pkg.version ?? "0.0.0";
  } catch {
    return "0.0.0";
  }
}

const USER_AGENT = `hindsight-openclaw/${loadPackageVersion()}`;
import { detectExternalApi, detectLLMConfig } from "./index.js";
import type { BankStats, PluginConfig } from "./types.js";
import {
  buildBackfillPlan,
  checkpointKey,
  defaultCheckpointPath,
  defaultOpenClawRoot,
  loadCheckpoint,
  loadPluginConfigFromOpenClawRoot,
  saveCheckpoint,
  type BackfillCheckpoint,
  type BackfillPlanEntry,
  type BackfillCliOptions,
} from "./backfill-lib.js";

interface ParsedArgs {
  openclawRoot: string;
  profile: string;
  agents: string[];
  includeArchive: boolean;
  limit?: number;
  dryRun: boolean;
  json: boolean;
  resume: boolean;
  checkpointPath: string;
  bankStrategy: "mirror-config" | "agent" | "fixed";
  fixedBank?: string;
  apiUrl?: string;
  apiToken?: string;
  maxPendingOperations?: number;
  waitUntilDrained: boolean;
}

interface BackfillRuntime {
  apiUrl: string;
  apiToken?: string;
  stop(): Promise<void>;
}

interface BankRuntime {
  bankId: string;
  touchedEntryKeys: string[];
  initialFailedOperations: number;
  missionApplied: boolean;
}

function usage(): string {
  return [
    "Usage: hindsight-openclaw-backfill [options]",
    "",
    "Options:",
    "  --openclaw-root <path>        OpenClaw root directory (default: ~/.openclaw)",
    "  --profile <name>              Logical profile name for reporting (default: openclaw)",
    "  --agent <id>                  Restrict import to a specific agent (repeatable)",
    "  --include-archive             Include migration archives (default)",
    "  --exclude-archive             Exclude migration archives",
    "  --limit <n>                   Stop after enqueueing N sessions",
    "  --dry-run                     Build and print the import plan without enqueueing",
    "  --json                        Print final summary as JSON",
    "  --resume                      Skip entries already marked completed in the checkpoint",
    "  --checkpoint <path>           Path to checkpoint JSON",
    "  --bank-strategy <mode>        mirror-config | agent | fixed",
    "  --fixed-bank <id>             Required when bank strategy is fixed",
    "  --api-url <url>               Hindsight API base URL override",
    "  --api-token <token>           Hindsight API bearer token override",
    "  --max-pending-operations <n>  Wait until target bank queue is <= n before enqueueing",
    "  --wait-until-drained          Wait for touched banks to drain and finalize checkpoint state",
    "  -h, --help                    Show this help",
  ].join("\n");
}

function parseArgs(argv: string[]): ParsedArgs {
  const args: ParsedArgs = {
    openclawRoot: defaultOpenClawRoot(),
    profile: "openclaw",
    agents: [],
    includeArchive: true,
    dryRun: false,
    json: false,
    resume: false,
    checkpointPath: "",
    bankStrategy: "mirror-config",
    waitUntilDrained: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const value = argv[++i];
      if (!value) {
        throw new Error(`missing value for ${arg}`);
      }
      return value;
    };

    switch (arg) {
      case "--openclaw-root":
        args.openclawRoot = resolve(next());
        break;
      case "--profile":
        args.profile = next();
        break;
      case "--agent":
        args.agents.push(next());
        break;
      case "--include-archive":
        args.includeArchive = true;
        break;
      case "--exclude-archive":
        args.includeArchive = false;
        break;
      case "--limit":
        args.limit = Number(next());
        break;
      case "--dry-run":
        args.dryRun = true;
        break;
      case "--json":
        args.json = true;
        break;
      case "--resume":
        args.resume = true;
        break;
      case "--checkpoint":
        args.checkpointPath = resolve(next());
        break;
      case "--bank-strategy": {
        const value = next();
        if (value !== "mirror-config" && value !== "agent" && value !== "fixed") {
          throw new Error(`invalid bank strategy: ${value}`);
        }
        args.bankStrategy = value;
        break;
      }
      case "--fixed-bank":
        args.fixedBank = next();
        break;
      case "--api-url":
        args.apiUrl = next();
        break;
      case "--api-token":
        args.apiToken = next();
        break;
      case "--max-pending-operations":
        args.maxPendingOperations = Number(next());
        break;
      case "--wait-until-drained":
        args.waitUntilDrained = true;
        break;
      case "-h":
      case "--help":
        console.log(usage());
        process.exit(0);
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }

  if (!args.checkpointPath) {
    args.checkpointPath = defaultCheckpointPath(args.openclawRoot);
  }
  if (args.bankStrategy === "fixed" && !args.fixedBank) {
    throw new Error("--fixed-bank is required when --bank-strategy fixed is used");
  }
  return args;
}

function inferApiSettings(
  pluginConfig: PluginConfig,
  explicitApiUrl?: string,
  explicitApiToken?: string
): { apiUrl: string; apiToken?: string } {
  const apiUrl =
    explicitApiUrl ||
    pluginConfig.hindsightApiUrl ||
    `http://127.0.0.1:${pluginConfig.apiPort || 9077}`;
  const apiToken = explicitApiToken || pluginConfig.hindsightApiToken;
  return { apiUrl, apiToken: apiToken || undefined };
}

async function checkHealth(apiUrl: string, apiToken?: string): Promise<boolean> {
  try {
    const headers: Record<string, string> = { "User-Agent": USER_AGENT };
    if (apiToken) headers.Authorization = `Bearer ${apiToken}`;
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/health`, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(5000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export function filterEntriesForResume(
  entries: BackfillPlanEntry[],
  checkpoint: BackfillCheckpoint,
  resume: boolean
): BackfillPlanEntry[] {
  if (!resume) {
    return entries;
  }
  return entries.filter(
    (entry) => checkpoint.entries[checkpointKey(entry)]?.status !== "completed"
  );
}

export function splitResumeEntries(
  entries: BackfillPlanEntry[],
  checkpoint: BackfillCheckpoint,
  waitUntilDrained: boolean
): { entriesToEnqueue: BackfillPlanEntry[]; alreadyEnqueuedKeys: string[] } {
  const entriesToEnqueue: BackfillPlanEntry[] = [];
  const alreadyEnqueuedKeys: string[] = [];
  for (const entry of entries) {
    const status = checkpoint.entries[checkpointKey(entry)]?.status;
    if (status === "enqueued") {
      if (waitUntilDrained) {
        alreadyEnqueuedKeys.push(checkpointKey(entry));
      } else {
        entriesToEnqueue.push(entry);
      }
      continue;
    }
    entriesToEnqueue.push(entry);
  }
  return { entriesToEnqueue, alreadyEnqueuedKeys };
}

export function applyDrainResults(
  checkpoint: BackfillCheckpoint,
  touchedEntriesByBank: Map<string, string[]>,
  finalStatsByBank: Map<string, BankStats>,
  initialFailedOperationsByBank: Map<string, number>
): { completed: number; unresolved: number; warnings: string[] } {
  let completed = 0;
  let unresolved = 0;
  const warnings: string[] = [];

  for (const [bankId, entryKeys] of touchedEntriesByBank.entries()) {
    const stats = finalStatsByBank.get(bankId);
    const initialFailed = initialFailedOperationsByBank.get(bankId) ?? 0;
    const hasNewFailures = !!stats && stats.failed_operations > initialFailed;

    if (hasNewFailures) {
      warnings.push(
        `bank ${bankId} reported ${stats!.failed_operations - initialFailed} new failed operations during drain; leaving ${entryKeys.length} checkpoint entries enqueued`
      );
    } else if (!stats || stats.pending_operations > 0) {
      warnings.push(
        `bank ${bankId} did not finish draining cleanly; leaving ${entryKeys.length} checkpoint entries enqueued`
      );
    }

    for (const entryKey of entryKeys) {
      const existing = checkpoint.entries[entryKey];
      if (!existing || existing.status !== "enqueued") {
        continue;
      }
      if (!hasNewFailures && stats && stats.pending_operations === 0) {
        checkpoint.entries[entryKey] = {
          ...existing,
          status: "completed",
          updatedAt: new Date().toISOString(),
          error: undefined,
        };
        completed += 1;
      } else {
        unresolved += 1;
      }
    }
  }

  return { completed, unresolved, warnings };
}

export async function createBackfillRuntime(
  pluginConfig: PluginConfig,
  explicitApiUrl?: string,
  explicitApiToken?: string
): Promise<BackfillRuntime> {
  const explicit = inferApiSettings(pluginConfig, explicitApiUrl, explicitApiToken);
  const externalApi = detectExternalApi(pluginConfig);
  const useExternalApi = !!(
    explicitApiUrl ||
    explicitApiToken ||
    externalApi.apiUrl ||
    pluginConfig.hindsightApiUrl
  );

  if (useExternalApi) {
    return {
      apiUrl: explicit.apiUrl,
      apiToken: explicit.apiToken,
      async stop() {},
    };
  }

  if (await checkHealth(explicit.apiUrl, explicit.apiToken)) {
    return {
      apiUrl: explicit.apiUrl,
      apiToken: explicit.apiToken,
      async stop() {},
    };
  }

  const llmConfig = detectLLMConfig(pluginConfig);
  const manager = new HindsightServer({
    profile: "openclaw",
    port: pluginConfig.apiPort || 9077,
    embedVersion: pluginConfig.embedVersion,
    embedPackagePath: pluginConfig.embedPackagePath,
    env: {
      HINDSIGHT_API_LLM_PROVIDER: llmConfig.provider || "",
      HINDSIGHT_API_LLM_API_KEY: llmConfig.apiKey || "",
      HINDSIGHT_API_LLM_MODEL: llmConfig.model,
      HINDSIGHT_API_LLM_BASE_URL: llmConfig.baseUrl,
      HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: String(pluginConfig.daemonIdleTimeout ?? 0),
    },
  });
  await manager.start();

  return {
    apiUrl: manager.getBaseUrl(),
    apiToken: undefined,
    async stop() {
      await manager.stop();
    },
  };
}

/**
 * Fetch stats for a single bank over HTTP. The high-level `HindsightClient`
 * doesn't yet wrap this endpoint, so we go direct — it's one call.
 */
async function fetchBankStats(
  baseUrl: string,
  apiToken: string | undefined,
  bankId: string
): Promise<BankStats> {
  const headers: Record<string, string> = { "User-Agent": USER_AGENT };
  if (apiToken) headers.Authorization = `Bearer ${apiToken}`;
  const res = await fetch(`${baseUrl}/v1/default/banks/${encodeURIComponent(bankId)}/stats`, {
    headers,
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text().catch(() => "")}`);
  }
  return res.json() as Promise<BankStats>;
}

async function waitForBankQueue(
  apiUrl: string,
  apiToken: string | undefined,
  bankId: string,
  maxPendingOperations: number
): Promise<void> {
  for (;;) {
    try {
      const stats = await fetchBankStats(apiUrl, apiToken, bankId);
      if (stats.pending_operations <= maxPendingOperations) {
        return;
      }
    } catch (error) {
      if (error instanceof Error && error.message.includes("HTTP 404")) {
        return;
      }
      throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

async function getInitialBankStats(
  apiUrl: string,
  apiToken: string | undefined,
  bankId: string
): Promise<BankStats | null> {
  try {
    return await fetchBankStats(apiUrl, apiToken, bankId);
  } catch (error) {
    if (error instanceof Error && error.message.includes("HTTP 404")) {
      return null;
    }
    throw error;
  }
}

async function waitForBanksToDrain(
  apiUrl: string,
  apiToken: string | undefined,
  bankIds: Iterable<string>
): Promise<Map<string, BankStats>> {
  const ids = Array.from(bankIds);
  for (;;) {
    const stats = await Promise.all(
      ids.map(async (bankId) => ({ bankId, stats: await fetchBankStats(apiUrl, apiToken, bankId) }))
    );
    const statsByBank = new Map(stats.map(({ bankId, stats: bankStats }) => [bankId, bankStats]));
    const pending = stats.filter(({ stats: bankStats }) => bankStats.pending_operations > 0);
    if (pending.length === 0) {
      return statsByBank;
    }
    console.log(
      pending
        .map(
          ({ bankId, stats: bankStats }) =>
            `${bankId}\tpending_operations=${bankStats.pending_operations}\tfailed_operations=${bankStats.failed_operations}\tpending_consolidation=${bankStats.pending_consolidation}`
        )
        .join("\n")
    );
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
}

export async function runCli(argv: string[] = process.argv.slice(2)): Promise<void> {
  const args = parseArgs(argv);
  if (!existsSync(join(args.openclawRoot, "openclaw.json"))) {
    throw new Error(`could not find openclaw.json under ${args.openclawRoot}`);
  }

  const pluginConfig = loadPluginConfigFromOpenClawRoot(args.openclawRoot);
  const backfillOptions: BackfillCliOptions = {
    openclawRoot: args.openclawRoot,
    includeArchive: args.includeArchive,
    selectedAgents: args.agents.length ? new Set(args.agents) : undefined,
    limit: args.limit,
    bankStrategy: args.bankStrategy,
    fixedBank: args.fixedBank,
  };
  const checkpoint = loadCheckpoint(args.checkpointPath);
  const { entries, discoveredSessions, skippedEmpty } = buildBackfillPlan(
    pluginConfig,
    backfillOptions
  );
  const plannedEntries = filterEntriesForResume(entries, checkpoint, args.resume);
  const { entriesToEnqueue, alreadyEnqueuedKeys } = splitResumeEntries(
    plannedEntries,
    checkpoint,
    args.waitUntilDrained
  );

  if (args.dryRun) {
    for (const entry of plannedEntries) {
      console.log(
        `${entry.agentId}\t${entry.bankId}\t${entry.sessionId}\tmsgs=${entry.messageCount}\tchars=${entry.transcript.length}`
      );
    }
    const summary = {
      profile: args.profile,
      dry_run: true,
      discovered_sessions: discoveredSessions,
      planned_sessions: plannedEntries.length,
      skipped_empty: skippedEmpty,
      bank_strategy: args.bankStrategy,
      checkpoint_path: args.checkpointPath,
    };
    console.log(args.json ? JSON.stringify(summary, null, 2) : JSON.stringify(summary));
    return;
  }

  const runtime = await createBackfillRuntime(pluginConfig, args.apiUrl, args.apiToken);
  // Single shared client — hindsight-client takes bankId as a parameter on
  // every call, so there's no reason to cache per-bank clients anymore.
  const client = new HindsightClient({ baseUrl: runtime.apiUrl, apiKey: runtime.apiToken });
  const bankRuntimes = new Map<string, BankRuntime>();
  let imported = 0;
  let failed = 0;
  let finalized = 0;

  try {
    for (const entryKey of alreadyEnqueuedKeys) {
      const checkpointEntry = checkpoint.entries[entryKey];
      if (!checkpointEntry) continue;
      let bankRuntime = bankRuntimes.get(checkpointEntry.bankId);
      if (!bankRuntime) {
        bankRuntime = {
          bankId: checkpointEntry.bankId,
          touchedEntryKeys: [],
          initialFailedOperations:
            (await getInitialBankStats(runtime.apiUrl, runtime.apiToken, checkpointEntry.bankId))
              ?.failed_operations ?? 0,
          missionApplied: false,
        };
        bankRuntimes.set(checkpointEntry.bankId, bankRuntime);
      }
      bankRuntime.touchedEntryKeys.push(entryKey);
    }

    for (const entry of entriesToEnqueue) {
      let bankRuntime = bankRuntimes.get(entry.bankId);
      if (!bankRuntime) {
        bankRuntime = {
          bankId: entry.bankId,
          touchedEntryKeys: [],
          initialFailedOperations:
            (await getInitialBankStats(runtime.apiUrl, runtime.apiToken, entry.bankId))
              ?.failed_operations ?? 0,
          missionApplied: false,
        };
        bankRuntimes.set(entry.bankId, bankRuntime);
      }

      if (!bankRuntime.missionApplied && pluginConfig.bankMission) {
        await client.createBank(entry.bankId, { reflectMission: pluginConfig.bankMission });
      }

      if (typeof args.maxPendingOperations === "number" && args.maxPendingOperations >= 0) {
        await waitForBankQueue(
          runtime.apiUrl,
          runtime.apiToken,
          entry.bankId,
          args.maxPendingOperations
        );
      }

      try {
        const metadata: Record<string, string> = {
          source: "openclaw-backfill",
          file_path: entry.filePath,
          agent_id: entry.agentId,
          session_id: entry.sessionId,
          retained_at: new Date().toISOString(),
        };
        if (entry.startedAt) {
          metadata.session_started_at = entry.startedAt;
        }
        await client.retain(entry.bankId, entry.transcript, {
          documentId: entry.documentId,
          metadata,
          async: true,
        });
        checkpoint.entries[checkpointKey(entry)] = {
          status: "enqueued",
          bankId: entry.bankId,
          filePath: entry.filePath,
          sessionId: entry.sessionId,
          updatedAt: new Date().toISOString(),
        };
        bankRuntime.touchedEntryKeys.push(checkpointKey(entry));
        if (!bankRuntime.missionApplied && pluginConfig.bankMission) {
          await client.createBank(entry.bankId, { reflectMission: pluginConfig.bankMission });
          bankRuntime.missionApplied = true;
        }
        saveCheckpoint(args.checkpointPath, checkpoint);
        console.log(`${entry.agentId}\t${entry.bankId}\t${entry.sessionId}\tenqueued`);
        imported += 1;
      } catch (error) {
        checkpoint.entries[checkpointKey(entry)] = {
          status: "failed",
          bankId: entry.bankId,
          filePath: entry.filePath,
          sessionId: entry.sessionId,
          updatedAt: new Date().toISOString(),
          error: error instanceof Error ? error.message : String(error),
        };
        saveCheckpoint(args.checkpointPath, checkpoint);
        failed += 1;
        console.error(
          `${entry.agentId}\t${entry.bankId}\t${entry.sessionId}\tfailed\t${error instanceof Error ? error.message : String(error)}`
        );
      }
    }

    if (args.waitUntilDrained && bankRuntimes.size > 0) {
      const finalStatsByBank = await waitForBanksToDrain(
        runtime.apiUrl,
        runtime.apiToken,
        bankRuntimes.keys()
      );
      const touchedEntriesByBank = new Map(
        Array.from(bankRuntimes.entries()).map(([bankId, value]) => [
          bankId,
          value.touchedEntryKeys,
        ])
      );
      const initialFailedByBank = new Map(
        Array.from(bankRuntimes.entries()).map(([bankId, value]) => [
          bankId,
          value.initialFailedOperations,
        ])
      );
      const finalization = applyDrainResults(
        checkpoint,
        touchedEntriesByBank,
        finalStatsByBank,
        initialFailedByBank
      );
      finalized = finalization.completed;
      for (const warning of finalization.warnings) {
        console.warn(warning);
      }
      saveCheckpoint(args.checkpointPath, checkpoint);
    }
  } finally {
    await runtime.stop();
  }

  const summary = {
    profile: args.profile,
    api_url: runtime.apiUrl,
    discovered_sessions: discoveredSessions,
    planned_sessions: plannedEntries.length,
    imported_sessions: imported,
    finalized_sessions: finalized,
    failed_sessions: failed,
    skipped_empty: skippedEmpty,
    bank_strategy: args.bankStrategy,
    checkpoint_path: args.checkpointPath,
  };
  console.log(args.json ? JSON.stringify(summary, null, 2) : JSON.stringify(summary));
}

function resolveSymlinks(path: string): string {
  const seen = new Set<string>();
  let current = path;
  while (true) {
    if (seen.has(current)) return current;
    seen.add(current);
    let target: string;
    try {
      target = realpathSync(current);
    } catch {
      return current;
    }
    if (target === current) return current;
    current = target;
  }
}

function canonicalizeExecutionPath(path: string): string {
  const resolved = resolve(path);
  return resolveSymlinks(resolved);
}

export function isDirectExecution(
  entrypoint: string | undefined = process.argv[1],
  moduleUrl: string = import.meta.url
): boolean {
  if (!entrypoint) {
    return false;
  }
  return (
    canonicalizeExecutionPath(entrypoint) === canonicalizeExecutionPath(fileURLToPath(moduleUrl))
  );
}

if (isDirectExecution()) {
  runCli().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
