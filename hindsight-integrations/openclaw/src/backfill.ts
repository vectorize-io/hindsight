#!/usr/bin/env node
import { existsSync } from 'fs';
import { join, resolve } from 'path';
import { HindsightClient } from './client.js';
import type { PluginConfig } from './types.js';
import {
  buildBackfillPlan,
  checkpointKey,
  defaultCheckpointPath,
  defaultOpenClawRoot,
  loadCheckpoint,
  loadPluginConfigFromOpenClawRoot,
  saveCheckpoint,
  type BackfillCliOptions,
} from './backfill-lib.js';

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
  bankStrategy: 'mirror-config' | 'agent' | 'fixed';
  fixedBank?: string;
  apiUrl?: string;
  apiToken?: string;
  maxPendingOperations?: number;
  waitUntilDrained: boolean;
}

function usage(): string {
  return [
    'Usage: hindsight-openclaw-backfill [options]',
    '',
    'Options:',
    '  --openclaw-root <path>        OpenClaw root directory (default: ~/.openclaw)',
    '  --profile <name>              Logical profile name for reporting (default: openclaw)',
    '  --agent <id>                  Restrict import to a specific agent (repeatable)',
    '  --include-archive             Include migration archives (default)',
    '  --exclude-archive             Exclude migration archives',
    '  --limit <n>                   Stop after enqueueing N sessions',
    '  --dry-run                     Build and print the import plan without enqueueing',
    '  --json                        Print final summary as JSON',
    '  --resume                      Skip entries already marked completed in the checkpoint',
    '  --checkpoint <path>           Path to checkpoint JSON',
    '  --bank-strategy <mode>        mirror-config | agent | fixed',
    '  --fixed-bank <id>             Required when bank strategy is fixed',
    '  --api-url <url>               Hindsight API base URL override',
    '  --api-token <token>           Hindsight API bearer token override',
    '  --max-pending-operations <n>  Wait until target bank queue is <= n before enqueueing',
    '  --wait-until-drained          Wait for all target banks to reach pending_operations=0 after enqueue',
    '  -h, --help                    Show this help',
  ].join('\n');
}

function parseArgs(argv: string[]): ParsedArgs {
  const args: ParsedArgs = {
    openclawRoot: defaultOpenClawRoot(),
    profile: 'openclaw',
    agents: [],
    includeArchive: true,
    dryRun: false,
    json: false,
    resume: false,
    checkpointPath: '',
    bankStrategy: 'mirror-config',
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
      case '--openclaw-root':
        args.openclawRoot = resolve(next());
        break;
      case '--profile':
        args.profile = next();
        break;
      case '--agent':
        args.agents.push(next());
        break;
      case '--include-archive':
        args.includeArchive = true;
        break;
      case '--exclude-archive':
        args.includeArchive = false;
        break;
      case '--limit':
        args.limit = Number(next());
        break;
      case '--dry-run':
        args.dryRun = true;
        break;
      case '--json':
        args.json = true;
        break;
      case '--resume':
        args.resume = true;
        break;
      case '--checkpoint':
        args.checkpointPath = resolve(next());
        break;
      case '--bank-strategy': {
        const value = next();
        if (value !== 'mirror-config' && value !== 'agent' && value !== 'fixed') {
          throw new Error(`invalid bank strategy: ${value}`);
        }
        args.bankStrategy = value;
        break;
      }
      case '--fixed-bank':
        args.fixedBank = next();
        break;
      case '--api-url':
        args.apiUrl = next();
        break;
      case '--api-token':
        args.apiToken = next();
        break;
      case '--max-pending-operations':
        args.maxPendingOperations = Number(next());
        break;
      case '--wait-until-drained':
        args.waitUntilDrained = true;
        break;
      case '-h':
      case '--help':
        console.log(usage());
        process.exit(0);
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }

  if (!args.checkpointPath) {
    args.checkpointPath = defaultCheckpointPath(args.openclawRoot);
  }
  if (args.bankStrategy === 'fixed' && !args.fixedBank) {
    throw new Error('--fixed-bank is required when --bank-strategy fixed is used');
  }
  return args;
}

function inferApiSettings(pluginConfig: PluginConfig, explicitApiUrl?: string, explicitApiToken?: string): { apiUrl: string; apiToken?: string } {
  const apiUrl = explicitApiUrl
    || process.env.HINDSIGHT_EMBED_API_URL
    || pluginConfig.hindsightApiUrl
    || `http://127.0.0.1:${pluginConfig.apiPort || 9077}`;
  const apiToken = explicitApiToken
    || process.env.HINDSIGHT_EMBED_API_TOKEN
    || pluginConfig.hindsightApiToken;
  return { apiUrl, apiToken: apiToken || undefined };
}

async function waitForBankQueue(client: HindsightClient, maxPendingOperations: number): Promise<void> {
  for (;;) {
    try {
      const stats = await client.getBankStats();
      if (stats.pending_operations <= maxPendingOperations) {
        return;
      }
    } catch (error) {
      // New banks do not have stats until the first retain creates them.
      if (error instanceof Error && error.message.includes('HTTP 404')) {
        return;
      }
      throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

async function waitForBanksToDrain(clientsByBankId: Map<string, HindsightClient>): Promise<void> {
  for (;;) {
    const stats = await Promise.all(
      Array.from(clientsByBankId.entries()).map(async ([bankId, client]) => ({ bankId, stats: await client.getBankStats() })),
    );
    const pending = stats.filter(({ stats: bankStats }) => bankStats.pending_operations > 0);
    if (pending.length === 0) {
      return;
    }
    console.log(
      pending
        .map(({ bankId, stats: bankStats }) => `${bankId}\tpending_operations=${bankStats.pending_operations}\tpending_consolidation=${bankStats.pending_consolidation}`)
        .join('\n'),
    );
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!existsSync(join(args.openclawRoot, 'openclaw.json'))) {
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
  const { entries, discoveredSessions, skippedEmpty } = buildBackfillPlan(pluginConfig, backfillOptions);
  const planEntries = args.resume
    ? entries.filter((entry) => checkpoint.entries[checkpointKey(entry)]?.status !== 'queued')
    : entries;

  if (args.dryRun) {
    for (const entry of planEntries) {
      console.log(`${entry.agentId}\t${entry.bankId}\t${entry.sessionId}\tmsgs=${entry.messageCount}\tchars=${entry.transcript.length}`);
    }
    const summary = {
      profile: args.profile,
      dry_run: true,
      discovered_sessions: discoveredSessions,
      planned_sessions: planEntries.length,
      skipped_empty: skippedEmpty,
      bank_strategy: args.bankStrategy,
      checkpoint_path: args.checkpointPath,
    };
    console.log(args.json ? JSON.stringify(summary, null, 2) : JSON.stringify(summary));
    return;
  }

  const { apiUrl, apiToken } = inferApiSettings(pluginConfig, args.apiUrl, args.apiToken);
  const clientsByBankId = new Map<string, HindsightClient>();
  let imported = 0;
  let failed = 0;

  for (const entry of planEntries) {
    let client = clientsByBankId.get(entry.bankId);
    if (!client) {
      client = new HindsightClient({
        llmModel: pluginConfig.llmModel,
        embedVersion: pluginConfig.embedVersion,
        embedPackagePath: pluginConfig.embedPackagePath,
        apiUrl,
        apiToken,
      });
      client.setBankId(entry.bankId);
      clientsByBankId.set(entry.bankId, client);
    }

    if (typeof args.maxPendingOperations === 'number' && args.maxPendingOperations >= 0) {
      await waitForBankQueue(client, args.maxPendingOperations);
    }

    try {
      const metadata: Record<string, string> = {
        source: 'openclaw-backfill',
        file_path: entry.filePath,
        agent_id: entry.agentId,
        session_id: entry.sessionId,
        retained_at: new Date().toISOString(),
      };
      if (entry.startedAt) {
        metadata.session_started_at = entry.startedAt;
      }
      await client.retain({
        content: entry.transcript,
        document_id: entry.documentId,
        metadata,
      });
      checkpoint.entries[checkpointKey(entry)] = {
        status: 'queued',
        bankId: entry.bankId,
        filePath: entry.filePath,
        sessionId: entry.sessionId,
        updatedAt: new Date().toISOString(),
      };
      saveCheckpoint(args.checkpointPath, checkpoint);
      console.log(`${entry.agentId}\t${entry.bankId}\t${entry.sessionId}\tqueued`);
      imported += 1;
    } catch (error) {
      checkpoint.entries[checkpointKey(entry)] = {
        status: 'failed',
        bankId: entry.bankId,
        filePath: entry.filePath,
        sessionId: entry.sessionId,
        updatedAt: new Date().toISOString(),
        error: error instanceof Error ? error.message : String(error),
      };
      saveCheckpoint(args.checkpointPath, checkpoint);
      failed += 1;
      console.error(`${entry.agentId}\t${entry.bankId}\t${entry.sessionId}\tfailed\t${error instanceof Error ? error.message : String(error)}`);
    }
  }

  if (args.waitUntilDrained) {
    await waitForBanksToDrain(clientsByBankId);
  }

  const summary = {
    profile: args.profile,
    api_url: apiUrl,
    discovered_sessions: discoveredSessions,
    planned_sessions: planEntries.length,
    imported_sessions: imported,
    failed_sessions: failed,
    skipped_empty: skippedEmpty,
    bank_strategy: args.bankStrategy,
    checkpoint_path: args.checkpointPath,
  };
  console.log(args.json ? JSON.stringify(summary, null, 2) : JSON.stringify(summary));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
