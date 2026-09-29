import { homedir } from "os";
import { dirname, join, resolve } from "path";
import { readFileSync, existsSync, mkdirSync, writeFileSync, readdirSync } from "fs";
import { deriveBankId, prepareRetentionTranscript } from "./index.js";
import type { PluginConfig, PluginHookAgentContext } from "./types.js";
import { parseSessionFile } from "./session-file.js";
import type { ParsedSessionFile, SessionMessage } from "./session-file.js";

// Re-exported: these used to live here and callers import them from this module.
export { parseSessionFile };
export type { ParsedSessionFile, SessionMessage };

export interface BackfillCliOptions {
  openclawRoot: string;
  includeArchive: boolean;
  selectedAgents?: Set<string>;
  limit?: number;
  bankStrategy: "mirror-config" | "agent" | "fixed";
  fixedBank?: string;
}

export interface BackfillPlanEntry {
  filePath: string;
  agentId: string;
  sessionId: string;
  startedAt?: string;
  bankId: string;
  documentId: string;
  transcript: string;
  messageCount: number;
}

export interface BackfillCheckpointEntry {
  status: "enqueued" | "completed" | "failed";
  bankId: string;
  filePath: string;
  sessionId: string;
  updatedAt: string;
  error?: string;
}

export interface BackfillCheckpoint {
  version: 1;
  entries: Record<string, BackfillCheckpointEntry>;
}

interface RawBackfillCheckpointEntry extends Omit<BackfillCheckpointEntry, "status"> {
  status: BackfillCheckpointEntry["status"] | "queued";
}

interface RawBackfillCheckpoint {
  version: 1;
  entries: Record<string, RawBackfillCheckpointEntry>;
}

interface SessionDirectory {
  agentId: string;
  path: string;
}

const DEFAULT_PLUGIN_CONFIG: PluginConfig = {
  dynamicBankId: true,
  retainRoles: ["user", "assistant"],
};

export function defaultOpenClawRoot(): string {
  return resolve(join(homedir(), ".openclaw"));
}

export function defaultCheckpointPath(openclawRoot: string): string {
  return join(openclawRoot, "data", "hindsight-backfill-checkpoint.json");
}

export function loadPluginConfigFromOpenClawRoot(openclawRoot: string): PluginConfig {
  const configPath = join(openclawRoot, "openclaw.json");
  const raw = JSON.parse(readFileSync(configPath, "utf8")) as {
    plugins?: { entries?: Record<string, { config?: PluginConfig }> };
  };
  return {
    ...DEFAULT_PLUGIN_CONFIG,
    ...(raw.plugins?.entries?.["hindsight-openclaw"]?.config || {}),
  };
}

function sessionDirectories(openclawRoot: string, includeArchive: boolean): SessionDirectory[] {
  const agentsRoot = join(openclawRoot, "agents");
  if (!existsSync(agentsRoot)) {
    return [];
  }
  const result: SessionDirectory[] = [];
  for (const entry of readdirSync(agentsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const agentId = entry.name;
    const sessionsDir = join(agentsRoot, agentId, "sessions");
    if (existsSync(sessionsDir)) {
      result.push({ agentId, path: sessionsDir });
    }
    if (includeArchive) {
      const archiveDir = join(agentsRoot, agentId, "sessions-archive-from-migration_backup");
      if (existsSync(archiveDir)) {
        result.push({ agentId, path: archiveDir });
      }
    }
  }
  return result.sort((a, b) => a.agentId.localeCompare(b.agentId) || a.path.localeCompare(b.path));
}

export function discoverSessionFiles(
  openclawRoot: string,
  includeArchive: boolean
): Array<{ agentId: string; filePath: string }> {
  const sessions: Array<{ agentId: string; filePath: string }> = [];
  for (const dir of sessionDirectories(openclawRoot, includeArchive)) {
    for (const entry of readdirSync(dir.path, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith(".jsonl")) continue;
      sessions.push({
        agentId: dir.agentId,
        filePath: join(dir.path, entry.name),
      });
    }
  }
  return sessions.sort(
    (a, b) => a.agentId.localeCompare(b.agentId) || a.filePath.localeCompare(b.filePath)
  );
}

function backfillContextForSession(session: ParsedSessionFile): PluginHookAgentContext {
  return {
    agentId: session.agentId,
    sessionKey: session.sessionKey,
  };
}

function deriveTargetBank(
  session: ParsedSessionFile,
  pluginConfig: PluginConfig,
  bankStrategy: BackfillCliOptions["bankStrategy"],
  fixedBank?: string
): string {
  if (bankStrategy === "agent") {
    return session.agentId;
  }
  if (bankStrategy === "fixed") {
    if (!fixedBank) {
      throw new Error("fixed bank strategy requires --fixed-bank");
    }
    return fixedBank;
  }
  return deriveBankId(backfillContextForSession(session), pluginConfig);
}

export function stableDocumentId(session: ParsedSessionFile, bankId: string): string {
  return `backfill::${bankId}::${session.agentId}::${session.sessionId}`;
}

export function buildBackfillPlan(
  pluginConfig: PluginConfig,
  opts: BackfillCliOptions
): { entries: BackfillPlanEntry[]; discoveredSessions: number; skippedEmpty: number } {
  const entries: BackfillPlanEntry[] = [];
  let discoveredSessions = 0;
  let skippedEmpty = 0;

  for (const candidate of discoverSessionFiles(opts.openclawRoot, opts.includeArchive)) {
    if (opts.selectedAgents && !opts.selectedAgents.has(candidate.agentId)) {
      continue;
    }
    discoveredSessions += 1;
    const parsed = parseSessionFile(candidate.filePath, candidate.agentId);
    const retention = prepareRetentionTranscript(parsed.messages, pluginConfig, true);
    if (!retention) {
      skippedEmpty += 1;
      continue;
    }
    const bankId = deriveTargetBank(parsed, pluginConfig, opts.bankStrategy, opts.fixedBank);
    entries.push({
      filePath: parsed.filePath,
      agentId: parsed.agentId,
      sessionId: parsed.sessionId,
      startedAt: parsed.startedAt,
      bankId,
      documentId: stableDocumentId(parsed, bankId),
      transcript: retention.transcript,
      messageCount: retention.messageCount,
    });
    if (opts.limit && entries.length >= opts.limit) {
      break;
    }
  }

  return { entries, discoveredSessions, skippedEmpty };
}

export function loadCheckpoint(checkpointPath: string): BackfillCheckpoint {
  if (!existsSync(checkpointPath)) {
    return { version: 1, entries: {} };
  }
  const raw = JSON.parse(readFileSync(checkpointPath, "utf8")) as RawBackfillCheckpoint;
  if (raw.version !== 1 || !raw.entries || typeof raw.entries !== "object") {
    return { version: 1, entries: {} };
  }
  return {
    version: 1,
    entries: Object.fromEntries(
      Object.entries(raw.entries).map(([key, entry]) => [
        key,
        {
          ...entry,
          status: entry.status === "queued" ? "enqueued" : entry.status,
        },
      ])
    ) as Record<string, BackfillCheckpointEntry>,
  };
}

export function saveCheckpoint(checkpointPath: string, checkpoint: BackfillCheckpoint): void {
  mkdirSync(dirname(checkpointPath), { recursive: true });
  writeFileSync(checkpointPath, JSON.stringify(checkpoint, null, 2) + "\n", "utf8");
}

export function checkpointKey(entry: Pick<BackfillPlanEntry, "bankId" | "documentId">): string {
  return `${entry.bankId}::${entry.documentId}`;
}
