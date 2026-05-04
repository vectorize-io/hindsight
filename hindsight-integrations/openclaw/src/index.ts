import type {
  MoltbotPluginAPI,
  PluginConfig,
  PluginHookAgentContext,
  PluginToolContext,
  MemoryResult,
  RetainRequest,
} from "./types.js";
import { HindsightServer, type Logger } from "@vectorize-io/hindsight-all";
import { HindsightClient, type HindsightClientOptions } from "@vectorize-io/hindsight-client";
import { RetainQueue } from "./retain-queue.js";
import { compileSessionPatterns, matchesSessionPattern } from "./session-patterns.js";
import { createHash } from "crypto";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import * as log from "./logger.js";
import { configureLogger, setApiLogger, stopLogger } from "./logger.js";
import { mkdirSync } from "fs";
import { createRequire } from "module";
import { homedir } from "os";
import { createKnowledgeTools, TOOL_NAMES } from "@vectorize-io/hindsight-agent-sdk";

function loadPackageVersion(): string {
  try {
    const require = createRequire(import.meta.url);
    const pkg = require("../package.json") as { version?: string };
    return pkg.version ?? "0.0.0";
  } catch {
    return "0.0.0";
  }
}

const USER_AGENT = `hindsight-openclaw/${loadPackageVersion()}`;

// Logger adapter that routes the embed wrapper's output through openclaw's
// batched structured logger so messages share the same prefix and respect
// the configured log level.
const embedLogger: Logger = {
  debug: (msg) => log.verbose(msg),
  info: (msg) => log.info(msg),
  warn: (msg) => log.warn(msg),
  error: (msg) => log.error(msg),
};

// Debug logging: silent by default, enable with debug: true or logLevel: 'debug'
let debugEnabled = false;
const debug = (...args: unknown[]) => {
  if (debugEnabled)
    log.verbose(
      args
        .map((a) => (typeof a === "string" ? a.replace(/^\[Hindsight\]\s*/, "") : String(a)))
        .join(" ")
    );
};

// Module-level state
let hindsightServer: HindsightServer | null = null;
let client: HindsightClient | null = null;
let clientOptions: HindsightClientOptions | null = null;
let initPromise: Promise<void> | null = null;
let isInitialized = false;
let usingExternalApi = false; // Track if using external API (skip daemon management)

// Capability detected once per service.start() against `<apiUrl>/version`.
// `true` when the Hindsight API supports `update_mode: 'append'` (added in
// 0.5.0 — see vectorize-io/hindsight#932). When false, retain falls back to a
// per-turn document id so prior turns aren't silently overwritten.
let supportsUpdateModeAppend = false;
let appendCapabilityProbed = false;
const MIN_VERSION_FOR_UPDATE_MODE_APPEND = "0.5.0";

// Store the current plugin config for bank ID derivation
let currentPluginConfig: PluginConfig | null = null;

// Track which banks have had their mission set (to avoid re-setting on every request).
// Under the old bespoke client we also cached a client instance per bank because the
// client carried a mutable bankId. HindsightClient takes bankId as a parameter on every
// call, so no per-bank caching is needed anymore — one module-level client is enough.
const banksWithMissionSet = new Set<string>();

// In-flight recall deduplication: concurrent recalls for the same bank reuse one promise
import type { RecallResponse } from "./types.js";
const inflightRecalls = new Map<string, Promise<RecallResponse>>();

// Lightweight bank-scoped facade over HindsightClient. Created per-request via
// getClientForContext() so hook bodies can keep their bankId-implicit style
// without going back to a stateful setBankId pattern. Also bridges the
// small shape differences (e.g. RetainRequest.metadata is Record<string, unknown>
// at build time; HindsightClient wants Record<string, string>).
export interface BankScopedClient {
  readonly bankId: string;
  retain(req: RetainRequest): Promise<void>;
  recall(
    req: {
      query: string;
      maxTokens?: number;
      budget?: "low" | "mid" | "high";
      types?: Array<"world" | "experience" | "observation">;
    },
    timeoutMs?: number
  ): Promise<RecallResponse>;
  setMission(mission: string): Promise<void>;
}

function scopeClient(c: HindsightClient, bankId: string): BankScopedClient {
  return {
    bankId,
    async retain(req) {
      await c.retain(bankId, req.content, {
        documentId: req.documentId,
        metadata: toStringMetadata(req.metadata),
        tags: req.tags,
        updateMode: req.updateMode,
        async: true,
      });
    },
    async recall(req, timeoutMs) {
      const call = c.recall(bankId, req.query, {
        maxTokens: req.maxTokens,
        budget: req.budget,
        types: req.types,
      });
      if (!timeoutMs) return call;
      // The generated client doesn't accept a per-call AbortSignal, so we race
      // against a TimeoutError here. The before_prompt_build caller already
      // special-cases `DOMException { name: 'TimeoutError' }` from the old
      // bespoke client, so we preserve that contract.
      return Promise.race([
        call,
        new Promise<never>((_, reject) =>
          setTimeout(
            () => reject(new DOMException(`Recall timed out after ${timeoutMs}ms`, "TimeoutError")),
            timeoutMs
          )
        ),
      ]);
    },
    async setMission(mission) {
      // createBank upserts the reflect mission. openclaw's old setBankMission
      // went through a dedicated PUT endpoint; this call lands on the same
      // server-side handler via the non-deprecated path.
      await c.createBank(bankId, { reflectMission: mission });
    },
  };
}

/**
 * The generated client's metadata type is `Record<string, string>`; the
 * openclaw builder uses `Record<string, unknown>` because some fields come
 * from optional plugin context. Drop undefined/null, stringify the rest.
 */
function toStringMetadata(
  input: Record<string, unknown> | undefined
): Record<string, string> | undefined {
  if (!input) return undefined;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(input)) {
    if (v === undefined || v === null) continue;
    out[k] = typeof v === "string" ? v : String(v);
  }
  return out;
}
const turnCountBySession = new Map<string, number>();
const MAX_TRACKED_SESSIONS = 10_000;
const DEFAULT_RECALL_TIMEOUT_MS = 10_000;

type SessionIdentityRecord = Pick<
  PluginHookAgentContext,
  "senderId" | "messageProvider" | "channelId"
>;
export type IdentitySkipReason =
  | {
      kind: "retryable";
      detail: "missing stable message provider" | "missing stable sender identity";
    }
  | { kind: "final"; detail: string };

const sessionIdentityBySession = new Map<string, SessionIdentityRecord>();
const skipHindsightTurnBySession = new Map<string, IdentitySkipReason>();
const documentSequenceBySession = new Map<string, number>();

// Cooldown + guard to prevent concurrent reinit attempts
let lastReinitAttempt = 0;
let isReinitInProgress = false;
const REINIT_COOLDOWN_MS = 30_000;

// Retain queue (external API mode only)
let retainQueue: RetainQueue | null = null;
let retainQueueFlushTimer: ReturnType<typeof setInterval> | null = null;
let isFlushInProgress = false;
const DEFAULT_FLUSH_INTERVAL_MS = 60_000; // 1 min

/**
 * Attempt to flush pending retains from the queue.
 * Each item is sent exactly as it would have been originally — same bank, payload, metadata.
 */
async function flushRetainQueue(): Promise<void> {
  if (!retainQueue || isFlushInProgress) return;
  const pending = retainQueue.size();
  if (pending === 0) return;

  isFlushInProgress = true;
  let flushed = 0;
  let failed = 0;

  try {
    if (!client) return; // no client yet — can't flush

    // Cleanup expired items first
    retainQueue.cleanup();

    const items = retainQueue.peek(50);
    const flushedIds: string[] = [];
    for (const item of items) {
      try {
        await client.retain(item.bankId, item.content, {
          documentId: item.documentId,
          metadata: toStringMetadata(item.metadata),
          tags: item.tags,
          updateMode: item.updateMode,
          async: true,
        });

        flushedIds.push(item.id);
        flushed++;
      } catch {
        // API still down — stop trying this batch
        failed++;
        break;
      }
    }

    if (flushedIds.length > 0) retainQueue.removeMany(flushedIds);
    const remaining = retainQueue.size();
    if (flushed > 0) {
      log.info(
        `queue flush: ${flushed} queued retains delivered${remaining > 0 ? `, ${remaining} still pending` : ", queue empty"}`
      );
    } else if (failed > 0) {
      debug(`[Hindsight] Queue flush: API still unreachable, ${remaining} retains pending`);
    }
  } finally {
    isFlushInProgress = false;
  }
}

const DEFAULT_RECALL_PROMPT_PREAMBLE =
  "Relevant memories from past conversations (prioritize recent when conflicting). Only use memories that are directly useful to continue this conversation; ignore the rest:";

function formatCurrentTimeForRecall(date = new Date()): string {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  const hours = String(date.getUTCHours()).padStart(2, "0");
  const minutes = String(date.getUTCMinutes()).padStart(2, "0");
  return `${year}-${month}-${day} ${hours}:${minutes}`;
}

/**
 * Lazy re-initialization after startup failure.
 * Called by waitForReady when initPromise rejected but API may now be reachable.
 * Throttled to one attempt per 30s to avoid hammering a down service.
 * Only works if initialization was attempted at least once (isInitialized guard).
 */
async function lazyReinit(configOverride?: PluginConfig): Promise<void> {
  const now = Date.now();
  if (now - lastReinitAttempt < REINIT_COOLDOWN_MS || isReinitInProgress) {
    return;
  }

  const config = configOverride ?? currentPluginConfig;
  if (!config) {
    debug("[Hindsight] lazyReinit skipped - no plugin config available");
    return;
  }

  // Persist config if we only have it from the live hook registration path.
  currentPluginConfig = config;

  isReinitInProgress = true;
  lastReinitAttempt = now;
  const externalApi = detectExternalApi(config);
  if (!externalApi.apiUrl) {
    isReinitInProgress = false;
    return; // Only external API mode supports lazy reinit
  }

  debug("[Hindsight] Attempting lazy re-initialization...");
  try {
    await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);
    await detectAppendCapability(externalApi.apiUrl, externalApi.apiToken);

    const llmConfig = detectLLMConfig(config);
    clientOptions = buildClientOptions(llmConfig, config, externalApi);
    banksWithMissionSet.clear();
    client = new HindsightClient(clientOptions);

    if (config.bankMission && usesStaticBank(config)) {
      const bankId = getStaticBankId(config);
      try {
        await scopeClient(client, bankId).setMission(config.bankMission);
        banksWithMissionSet.add(bankId);
      } catch (err) {
        log.warn(
          `could not set bank mission for ${bankId}: ${err instanceof Error ? err.message : err}`
        );
      }
    }

    usingExternalApi = true;
    isInitialized = true;
    // Replace the rejected initPromise with a resolved one
    initPromise = Promise.resolve();
    debug("[Hindsight] ✓ Lazy re-initialization succeeded");
  } catch (error) {
    log.warn(
      `lazy re-init failed (retry in ${REINIT_COOLDOWN_MS / 1000}s): ${error instanceof Error ? error.message : error}`
    );
  } finally {
    isReinitInProgress = false;
  }
}

// Global access for hooks (Moltbot loads hooks separately)
if (typeof global !== "undefined") {
  (global as any).__hindsightClient = {
    getClient: () => client,
    waitForReady: async () => {
      if (isInitialized) {
        return;
      }
      // If initPromise is null, it means service.start() hasn't been called yet
      // (CLI mode, not gateway mode). Hooks should gracefully no-op.
      if (!initPromise) {
        if (currentPluginConfig) {
          log.warn(
            "waitForReady called before service.start() — attempting lazy initialization fallback"
          );
          await lazyReinit(currentPluginConfig);
          return;
        }
        log.warn(
          "waitForReady called before service.start() — hooks will no-op (expected in CLI mode)"
        );
        return;
      }
      try {
        await initPromise;
      } catch {
        // Init failed (e.g., health check timeout at startup).
        // Attempt lazy re-initialization so Hindsight recovers
        // once the API becomes reachable again.
        if (!isInitialized) {
          await lazyReinit();
        }
      }
    },
    /**
     * Get a bank-scoped client handle for a specific agent context.
     * Derives the bank ID from the context for per-channel isolation and
     * ensures the bank mission is set on first use.
     */
    getClientForContext: async (
      ctx: PluginHookAgentContext | undefined
    ): Promise<BankScopedClient | null> => {
      if (!client) return null;
      const config = currentPluginConfig || {};
      const bankId = usesStaticBank(config) ? getStaticBankId(config) : deriveBankId(ctx, config);
      const scoped = scopeClient(client, bankId);

      // Set bank mission on first use of this bank (if configured).
      if (config.bankMission && !banksWithMissionSet.has(bankId)) {
        try {
          await scoped.setMission(config.bankMission);
          banksWithMissionSet.add(bankId);
          debug(`[Hindsight] Set mission for new bank: ${bankId}`);
        } catch (error) {
          // Log but don't fail - bank mission is not critical
          log.warn(`could not set bank mission for ${bankId}: ${error}`);
        }
      }

      return scoped;
    },
    getPluginConfig: () => currentPluginConfig,
  };
}

// Get directory of current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Default bank name (fallback when channel context not available)
const DEFAULT_BANK_NAME = "openclaw";

// Default granularity fields used by deriveBankId when not explicitly configured.
// This constant is shared between getPluginConfig (normalisation) and deriveBankId
// (fallback) so the skip-reason check and the bank-routing logic always agree.
const DEFAULT_DYNAMIC_BANK_GRANULARITY: Array<"agent" | "provider" | "channel" | "user"> = [
  "agent",
  "channel",
  "user",
];

// Throttle set: log an info-level skip message at most once per (sessionKey) per
// process lifetime so operators can discover silent retention/recall skips without
// flooding the log on every turn.
const loggedSkipSessions = new Set<string>();

function getConfiguredBankId(pluginConfig: PluginConfig): string | undefined {
  if (typeof pluginConfig.bankId !== "string") {
    return undefined;
  }

  const trimmed = pluginConfig.bankId.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function usesStaticBank(pluginConfig: PluginConfig): boolean {
  return pluginConfig.dynamicBankId === false;
}

function getDefaultBankId(pluginConfig: PluginConfig): string {
  return pluginConfig.bankIdPrefix
    ? `${pluginConfig.bankIdPrefix}-${DEFAULT_BANK_NAME}`
    : DEFAULT_BANK_NAME;
}

function getStaticBankId(pluginConfig: PluginConfig): string {
  const configuredBankId = getConfiguredBankId(pluginConfig);
  const baseBankId = configuredBankId || DEFAULT_BANK_NAME;
  return pluginConfig.bankIdPrefix ? `${pluginConfig.bankIdPrefix}-${baseBankId}` : baseBankId;
}

/**
 * Strip plugin-injected memory tags from content to prevent retain feedback loop.
 * Removes <hindsight_memories> and <relevant_memories> blocks that were injected
 * during before_agent_start so they don't get re-stored into the memory bank.
 */
export function stripMemoryTags(content: string): string {
  content = content.replace(/<hindsight_memories>[\s\S]*?<\/hindsight_memories>/g, "");
  content = content.replace(/<relevant_memories>[\s\S]*?<\/relevant_memories>/g, "");
  return content;
}

/**
 * Extract per-message retain tag overrides from inline user content.
 *
 * Supported forms:
 * - <retain_tags>tag:a, tag:b</retain_tags>
 * - <hindsight_retain_tags>tag:a, tag:b</hindsight_retain_tags>
 */
export function extractInlineRetainTags(content: string): string[] {
  if (!content) return [];

  const tags: string[] = [];
  const blockRe = /<(?:hindsight_)?retain_tags>([\s\S]*?)<\/(?:hindsight_)?retain_tags>/gi;
  let match: RegExpExecArray | null;

  while ((match = blockRe.exec(content)) !== null) {
    const normalized = normalizeRetainTags(match[1]);
    for (const tag of normalized) {
      if (!tags.includes(tag)) {
        tags.push(tag);
      }
    }
  }

  return tags;
}

/**
 * Remove inline retain tag directives from message content before storing it.
 */
export function stripInlineRetainTags(content: string): string {
  if (!content) return content;
  return content.replace(
    /<(?:hindsight_)?retain_tags>[\s\S]*?<\/(?:hindsight_)?retain_tags>/gi,
    ""
  );
}

/**
 * Strip OpenClaw's inline timestamp prefix (e.g. "[Wed 2026-04-15 10:44 GMT+2] ")
 * from the start of user-facing text. We lift this into a structured `timestamp`
 * field on the retained message instead, so facts aren't polluted by a weekday
 * prefix that varies per message.
 */
const INLINE_TIMESTAMP_PREFIX_RE =
  /^\s*\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\s+(?:GMT|UTC)(?:[+\-]\d{1,2}(?::\d{2})?)?\]\s*/;

export function stripInlineTimestampPrefix(content: string): string {
  if (!content) return content;
  return content.replace(INLINE_TIMESTAMP_PREFIX_RE, "");
}

/**
 * Extract sender_id from OpenClaw's injected inbound metadata blocks.
 * Checks both "Conversation info (untrusted metadata)" and "Sender (untrusted metadata)" blocks.
 * Returns the first sender_id / id string found, or undefined if none.
 */
export function extractSenderIdFromText(text: string): string | undefined {
  if (!text) return undefined;
  const metaBlockRe = /[\w\s]+\(untrusted metadata\)[^\n]*\n```json\n([\s\S]*?)\n```/gi;
  let match: RegExpExecArray | null;
  while ((match = metaBlockRe.exec(text)) !== null) {
    try {
      const obj = JSON.parse(match[1]);
      const id = obj?.sender_id ?? obj?.id;
      if (id && typeof id === "string") return id;
    } catch {
      // continue to next block
    }
  }
  return undefined;
}

/**
 * Strip OpenClaw sender/conversation metadata envelopes from message content.
 * These blocks are injected by OpenClaw but are noise for memory storage and recall.
 */
export function stripMetadataEnvelopes(content: string): string {
  // Strip: ---\n<Label> (untrusted metadata):\n```json\n{...}\n```\n<message>\n---
  content = content
    .replace(/^---\n[\w\s]+\(untrusted metadata\)[^\n]*\n```json[\s\S]*?```\n\n?/im, "")
    .replace(/\n---$/, "");
  // Strip: <Label> (untrusted metadata):\n```json\n{...}\n```  (without --- wrapper)
  content = content.replace(/[\w\s]+\(untrusted metadata\)[^\n]*\n```json[\s\S]*?```\n?/gim, "");
  return content.trim();
}

/**
 * Extract a recall query from a hook event's rawMessage or prompt.
 *
 * Prefers rawMessage (clean user text). Falls back to prompt, stripping
 * envelope formatting (System: lines, [Channel ...] headers, [from: X] footers).
 *
 * Returns null when no usable query (< 5 chars) can be extracted.
 */
export function extractRecallQuery(
  rawMessage: string | undefined,
  prompt: string | undefined
): string | null {
  // Reject known metadata/system message patterns — these are not user queries
  const METADATA_PATTERNS = [
    /^\s*conversation info\s*\(untrusted metadata\)/i,
    /^\s*\(untrusted metadata\)/i,
    /^\s*system:/i,
  ];
  const isMetadata = (s: string) => METADATA_PATTERNS.some((p) => p.test(s));

  let recallQuery = rawMessage;
  // Strip sender metadata envelope before any checks
  if (recallQuery) {
    recallQuery = stripMetadataEnvelopes(recallQuery);
  }
  if (
    !recallQuery ||
    typeof recallQuery !== "string" ||
    recallQuery.trim().length < 5 ||
    isMetadata(recallQuery)
  ) {
    recallQuery = prompt;
    // Strip metadata envelopes from prompt too, then check if anything useful remains
    if (recallQuery) {
      recallQuery = stripMetadataEnvelopes(recallQuery);
    }
    if (!recallQuery || recallQuery.length < 5) {
      return null;
    }

    // Strip envelope-formatted prompts from any channel
    let cleaned = recallQuery;

    // Remove leading "System: ..." lines (from prependSystemEvents)
    cleaned = cleaned.replace(/^(?:System:.*\n)+\n?/, "");

    // Remove session abort hint
    cleaned = cleaned.replace(/^Note: The previous agent run was aborted[^\n]*\n\n/, "");

    // Extract message after [ChannelName ...] envelope header
    const envelopeMatch = cleaned.match(/\[[A-Z][A-Za-z]*(?:\s[^\]]+)?\]\s*([\s\S]+)$/);
    if (envelopeMatch) {
      cleaned = envelopeMatch[1];
    }

    // Remove trailing [from: SenderName] metadata (group chats)
    cleaned = cleaned.replace(/\n\[from:[^\]]*\]\s*$/, "");

    // Strip metadata envelopes again after channel envelope extraction, in case
    // the metadata block appeared after the [ChannelName] header
    cleaned = stripMetadataEnvelopes(cleaned);

    recallQuery = cleaned.trim() || recallQuery;
  }

  const trimmed = recallQuery.trim();
  if (trimmed.length < 5 || isMetadata(trimmed)) return null;
  return trimmed;
}

export function composeRecallQuery(
  latestQuery: string,
  messages: any[] | undefined,
  recallContextTurns: number,
  recallRoles: Array<"user" | "assistant" | "system" | "tool"> = ["user", "assistant"]
): string {
  const latest = latestQuery.trim();
  if (recallContextTurns <= 1 || !Array.isArray(messages) || messages.length === 0) {
    return latest;
  }

  const allowedRoles = new Set(recallRoles);
  const contextualMessages = sliceLastTurnsByUserBoundary(messages, recallContextTurns);
  const contextLines = contextualMessages
    .map((msg: any) => {
      const role = msg?.role;
      if (!allowedRoles.has(role)) {
        return null;
      }

      let content = "";
      if (typeof msg?.content === "string") {
        content = msg.content;
      } else if (Array.isArray(msg?.content)) {
        content = msg.content
          .filter((block: any) => block?.type === "text" && typeof block?.text === "string")
          .map((block: any) => block.text)
          .join("\n");
      }

      content = stripMemoryTags(content).trim();
      content = stripMetadataEnvelopes(content);
      if (!content) {
        return null;
      }
      if (role === "user" && content === latest) {
        return null;
      }
      return `${role}: ${content}`;
    })
    .filter((line: string | null): line is string => Boolean(line));

  if (contextLines.length === 0) {
    return latest;
  }

  return ["Prior context:", contextLines.join("\n"), latest].join("\n\n");
}

export function truncateRecallQuery(query: string, latestQuery: string, maxChars: number): string {
  if (maxChars <= 0) {
    return query;
  }

  const latest = latestQuery.trim();
  if (query.length <= maxChars) {
    return query;
  }

  const latestOnly = latest.length <= maxChars ? latest : latest.slice(0, maxChars);

  if (!query.includes("Prior context:")) {
    return latestOnly;
  }

  // New order: Prior context at top, latest user message at bottom.
  // Truncate by dropping oldest context lines first to preserve the suffix.
  const contextMarker = "Prior context:\n\n";
  const markerIndex = query.indexOf(contextMarker);
  if (markerIndex === -1) {
    return latestOnly;
  }

  const suffixMarker = "\n\n" + latest;
  const suffixIndex = query.lastIndexOf(suffixMarker);
  if (suffixIndex === -1) {
    return latestOnly;
  }

  const suffix = query.slice(suffixIndex); // \n\n<latest>
  if (suffix.length >= maxChars) {
    return latestOnly;
  }

  const contextBody = query.slice(markerIndex + contextMarker.length, suffixIndex);
  const contextLines = contextBody.split("\n").filter(Boolean);
  const keptContextLines: string[] = [];

  // Add context lines from newest (bottom) to oldest (top), stopping when we exceed maxChars
  for (let i = contextLines.length - 1; i >= 0; i--) {
    keptContextLines.unshift(contextLines[i]);
    const candidate = `${contextMarker}${keptContextLines.join("\n")}${suffix}`;
    if (candidate.length > maxChars) {
      keptContextLines.shift();
      break;
    }
  }

  if (keptContextLines.length > 0) {
    return `${contextMarker}${keptContextLines.join("\n")}${suffix}`;
  }

  return latestOnly;
}

/**
 * Parse the OpenClaw sessionKey to extract context fields.
 * Format: "agent:{agentId}:{provider}:{channelType}:{channelId}[:{extra}]"
 * Example: "agent:c0der:telegram:group:-1003825475854:topic:42"
 */
// Some OpenClaw hook contexts populate `ctx.channelId` with the provider name
// (e.g. "discord") instead of the actual channel ID. Treat those as missing so
// we fall through to the sessionKey-derived channel. See issue #854.
const PROVIDER_CHANNEL_ID_TOKENS = new Set([
  "discord",
  "telegram",
  "slack",
  "matrix",
  "whatsapp",
  "signal",
  "messenger",
  "sms",
  "email",
  "web",
  "cli",
]);

function sanitizeChannelId(channelId: string | undefined, provider?: string): string | undefined {
  if (!channelId) return undefined;
  if (provider && channelId === provider) return undefined;
  if (PROVIDER_CHANNEL_ID_TOKENS.has(channelId.toLowerCase())) return undefined;
  return channelId;
}

export interface ParsedSessionKey {
  agentId?: string;
  provider?: string;
  channel?: string;
}

export function parseSessionKey(sessionKey: string): ParsedSessionKey {
  const parts = sessionKey.split(":");
  if (parts[0] !== "agent") return {};
  if (parts.length === 3 && parts[2] === "main") {
    return {
      agentId: parts[1],
      provider: "main",
      channel: "main",
    };
  }
  if (parts.length >= 4 && ["cron", "heartbeat", "subagent"].includes(parts[2])) {
    return {
      agentId: parts[1],
      provider: parts[2],
      channel: parts.slice(3).join(":"),
    };
  }
  if (parts.length < 5) return {};
  // parts[1] = agentId, parts[2] = provider, parts[3] = channelType, parts[4..] = channelId + extras
  return {
    agentId: parts[1],
    provider: parts[2],
    // Rejoin from channelType onward as the channel identifier (e.g. "group:-1003825475854:topic:42")
    channel: parts.slice(3).join(":"),
  };
}

export function extractTelegramDirectSenderId(channelId: string | undefined): string | undefined {
  if (typeof channelId !== "string") return undefined;
  const match = channelId.match(/^direct:([^:]+)$/);
  return match?.[1];
}

export function resolveSessionIdentity(
  ctx: PluginHookAgentContext | undefined
): PluginHookAgentContext | undefined {
  if (!ctx) return undefined;

  const sessionParsed = ctx.sessionKey ? parseSessionKey(ctx.sessionKey) : {};
  const messageProvider = ctx.messageProvider || sessionParsed.provider;
  const channelId = ctx.channelId || sessionParsed.channel;
  const senderId =
    ctx.senderId ||
    (messageProvider === "telegram" ? extractTelegramDirectSenderId(channelId) : undefined);

  return {
    ...ctx,
    agentId: ctx.agentId || sessionParsed.agentId,
    messageProvider,
    channelId,
    senderId,
  };
}

function retryableSkipReason(
  detail: "missing stable message provider" | "missing stable sender identity"
): IdentitySkipReason {
  return { kind: "retryable", detail };
}

function finalSkipReason(detail: string): IdentitySkipReason {
  return { kind: "final", detail };
}

function formatIdentitySkipReason(reason: IdentitySkipReason | undefined): string | undefined {
  return reason?.detail;
}

function isRetryableIdentitySkipReason(reason: IdentitySkipReason | undefined): boolean {
  return reason?.kind === "retryable";
}

/**
 * Log an identity-skip event at info level, throttled to once per session key
 * per process lifetime. This makes silent skips visible to operators without
 * flooding the log on every turn.
 */
function logSkipOnce(
  operation: "recall" | "retain" | "dispatch" | "agent_start",
  sessionKey: string | undefined,
  reason: IdentitySkipReason
): void {
  if (!sessionKey) return;
  const cacheKey = `${operation}:${sessionKey}`;
  if (loggedSkipSessions.has(cacheKey)) return;
  loggedSkipSessions.add(cacheKey);
  const hint =
    reason.kind === "final"
      ? ". If unexpected, set dynamicBankGranularity to ['agent','channel','user'] or use static banking (dynamicBankId: false + bankId: '<name>')"
      : "";
  log.info(`Skipping ${operation} on session '${sessionKey}': ${reason.detail}${hint}`);
}

function cacheSessionIdentity(
  sessionKey: string | undefined,
  resolvedCtx: PluginHookAgentContext | undefined
): void {
  if (!sessionKey || !resolvedCtx) return;
  if (!resolvedCtx.messageProvider && !resolvedCtx.channelId && !resolvedCtx.senderId) return;

  setCappedMapValue(sessionIdentityBySession, sessionKey, {
    senderId: resolvedCtx.senderId,
    messageProvider: resolvedCtx.messageProvider,
    channelId: resolvedCtx.channelId,
  });
}

interface ResolveAndCacheIdentityOptions {
  sessionKey?: string;
  ctx?: PluginHookAgentContext;
  senderIdHint?: string;
  dispatchChannel?: string;
  pluginConfig?: PluginConfig;
}

function resolveAndCacheIdentity(options: ResolveAndCacheIdentityOptions): {
  effectiveCtx: PluginHookAgentContext | undefined;
  resolvedCtx: PluginHookAgentContext | undefined;
  skipReason?: IdentitySkipReason;
} {
  const sessionKey = options.sessionKey ?? options.ctx?.sessionKey;
  const parsedSession = sessionKey ? parseSessionKey(sessionKey) : {};
  const cachedIdentity = sessionKey ? sessionIdentityBySession.get(sessionKey) : undefined;
  const baseCtx =
    options.ctx || (sessionKey ? ({ sessionKey } as PluginHookAgentContext) : undefined);
  const effectiveCtx =
    baseCtx || cachedIdentity || options.senderIdHint || options.dispatchChannel || sessionKey
      ? {
          ...baseCtx,
          sessionKey: baseCtx?.sessionKey || sessionKey,
          agentId: baseCtx?.agentId || parsedSession.agentId,
          messageProvider: baseCtx?.messageProvider ?? cachedIdentity?.messageProvider,
          channelId: baseCtx?.channelId ?? cachedIdentity?.channelId,
          senderId: baseCtx?.senderId || cachedIdentity?.senderId || options.senderIdHint,
        }
      : undefined;
  const resolvedCtx = resolveSessionIdentity(
    effectiveCtx
      ? {
          ...effectiveCtx,
          messageProvider:
            effectiveCtx.messageProvider ?? parsedSession.provider ?? options.dispatchChannel,
          channelId: effectiveCtx.channelId ?? parsedSession.channel,
        }
      : undefined
  );

  if (
    parsedSession.provider &&
    options.dispatchChannel &&
    parsedSession.provider !== options.dispatchChannel
  ) {
    const skipReason = finalSkipReason(
      `dispatch surface ${options.dispatchChannel} does not match session provider ${parsedSession.provider}`
    );
    if (sessionKey) {
      setCappedMapValue(skipHindsightTurnBySession, sessionKey, skipReason);
    }
    return { effectiveCtx, resolvedCtx, skipReason };
  }

  cacheSessionIdentity(sessionKey, resolvedCtx);

  const { reason: skipReason } = getIdentitySkipReason(resolvedCtx, options.pluginConfig);
  if (sessionKey) {
    if (skipReason) {
      setCappedMapValue(skipHindsightTurnBySession, sessionKey, skipReason);
    } else {
      skipHindsightTurnBySession.delete(sessionKey);
    }
  }

  return { effectiveCtx, resolvedCtx, skipReason };
}

export function getIdentitySkipReason(
  ctx: PluginHookAgentContext | undefined,
  pluginConfig?: PluginConfig
): { resolvedCtx: PluginHookAgentContext | undefined; reason?: IdentitySkipReason } {
  const resolvedCtx = resolveSessionIdentity(ctx);
  const sessionKey = resolvedCtx?.sessionKey;
  // The "internal main" / "operational provider main" / "anonymous sender" filters
  // exist to keep the default multi-tenant bank from being polluted by CLI/main
  // sessions that lack a stable identity. They should NOT fire when the user has
  // explicitly opted into a routing scheme that expects those sessions:
  //   - dynamicBankGranularity includes 'agent' (the default) → each agent
  //     (including 'main') gets its own bank
  //   - dynamicBankId === false with a configured bankId → user pinned a single
  //     named bank and wants every session retained into it
  // When dynamicBankGranularity is unset, the default is ["agent","channel","user"]
  // which includes "agent", so agentBanking defaults to true to match deriveBankId.
  const agentBanking = pluginConfig?.dynamicBankGranularity?.includes("agent") ?? true;
  const staticBanking =
    pluginConfig?.dynamicBankId === false &&
    typeof pluginConfig?.bankId === "string" &&
    pluginConfig.bankId.length > 0;
  const allowCliSessions = agentBanking || staticBanking;

  if (typeof sessionKey === "string") {
    if (/^agent:[^:]+:(cron|heartbeat|subagent):/.test(sessionKey)) {
      return { resolvedCtx, reason: finalSkipReason(`operational session ${sessionKey}`) };
    }
    if (!allowCliSessions && /^agent:[^:]+:main$/.test(sessionKey)) {
      return { resolvedCtx, reason: finalSkipReason(`internal main session ${sessionKey}`) };
    }
    if (/^temp:/.test(sessionKey)) {
      return { resolvedCtx, reason: finalSkipReason(`ephemeral temp session ${sessionKey}`) };
    }
  }

  const operationalProviders = allowCliSessions
    ? ["cron", "heartbeat", "subagent"]
    : ["cron", "heartbeat", "subagent", "main"];
  if (resolvedCtx?.messageProvider && operationalProviders.includes(resolvedCtx.messageProvider)) {
    return {
      resolvedCtx,
      reason: finalSkipReason(`operational provider ${resolvedCtx.messageProvider}`),
    };
  }
  if (!resolvedCtx?.messageProvider || resolvedCtx.messageProvider === "unknown") {
    return { resolvedCtx, reason: retryableSkipReason("missing stable message provider") };
  }
  if (!resolvedCtx?.senderId || resolvedCtx.senderId === "anonymous") {
    if (allowCliSessions && resolvedCtx?.agentId) {
      resolvedCtx.senderId = `agent-user:${resolvedCtx.agentId}`;
    } else {
      return { resolvedCtx, reason: retryableSkipReason("missing stable sender identity") };
    }
  }
  if (
    resolvedCtx.messageProvider === "telegram" &&
    typeof resolvedCtx.channelId === "string" &&
    resolvedCtx.channelId.startsWith("direct:")
  ) {
    const directSenderId = extractTelegramDirectSenderId(resolvedCtx.channelId);
    if (!directSenderId || directSenderId !== resolvedCtx.senderId) {
      return {
        resolvedCtx,
        reason: finalSkipReason(
          `telegram direct identity mismatch (${resolvedCtx.channelId} vs ${resolvedCtx.senderId})`
        ),
      };
    }
  }

  return { resolvedCtx, reason: undefined };
}

export function isEphemeralOperationalText(text: string | undefined): boolean {
  if (!text || typeof text !== "string") return false;

  const normalized = text
    .replace(/\[role:\s*[^\]]+\]\s*/gi, "")
    .replace(/\[[a-z]+:end\]\s*/gi, "")
    .trim();

  // These prefixes are OpenClaw-generated operational/session-bootstrap strings,
  // not user-authored content, so they should not create recall/retain entries.
  return [
    /^A new session was started via \/(?:new|reset)\./i,
    /^Based on this conversation, generate a short 1-2/i,
    /^This (?:script|task|job|workflow) updates .* index/i,
  ].some((pattern) => pattern.test(normalized));
}

function setCappedMapValue<K, V>(map: Map<K, V>, key: K, value: V): void {
  // FIFO cap, not LRU: updating an existing key keeps its original insertion order.
  map.set(key, value);
  if (map.size > MAX_TRACKED_SESSIONS) {
    const oldest = map.keys().next().value;
    if (oldest) map.delete(oldest);
  }
}

/**
 * Derive a bank ID from the agent context.
 * Uses configurable dynamicBankGranularity to determine bank segmentation.
 * Falls back to default bank when context is unavailable.
 */
export function deriveBankId(
  ctx: PluginHookAgentContext | undefined,
  pluginConfig: PluginConfig
): string {
  if (pluginConfig.dynamicBankId === false) {
    return getStaticBankId(pluginConfig);
  }

  // When no context is available, fall back to the static default bank.
  if (!ctx) {
    return getDefaultBankId(pluginConfig);
  }

  const resolvedCtx = resolveSessionIdentity(ctx);
  const fields = pluginConfig.dynamicBankGranularity?.length
    ? pluginConfig.dynamicBankGranularity
    : DEFAULT_DYNAMIC_BANK_GRANULARITY;

  // Validate field names at runtime — typos silently produce 'unknown' segments
  const validFields = new Set(["agent", "channel", "user", "provider"]);
  for (const f of fields) {
    if (!validFields.has(f)) {
      log.warn(
        `unknown dynamicBankGranularity field "${f}" — will resolve to "unknown". Valid: agent, channel, user, provider`
      );
    }
  }

  // Parse sessionKey as fallback when direct context fields are missing
  const sessionParsed = resolvedCtx?.sessionKey ? parseSessionKey(resolvedCtx.sessionKey) : {};

  // Warn when 'user' is in active fields but senderId is missing — bank ID will contain "anonymous"
  if (fields.includes("user") && resolvedCtx && !resolvedCtx.senderId) {
    debug(
      '[Hindsight] senderId not available in context — bank ID will use "anonymous". Ensure your OpenClaw provider passes senderId.'
    );
  }

  const fieldMap: Record<string, string> = {
    agent: resolvedCtx?.agentId || sessionParsed.agentId || "default",
    channel:
      sanitizeChannelId(
        resolvedCtx?.channelId,
        resolvedCtx?.messageProvider || sessionParsed.provider
      ) ||
      sessionParsed.channel ||
      "unknown",
    user: resolvedCtx?.senderId || "anonymous",
    provider: resolvedCtx?.messageProvider || sessionParsed.provider || "unknown",
  };

  const baseBankId = fields.map((f) => encodeURIComponent(fieldMap[f] || "unknown")).join("::");

  return pluginConfig.bankIdPrefix ? `${pluginConfig.bankIdPrefix}-${baseBankId}` : baseBankId;
}

export function formatMemories(results: MemoryResult[]): string {
  if (!results || results.length === 0) return "";
  return results
    .map((r) => {
      const type = r.type ? ` [${r.type}]` : "";
      const date = r.mentioned_at ? ` (${r.mentioned_at})` : "";
      return `- ${r.text}${type}${date}`;
    })
    .join("\n\n");
}

// Providers that authenticate via OAuth or run locally — no API key needed.
const NO_KEY_REQUIRED_PROVIDERS = new Set(["ollama", "openai-codex", "claude-code"]);

export function detectLLMConfig(pluginConfig?: PluginConfig): {
  provider?: string;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  source: string;
} {
  // External API mode: the daemon handles LLM credentials, plugin doesn't need them.
  const externalApiCheck = detectExternalApi(pluginConfig);
  if (externalApiCheck.apiUrl) {
    return {
      provider: undefined,
      apiKey: undefined,
      model: undefined,
      baseUrl: undefined,
      source: "external-api-mode-no-llm",
    };
  }

  const provider = pluginConfig?.llmProvider;
  if (!provider) {
    throw new Error(
      `No LLM provider configured for the Hindsight memory plugin.\n\n` +
        `Set the provider via 'openclaw config set':\n` +
        `  openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai\n\n` +
        `For providers that need an API key, configure it as a SecretRef so the value\n` +
        `is read from an env var (or file/exec source) at runtime instead of stored in plain text:\n` +
        `  openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \\\n` +
        `      --ref-source env --ref-provider default --ref-id OPENAI_API_KEY\n\n` +
        `Providers that don't need an API key: ${[...NO_KEY_REQUIRED_PROVIDERS].join(", ")}.\n` +
        `Or point the plugin at an external Hindsight API by setting hindsightApiUrl instead.`
    );
  }

  const apiKey = pluginConfig?.llmApiKey ?? "";
  if (!apiKey && !NO_KEY_REQUIRED_PROVIDERS.has(provider)) {
    throw new Error(
      `llmProvider is set to "${provider}" but llmApiKey is empty.\n\n` +
        `Configure it via 'openclaw config set' as a SecretRef:\n` +
        `  openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \\\n` +
        `      --ref-source env --ref-provider default --ref-id OPENAI_API_KEY`
    );
  }

  return {
    provider,
    apiKey,
    model: pluginConfig?.llmModel,
    baseUrl: pluginConfig?.llmBaseUrl,
    source: "plugin config",
  };
}

/**
 * Detect external Hindsight API configuration from plugin config.
 */
export function detectExternalApi(pluginConfig?: PluginConfig): {
  apiUrl: string | null;
  apiToken: string | null;
} {
  return {
    apiUrl: pluginConfig?.hindsightApiUrl ?? null,
    apiToken: pluginConfig?.hindsightApiToken ?? null,
  };
}

/**
 * Build HindsightClientOptions for the generated hindsight-client. In
 * external-API mode we use the configured URL/token; in local daemon mode
 * the caller overrides with the daemon's base URL after start().
 * The llmConfig parameter is currently only consumed by the daemon manager
 * (via env vars); it's kept on the client builder signature so callers
 * don't need to branch and so future features can forward it.
 */
export function buildClientOptions(
  _llmConfig: { provider?: string; apiKey?: string; model?: string },
  _pluginCfg: PluginConfig,
  externalApi: { apiUrl: string | null; apiToken: string | null }
): HindsightClientOptions {
  return {
    baseUrl: externalApi.apiUrl ?? "",
    apiKey: externalApi.apiToken ?? undefined,
  };
}

/**
 * Health check for external Hindsight API.
 * Retries up to 3 times with 2s delay — container DNS may not be ready on first boot.
 */
/**
 * Compare two semver-shaped strings ("0.5.0", "0.4.22"). Returns true when
 * `actual >= minimum`. Tolerates pre-release suffixes (treats them as the
 * same major.minor.patch as the bare version — good enough for capability
 * gating).
 */
export function meetsMinimumVersion(actual: string, minimum: string): boolean {
  const parse = (v: string): number[] =>
    v
      .split("-")[0]
      .split(".")
      .map((part) => Number.parseInt(part, 10))
      .map((n) => (Number.isFinite(n) ? n : 0));
  const a = parse(actual);
  const m = parse(minimum);
  for (let i = 0; i < Math.max(a.length, m.length); i++) {
    const av = a[i] ?? 0;
    const mv = m[i] ?? 0;
    if (av > mv) return true;
    if (av < mv) return false;
  }
  return true;
}

/**
 * Probe `<apiUrl>/version` once at service.start to learn the running
 * Hindsight API version. Returns `null` (treated as "no append support") if
 * the endpoint is unreachable or returns malformed payload — conservative
 * fallback path is the right call when we can't be sure.
 */
async function fetchHindsightApiVersion(
  apiUrl: string,
  apiToken?: string | null
): Promise<string | null> {
  const versionUrl = `${apiUrl.replace(/\/$/, "")}/version`;
  try {
    const headers: Record<string, string> = { "User-Agent": USER_AGENT };
    if (apiToken) headers["Authorization"] = `Bearer ${apiToken}`;
    const response = await fetch(versionUrl, {
      signal: AbortSignal.timeout(5000),
      headers,
    });
    if (!response.ok) {
      debug(`[Hindsight] /version returned HTTP ${response.status}; assuming legacy`);
      return null;
    }
    const data = (await response.json()) as { api_version?: unknown };
    const v = typeof data.api_version === "string" ? data.api_version : null;
    if (!v) {
      debug(`[Hindsight] /version payload missing api_version; assuming legacy`);
    }
    return v;
  } catch (error) {
    debug(`[Hindsight] /version probe failed: ${String(error)}; assuming legacy`);
    return null;
  }
}

/**
 * Probe `/version` and update the module-level `supportsUpdateModeAppend`
 * capability flag accordingly. Logs a one-time WARN block when the API is
 * older than 0.5.0 — without `update_mode: 'append'`, every retain on the
 * same session id silently overwrites prior turns server-side.
 *
 * Called from the same code paths as the health check, so capability is
 * always re-evaluated when the plugin (re)connects to the API.
 */
async function detectAppendCapability(apiUrl: string, apiToken?: string | null): Promise<void> {
  const version = await fetchHindsightApiVersion(apiUrl, apiToken);
  const supported =
    version !== null && meetsMinimumVersion(version, MIN_VERSION_FOR_UPDATE_MODE_APPEND);
  const transitionedToUnsupported = supportsUpdateModeAppend && !supported;
  const firstProbe = !appendCapabilityProbed;
  appendCapabilityProbed = true;
  supportsUpdateModeAppend = supported;
  if (supported) {
    debug(`[Hindsight] API version ${version} supports update_mode=append`);
    return;
  }
  // Warn on the first probe when unsupported, and on any transition from
  // supported -> unsupported. Stay silent on subsequent re-probes that
  // confirm the same unsupported state.
  if (!firstProbe && !transitionedToUnsupported) return;
  log.warn(
    `[Hindsight] ⚠️  API at ${apiUrl} reports version "${version ?? "unknown"}", which is older than ${MIN_VERSION_FOR_UPDATE_MODE_APPEND}. ` +
      `Falling back to per-turn document ids — each retain becomes its own document instead of accumulating into one per-session document. ` +
      `Upgrade Hindsight to ${MIN_VERSION_FOR_UPDATE_MODE_APPEND} or newer to enable session-scoped retention with update_mode=append.`
  );
}

async function checkExternalApiHealth(apiUrl: string, apiToken?: string | null): Promise<void> {
  const healthUrl = `${apiUrl.replace(/\/$/, "")}/health`;
  const maxRetries = 3;
  const retryDelay = 2000;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      debug(
        `[Hindsight] Checking external API health at ${healthUrl}... (attempt ${attempt}/${maxRetries})`
      );
      const headers: Record<string, string> = { "User-Agent": USER_AGENT };
      if (apiToken) {
        headers["Authorization"] = `Bearer ${apiToken}`;
      }
      const response = await fetch(healthUrl, { signal: AbortSignal.timeout(10000), headers });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = (await response.json()) as { status?: string };
      debug(`[Hindsight] External API health: ${JSON.stringify(data)}`);
      return;
    } catch (error) {
      if (attempt < maxRetries) {
        debug(`[Hindsight] Health check attempt ${attempt} failed, retrying in ${retryDelay}ms...`);
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
      } else {
        throw new Error(`Cannot connect to external Hindsight API at ${apiUrl}: ${error}`, {
          cause: error,
        });
      }
    }
  }
}

export function normalizeRetainTags(value: unknown): string[] {
  if (value == null) return [];

  const rawItems = Array.isArray(value) ? value : typeof value === "string" ? value.split(",") : [];

  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const item of rawItems) {
    if (typeof item !== "string") continue;
    const tag = item.trim();
    if (!tag || seen.has(tag)) continue;
    seen.add(tag);
    normalized.push(tag);
  }
  return normalized;
}

function getPluginConfig(api: MoltbotPluginAPI): PluginConfig {
  const config = api.config.plugins?.entries?.["hindsight-openclaw"]?.config || {};
  const defaultMission =
    "You are an AI assistant helping users across multiple communication channels (Telegram, Slack, Discord, etc.). Remember user preferences, instructions, and important context from conversations to provide personalized assistance.";

  return {
    bankMission: config.bankMission || defaultMission,
    embedPort: config.embedPort || 0,
    daemonIdleTimeout: config.daemonIdleTimeout !== undefined ? config.daemonIdleTimeout : 0,
    embedVersion: config.embedVersion || "latest",
    embedPackagePath: config.embedPackagePath,
    llmProvider: config.llmProvider,
    llmModel: config.llmModel,
    llmApiKey: config.llmApiKey,
    llmBaseUrl: config.llmBaseUrl,
    hindsightApiUrl: config.hindsightApiUrl,
    hindsightApiToken: config.hindsightApiToken,
    apiPort: config.apiPort || 9077,
    // Dynamic bank ID options (default: enabled)
    dynamicBankId: config.dynamicBankId !== false,
    bankId:
      typeof config.bankId === "string" && config.bankId.trim().length > 0
        ? config.bankId.trim()
        : undefined,
    bankIdPrefix: config.bankIdPrefix,
    retainTags: normalizeRetainTags(config.retainTags),
    retainSource:
      typeof config.retainSource === "string" && config.retainSource.trim().length > 0
        ? config.retainSource.trim()
        : undefined,
    excludeProviders: Array.isArray(config.excludeProviders)
      ? Array.from(
          new Set([
            "heartbeat",
            ...config.excludeProviders.filter(
              (provider): provider is string => typeof provider === "string"
            ),
          ])
        )
      : ["heartbeat"],
    autoRecall: config.autoRecall !== false, // Default: true (on) — backward compatible
    dynamicBankGranularity: Array.isArray(config.dynamicBankGranularity)
      ? config.dynamicBankGranularity
      : DEFAULT_DYNAMIC_BANK_GRANULARITY,
    autoRetain: config.autoRetain !== false, // Default: true
    retainRoles: Array.isArray(config.retainRoles) ? config.retainRoles : undefined,
    retainFormat: config.retainFormat === "text" ? "text" : "json",
    retainToolCalls: config.retainToolCalls !== false,
    recallBudget: config.recallBudget || "mid",
    recallMaxTokens: config.recallMaxTokens || 1024,
    recallTypes: Array.isArray(config.recallTypes) ? config.recallTypes : ["world", "experience"],
    recallRoles: Array.isArray(config.recallRoles) ? config.recallRoles : ["user", "assistant"],
    retainEveryNTurns:
      typeof config.retainEveryNTurns === "number" && config.retainEveryNTurns >= 1
        ? config.retainEveryNTurns
        : 1,
    retainOverlapTurns:
      typeof config.retainOverlapTurns === "number" && config.retainOverlapTurns >= 0
        ? config.retainOverlapTurns
        : 0,
    recallTopK: typeof config.recallTopK === "number" ? config.recallTopK : undefined,
    recallContextTurns:
      typeof config.recallContextTurns === "number" && config.recallContextTurns >= 1
        ? config.recallContextTurns
        : 1,
    recallMaxQueryChars:
      typeof config.recallMaxQueryChars === "number" && config.recallMaxQueryChars >= 1
        ? config.recallMaxQueryChars
        : 800,
    recallPromptPreamble:
      typeof config.recallPromptPreamble === "string" &&
      config.recallPromptPreamble.trim().length > 0
        ? config.recallPromptPreamble
        : DEFAULT_RECALL_PROMPT_PREAMBLE,
    recallInjectionPosition:
      typeof config.recallInjectionPosition === "string" &&
      ["prepend", "append", "user"].includes(config.recallInjectionPosition)
        ? (config.recallInjectionPosition as PluginConfig["recallInjectionPosition"])
        : undefined,
    recallTimeoutMs:
      typeof config.recallTimeoutMs === "number" && config.recallTimeoutMs >= 1000
        ? config.recallTimeoutMs
        : undefined,
    ignoreSessionPatterns: Array.isArray(config.ignoreSessionPatterns)
      ? config.ignoreSessionPatterns
      : [],
    statelessSessionPatterns: Array.isArray(config.statelessSessionPatterns)
      ? config.statelessSessionPatterns
      : [],
    skipStatelessSessions: config.skipStatelessSessions !== false,
    debug: config.debug ?? false,
  };
}

// Registration guard: WeakSet keyed by api instance to prevent double-registration
// on the same api object while allowing fresh registration on new api objects.
// Does not reintroduce issue #1029 because WeakSet.has() checks object identity,
// not a module-level boolean.
const _registeredApis = new WeakSet<MoltbotPluginAPI>();

export default function (api: MoltbotPluginAPI) {
  if (_registeredApis.has(api)) {
    debug("[Hindsight] Plugin entry skipped (this api instance already registered)");
    return;
  }
  _registeredApis.add(api);
  try {
    log.info("plugin entry invoked");
    debug("[Hindsight] Plugin loading...");

    // Get plugin config first (needed for debug flag and service registration)
    const pluginConfig = getPluginConfig(api);
    // If logLevel is 'debug', also enable legacy debug flag
    debugEnabled = pluginConfig.debug ?? pluginConfig.logLevel === "debug";

    // Configure structured logger — route through OpenClaw's api.logger for consistent formatting
    if (api.logger) setApiLogger(api.logger);
    configureLogger({
      logLevel: pluginConfig.logLevel ?? (pluginConfig.debug ? "debug" : "info"),
      logSummaryIntervalMs: pluginConfig.logSummaryIntervalMs,
    });

    // Store config globally for bank ID derivation in hooks
    currentPluginConfig = pluginConfig;

    debug("[Hindsight] Plugin loaded successfully (deferred heavy init to gateway start)");

    // Register background service for cleanup
    // IMPORTANT: Heavy initialization (LLM detection, daemon start, API health checks)
    // happens in service.start() which is ONLY called on gateway start,
    // not on every CLI command.
    debug("[Hindsight] Registering service...");
    log.info("registering plugin service");
    api.registerService({
      id: "hindsight-memory",
      async start() {
        log.info("service.start invoked");
        debug("[Hindsight] Service start called - beginning heavy initialization...");

        // Detect LLM configuration (env vars > plugin config > auto-detect)
        debug("[Hindsight] Detecting LLM config...");
        const llmConfig = detectLLMConfig(pluginConfig);

        const baseUrlInfo = llmConfig.baseUrl ? `, base URL: ${llmConfig.baseUrl}` : "";
        const modelInfo = llmConfig.model || "default";

        if (llmConfig.provider === "ollama") {
          debug(
            `[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${modelInfo} (${llmConfig.source})`
          );
        } else {
          debug(
            `[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${modelInfo} (${llmConfig.source}${baseUrlInfo})`
          );
        }
        if (pluginConfig.bankMission) {
          debug(
            `[Hindsight] Custom bank mission configured: "${pluginConfig.bankMission.substring(0, 50)}..."`
          );
        }

        // Log bank ID mode
        if (pluginConfig.dynamicBankId) {
          const prefixInfo = pluginConfig.bankIdPrefix
            ? ` (prefix: ${pluginConfig.bankIdPrefix})`
            : "";
          debug(
            `[Hindsight] ✓ Dynamic bank IDs enabled${prefixInfo} - each channel gets isolated memory`
          );
        } else {
          const sourceInfo = getConfiguredBankId(pluginConfig) ? "configured" : "default";
          debug(
            `[Hindsight] Dynamic bank IDs disabled - using ${sourceInfo} static bank: ${getStaticBankId(pluginConfig)}`
          );
        }

        // Detect external API mode
        const externalApi = detectExternalApi(pluginConfig);

        // Get API port from config (default: 9077)
        const apiPort = pluginConfig.apiPort || 9077;

        if (externalApi.apiUrl) {
          // External API mode - skip local daemon
          usingExternalApi = true;
          debug(`[Hindsight] ✓ Using external API: ${externalApi.apiUrl}`);

          // Initialize retain queue (external API mode only)
          try {
            const queueDir = pluginConfig.retainQueuePath
              ? dirname(pluginConfig.retainQueuePath)
              : join(homedir(), ".openclaw", "data");
            mkdirSync(queueDir, { recursive: true });
            const queuePath =
              pluginConfig.retainQueuePath || join(queueDir, "hindsight-retain-queue.jsonl");
            const queueFlushInterval =
              pluginConfig.retainQueueFlushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;
            const queueMaxAge = pluginConfig.retainQueueMaxAgeMs ?? -1;
            retainQueue = new RetainQueue({ filePath: queuePath, maxAgeMs: queueMaxAge });
            const pending = retainQueue.size();
            if (pending > 0) {
              log.info(
                `retain queue: ${pending} items pending from previous session, will flush shortly`
              );
            }
            debug(`[Hindsight] Retain queue initialized: ${queuePath}`);

            // Periodic flush timer
            if (queueFlushInterval > 0) {
              retainQueueFlushTimer = setInterval(flushRetainQueue, queueFlushInterval);
              retainQueueFlushTimer.unref?.();
            }
          } catch (error) {
            log.warn(`could not initialize retain queue: ${error}`);
          }

          if (externalApi.apiToken) {
            debug("[Hindsight] API token configured");
          }
        } else {
          debug(
            `[Hindsight] Daemon idle timeout: ${pluginConfig.daemonIdleTimeout}s (0 = never timeout)`
          );
          debug(`[Hindsight] API Port: ${apiPort}`);
        }

        // Initialize (runs synchronously in service.start())
        debug("[Hindsight] Starting initialization...");
        initPromise = (async () => {
          try {
            if (usingExternalApi && externalApi.apiUrl) {
              // External API mode - check health, skip daemon startup
              debug("[Hindsight] External API mode - skipping local daemon...");
              await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);
              await detectAppendCapability(externalApi.apiUrl, externalApi.apiToken);

              // Initialize client for external API
              debug("[Hindsight] Creating HindsightClient (external API)...");
              clientOptions = buildClientOptions(llmConfig, pluginConfig, externalApi);
              banksWithMissionSet.clear();
              client = new HindsightClient(clientOptions);

              const defaultBankId = deriveBankId(undefined, pluginConfig);
              debug(`[Hindsight] Default bank: ${defaultBankId}`);

              // Note: Bank mission will be set per-bank when dynamic bank IDs are enabled
              // For now, set it on the static default bank only.
              if (pluginConfig.bankMission && usesStaticBank(pluginConfig)) {
                debug(`[Hindsight] Setting bank mission...`);
                try {
                  await scopeClient(client, defaultBankId).setMission(pluginConfig.bankMission);
                  banksWithMissionSet.add(defaultBankId);
                } catch (err) {
                  log.warn(
                    `could not set bank mission for ${defaultBankId}: ${err instanceof Error ? err.message : err}`
                  );
                }
              }

              if (!isInitialized) {
                const mode = "external API";
                const autoRecall = pluginConfig.autoRecall !== false;
                const autoRetain = pluginConfig.autoRetain !== false;
                log.info(
                  `initialized (mode: ${mode}, bank: ${defaultBankId}, autoRecall: ${autoRecall}, autoRetain: ${autoRetain})`
                );
              }
              isInitialized = true;
              debug("[Hindsight] ✓ Ready (external API mode)");
            } else {
              // Local daemon mode - start hindsight-embed daemon
              debug("[Hindsight] Creating HindsightServer...");
              hindsightServer = new HindsightServer({
                profile: "openclaw",
                port: apiPort,
                embedVersion: pluginConfig.embedVersion,
                embedPackagePath: pluginConfig.embedPackagePath,
                env: {
                  HINDSIGHT_API_LLM_PROVIDER: llmConfig.provider || "",
                  HINDSIGHT_API_LLM_API_KEY: llmConfig.apiKey || "",
                  HINDSIGHT_API_LLM_MODEL: llmConfig.model,
                  HINDSIGHT_API_LLM_BASE_URL: llmConfig.baseUrl,
                  HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: String(pluginConfig.daemonIdleTimeout ?? 0),
                },
                logger: embedLogger,
              });

              // Start the embedded server
              debug("[Hindsight] Starting embedded server...");
              await hindsightServer.start();

              // Initialize client pointed at the local daemon URL
              debug("[Hindsight] Creating HindsightClient (local daemon)...");
              clientOptions = { baseUrl: hindsightServer.getBaseUrl() };
              banksWithMissionSet.clear();
              client = new HindsightClient(clientOptions);

              const defaultBankId = deriveBankId(undefined, pluginConfig);
              debug(`[Hindsight] Default bank: ${defaultBankId}`);

              // Note: Bank mission will be set per-bank when dynamic bank IDs are enabled
              // For now, set it on the static default bank only.
              if (pluginConfig.bankMission && usesStaticBank(pluginConfig)) {
                debug(`[Hindsight] Setting bank mission...`);
                try {
                  await scopeClient(client, defaultBankId).setMission(pluginConfig.bankMission);
                  banksWithMissionSet.add(defaultBankId);
                } catch (err) {
                  log.warn(
                    `could not set bank mission for ${defaultBankId}: ${err instanceof Error ? err.message : err}`
                  );
                }
              }

              if (!isInitialized) {
                const mode = "local daemon";
                const autoRecall = pluginConfig.autoRecall !== false;
                const autoRetain = pluginConfig.autoRetain !== false;
                log.info(
                  `initialized (mode: ${mode}, bank: ${defaultBankId}, autoRecall: ${autoRecall}, autoRetain: ${autoRetain})`
                );
              }
              isInitialized = true;
              debug("[Hindsight] ✓ Ready");
            }
          } catch (error) {
            log.error("initialization error", error);
            throw error;
          }
        })();

        // Wait for initialization to complete
        try {
          await initPromise;
        } catch (error) {
          log.error("initial initialization failed", error);
          // Continue to health check below
        }

        // External API mode: check external API health
        if (usingExternalApi) {
          const externalApi = detectExternalApi(pluginConfig);
          if (externalApi.apiUrl && isInitialized) {
            try {
              await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);
              await detectAppendCapability(externalApi.apiUrl, externalApi.apiToken);
              debug("[Hindsight] External API is healthy");
              return;
            } catch (error) {
              log.error("external API health check failed", error);
              // Reset state for reinitialization attempt
              client = null;
              clientOptions = null;
              banksWithMissionSet.clear();
              isInitialized = false;
            }
          }
        } else {
          // Local daemon mode: check daemon health (handles SIGUSR1 restart case)
          if (hindsightServer && isInitialized) {
            const healthy = await hindsightServer.checkHealth();
            if (healthy) {
              debug("[Hindsight] Daemon is healthy");
              return;
            }

            debug("[Hindsight] Daemon is not responding - reinitializing...");
            // Reset state for reinitialization
            hindsightServer = null;
            client = null;
            clientOptions = null;
            banksWithMissionSet.clear();
            isInitialized = false;
          }
        }

        // Reinitialize if needed (fresh start or recovery)
        if (!isInitialized) {
          debug("[Hindsight] Reinitializing...");
          const reinitPluginConfig = getPluginConfig(api);
          currentPluginConfig = reinitPluginConfig;
          const llmConfig = detectLLMConfig(reinitPluginConfig);
          const externalApi = detectExternalApi(reinitPluginConfig);
          const apiPort = reinitPluginConfig.apiPort || 9077;

          if (externalApi.apiUrl) {
            // External API mode
            usingExternalApi = true;

            await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);
            await detectAppendCapability(externalApi.apiUrl, externalApi.apiToken);

            clientOptions = buildClientOptions(llmConfig, reinitPluginConfig, externalApi);
            banksWithMissionSet.clear();
            client = new HindsightClient(clientOptions);
            const defaultBankId = deriveBankId(undefined, reinitPluginConfig);

            if (reinitPluginConfig.bankMission && usesStaticBank(reinitPluginConfig)) {
              try {
                await scopeClient(client, defaultBankId).setMission(reinitPluginConfig.bankMission);
                banksWithMissionSet.add(defaultBankId);
              } catch (err) {
                log.warn(
                  `could not set bank mission for ${defaultBankId}: ${err instanceof Error ? err.message : err}`
                );
              }
            }

            isInitialized = true;
            debug("[Hindsight] Reinitialization complete (external API mode)");
          } else {
            // Local daemon mode
            hindsightServer = new HindsightServer({
              profile: "openclaw",
              port: apiPort,
              embedVersion: reinitPluginConfig.embedVersion,
              embedPackagePath: reinitPluginConfig.embedPackagePath,
              env: {
                HINDSIGHT_API_LLM_PROVIDER: llmConfig.provider || "",
                HINDSIGHT_API_LLM_API_KEY: llmConfig.apiKey || "",
                HINDSIGHT_API_LLM_MODEL: llmConfig.model,
                HINDSIGHT_API_LLM_BASE_URL: llmConfig.baseUrl,
                HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: String(
                  reinitPluginConfig.daemonIdleTimeout ?? 0
                ),
              },
              logger: embedLogger,
            });

            await hindsightServer.start();

            clientOptions = { baseUrl: hindsightServer.getBaseUrl() };
            banksWithMissionSet.clear();
            client = new HindsightClient(clientOptions);
            const defaultBankId = deriveBankId(undefined, reinitPluginConfig);

            if (reinitPluginConfig.bankMission && usesStaticBank(reinitPluginConfig)) {
              try {
                await scopeClient(client, defaultBankId).setMission(reinitPluginConfig.bankMission);
                banksWithMissionSet.add(defaultBankId);
              } catch (err) {
                log.warn(
                  `could not set bank mission for ${defaultBankId}: ${err instanceof Error ? err.message : err}`
                );
              }
            }

            isInitialized = true;
            debug("[Hindsight] Reinitialization complete");
          }
        }
      },

      async stop() {
        try {
          debug("[Hindsight] Service stopping...");

          // Only stop daemon if in local mode
          if (!usingExternalApi && hindsightServer) {
            await hindsightServer.stop();
            hindsightServer = null;
          }

          // Close retain queue
          if (retainQueueFlushTimer) {
            clearInterval(retainQueueFlushTimer);
            retainQueueFlushTimer = null;
          }
          if (retainQueue) {
            const pending = retainQueue.size();
            if (pending > 0) {
              debug(
                `[Hindsight] Service stopping with ${pending} queued retains (will resume on next start)`
              );
            }
            retainQueue.close();
            retainQueue = null;
          }

          client = null;
          clientOptions = null;
          banksWithMissionSet.clear();
          isInitialized = false;

          stopLogger();
          debug("[Hindsight] Service stopped");
        } catch (error) {
          log.error("service stop error", error);
          throw error;
        }
      },
    });

    debug("[Hindsight] Plugin loaded successfully");

    // Register agent hooks for auto-recall and auto-retention.
    //
    // Why no module-level "already registered" guard: each plugin entry invocation
    // hands us a fresh `api` tied to a specific plugin registry. OpenClaw may call
    // the plugin entry multiple times per process (CLI vs gateway vs lazy reloads),
    // and the registry that's active when an agent actually runs is not guaranteed
    // to be the first one we saw. A process-global flag would let the first call
    // "win" and leave subsequent registries with zero hindsight hooks — which is
    // exactly how auto-recall/auto-retain silently stopped firing in 0.6.x.
    debug("[Hindsight] Registering agent hooks...");
    log.info("registering agent hooks");

    api.on("before_dispatch", async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        const sessionKey =
          ctx?.sessionKey ?? (typeof event?.sessionKey === "string" ? event.sessionKey : undefined);
        if (!sessionKey) {
          return;
        }

        const dispatchChannel =
          (typeof event?.channel === "string" ? event.channel : undefined) ||
          ctx?.messageProvider ||
          parseSessionKey(sessionKey).provider;
        const { resolvedCtx, skipReason } = resolveAndCacheIdentity({
          sessionKey,
          ctx: {
            ...ctx,
            sessionKey,
            senderId:
              (typeof event?.senderId === "string" ? event.senderId : undefined) || ctx?.senderId,
          },
          dispatchChannel,
          pluginConfig,
        });

        if (skipReason) {
          debug(
            `[Hindsight] before_dispatch marked session ${sessionKey} to skip this turn: ${formatIdentitySkipReason(skipReason)}`
          );
          logSkipOnce("dispatch", sessionKey, skipReason);
          return;
        }
        if (!resolvedCtx?.senderId || typeof resolvedCtx.senderId !== "string") {
          return;
        }

        debug(
          `[Hindsight] before_dispatch cached identity for ${sessionKey}: ${resolvedCtx.messageProvider}/${resolvedCtx.channelId} sender=${resolvedCtx.senderId}`
        );
      } catch (error) {
        log.warn(`before_dispatch identity cache error: ${error}`);
      }
    });

    api.on("before_agent_start", async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        const sessionKey =
          ctx?.sessionKey ?? (typeof event?.sessionKey === "string" ? event.sessionKey : undefined);
        const { resolvedCtx, skipReason } = resolveAndCacheIdentity({
          sessionKey,
          ctx,
          pluginConfig,
        });

        if (sessionKey && skipReason) {
          debug(
            `[Hindsight] before_agent_start skipping session ${sessionKey}: ${formatIdentitySkipReason(skipReason)}`
          );
          logSkipOnce("agent_start", sessionKey, skipReason);
          return;
        }
        if (!resolvedCtx) {
          return;
        }
        const bankId = deriveBankId(resolvedCtx, pluginConfig);
        debug(
          `[Hindsight] before_agent_start - bank: ${bankId}, channel: ${resolvedCtx.messageProvider}/${resolvedCtx.channelId}`
        );
      } catch (error) {
        log.warn(`before_agent_start identity resolution error: ${error}`);
      }
    });
    // Auto-recall: Inject relevant memories before agent processes the message
    // Hook signature: (event, ctx) where event has {prompt, messages?} and ctx has agent context
    api.on("before_prompt_build", async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        // Check if this provider is excluded
        if (ctx?.messageProvider && pluginConfig.excludeProviders?.includes(ctx.messageProvider)) {
          debug(`[Hindsight] Skipping recall for excluded provider: ${ctx.messageProvider}`);
          return;
        }

        // Session pattern filtering
        const sessionKey = ctx?.sessionKey;
        if (sessionKey) {
          const ignorePatterns = compileSessionPatterns(pluginConfig.ignoreSessionPatterns ?? []);
          if (ignorePatterns.length > 0 && matchesSessionPattern(sessionKey, ignorePatterns)) {
            debug(
              `[Hindsight] Skipping recall: session '${sessionKey}' matches ignoreSessionPatterns`
            );
            return;
          }
          const skipStateless = pluginConfig.skipStatelessSessions !== false;
          if (skipStateless) {
            const statelessPatterns = compileSessionPatterns(
              pluginConfig.statelessSessionPatterns ?? []
            );
            if (
              statelessPatterns.length > 0 &&
              matchesSessionPattern(sessionKey, statelessPatterns)
            ) {
              debug(
                `[Hindsight] Skipping recall: session '${sessionKey}' matches statelessSessionPatterns (skipStatelessSessions=true)`
              );
              return;
            }
          }
        }

        // Skip auto-recall when disabled (agent has its own recall tool)
        if (!pluginConfig.autoRecall) {
          debug("[Hindsight] Auto-recall disabled via config, skipping");
          return;
        }

        const sessionKeyForCache =
          ctx?.sessionKey ?? (typeof event?.sessionKey === "string" ? event.sessionKey : undefined);
        const skipTurnReason = sessionKeyForCache
          ? skipHindsightTurnBySession.get(sessionKeyForCache)
          : undefined;
        if (skipTurnReason && !isRetryableIdentitySkipReason(skipTurnReason)) {
          debug(
            `[Hindsight] Skipping recall for session ${sessionKeyForCache}: ${formatIdentitySkipReason(skipTurnReason)}`
          );
          logSkipOnce("recall", sessionKeyForCache, skipTurnReason);
          return;
        }

        const senderIdFromPrompt = !ctx?.senderId
          ? extractSenderIdFromText(event.prompt ?? event.rawMessage ?? "")
          : undefined;
        const { resolvedCtx: resolvedCtxForRecall, skipReason: identitySkipReason } =
          resolveAndCacheIdentity({
            sessionKey: sessionKeyForCache,
            ctx,
            senderIdHint: senderIdFromPrompt,
            pluginConfig,
          });
        if (identitySkipReason) {
          debug(
            `[Hindsight] Skipping recall for session ${sessionKeyForCache}: ${formatIdentitySkipReason(identitySkipReason)}`
          );
          logSkipOnce("recall", sessionKeyForCache, identitySkipReason);
          return;
        }

        const bankId = deriveBankId(resolvedCtxForRecall, pluginConfig);
        debug(
          `[Hindsight] before_prompt_build - bank: ${bankId}, channel: ${resolvedCtxForRecall?.messageProvider}/${resolvedCtxForRecall?.channelId}`
        );
        debug(`[Hindsight] event keys: ${Object.keys(event).join(", ")}`);
        debug(`[Hindsight] event.context keys: ${Object.keys(event.context ?? {}).join(", ")}`);

        // Get the user's latest message for recall — only the raw user text, not the full prompt
        // rawMessage is clean user text; prompt includes envelope, system events, media notes, etc.
        debug(
          `[Hindsight] extractRecallQuery input lengths - raw: ${event.rawMessage?.length ?? 0}, prompt: ${event.prompt?.length ?? 0}`
        );
        const extracted = extractRecallQuery(event.rawMessage, event.prompt);
        if (!extracted) {
          debug("[Hindsight] extractRecallQuery returned null, skipping recall");
          return;
        }
        if (isEphemeralOperationalText(extracted)) {
          debug("[Hindsight] Recall query is operational/ephemeral noise, skipping recall");
          return;
        }
        debug(`[Hindsight] extractRecallQuery result length: ${extracted.length}`);
        const recallContextTurns = pluginConfig.recallContextTurns ?? 1;
        const recallMaxQueryChars = pluginConfig.recallMaxQueryChars ?? 800;
        const sessionMessages = event.context?.sessionEntry?.messages ?? event.messages ?? [];
        const messageCount = sessionMessages.length;
        debug(
          `[Hindsight] event.messages count: ${messageCount}, roles: ${sessionMessages.map((m: any) => m.role).join(",")}`
        );
        if (recallContextTurns > 1 && messageCount === 0) {
          debug(
            "[Hindsight] recallContextTurns > 1 but event.messages is empty — prior context unavailable at before_agent_start for this provider"
          );
        }
        const recallRoles = pluginConfig.recallRoles ?? ["user", "assistant"];
        const composedPrompt = composeRecallQuery(
          extracted,
          sessionMessages,
          recallContextTurns,
          recallRoles
        );
        let prompt = truncateRecallQuery(composedPrompt, extracted, recallMaxQueryChars);

        // Final defensive cap
        if (prompt.length > recallMaxQueryChars) {
          prompt = prompt.substring(0, recallMaxQueryChars);
        }

        // Wait for client to be ready
        const clientGlobal = (global as any).__hindsightClient;
        if (!clientGlobal) {
          debug("[Hindsight] Client global not available, skipping auto-recall");
          return;
        }

        await clientGlobal.waitForReady();

        // Get client configured for this context's bank (async to handle mission setup)
        const client = await clientGlobal.getClientForContext(resolvedCtxForRecall);
        if (!client) {
          debug("[Hindsight] Client not initialized, skipping auto-recall");
          return;
        }

        debug(`[Hindsight] Auto-recall for bank ${bankId}, full query:\n---\n${prompt}\n---`);

        // Recall with deduplication: reuse in-flight request for same bank
        const normalizedPrompt = prompt.trim().toLowerCase().replace(/\s+/g, " ");
        const queryHash = createHash("sha256").update(normalizedPrompt).digest("hex").slice(0, 16);
        const recallKey = `${bankId}::${queryHash}`;
        const existing = inflightRecalls.get(recallKey);
        let recallPromise: Promise<RecallResponse>;
        if (existing) {
          debug(`[Hindsight] Reusing in-flight recall for bank ${bankId}`);
          recallPromise = existing;
        } else {
          const recallTimeoutMs = pluginConfig.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS;
          recallPromise = client.recall(
            {
              query: prompt,
              maxTokens: pluginConfig.recallMaxTokens || 1024,
              budget: pluginConfig.recallBudget,
              types: pluginConfig.recallTypes,
            },
            recallTimeoutMs
          );
          inflightRecalls.set(recallKey, recallPromise);
          void recallPromise.catch(() => {}).finally(() => inflightRecalls.delete(recallKey));
        }

        const response = await recallPromise;

        if (!response.results || response.results.length === 0) {
          debug("[Hindsight] No memories found for auto-recall");
          return;
        }

        debug(
          `[Hindsight] Raw recall response (${response.results.length} results before topK):\n${response.results.map((r: any, i: number) => `  [${i}] score=${r.score?.toFixed(3) ?? "n/a"} type=${r.type ?? "n/a"}: ${JSON.stringify(r.content ?? r.text ?? r).substring(0, 200)}`).join("\n")}`
        );

        const results = pluginConfig.recallTopK
          ? response.results.slice(0, pluginConfig.recallTopK)
          : response.results;

        debug(
          `[Hindsight] After topK (${pluginConfig.recallTopK ?? "unlimited"}): ${results.length} results injected`
        );

        // Format memories as JSON with all fields from recall
        const memoriesFormatted = formatMemories(results);

        const contextMessage = `<hindsight_memories>
${pluginConfig.recallPromptPreamble || DEFAULT_RECALL_PROMPT_PREAMBLE}
Current time - ${formatCurrentTimeForRecall()}

${memoriesFormatted}
</hindsight_memories>`;

        debug(`[Hindsight] Auto-recall: Injecting ${results.length} memories from bank ${bankId}`);
        log.info(`injecting ${results.length} memories into context (bank: ${bankId})`);
        log.trackRecall(bankId, results.length);

        // Inject recalled memories. Position is configurable to preserve prompt caching
        // when agents have large static system prompts.
        const position = pluginConfig.recallInjectionPosition || "prepend";
        switch (position) {
          case "append":
            return { appendSystemContext: contextMessage };
          case "user":
            return { prependContext: contextMessage };
          case "prepend":
          default:
            return { prependSystemContext: contextMessage };
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === "TimeoutError") {
          log.warn(
            `[Hindsight] Auto-recall timed out after ${pluginConfig.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS}ms, skipping memory injection`
          );
        } else if (error instanceof Error && error.name === "AbortError") {
          log.warn(
            `[Hindsight] Auto-recall aborted after ${pluginConfig.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS}ms, skipping memory injection`
          );
        } else {
          log.error("auto-recall error", error);
        }
        return;
      }
    });

    // Hook signature: (event, ctx) where event has {messages, success, error?, durationMs?}
    api.on("agent_end", async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        // Avoid cross-session contamination: only use context carried by this event.
        const eventSessionKey =
          typeof event?.sessionKey === "string" ? event.sessionKey : undefined;
        const effectiveCtx =
          ctx ||
          (eventSessionKey
            ? ({ sessionKey: eventSessionKey } as PluginHookAgentContext)
            : undefined);

        // Check if this provider is excluded
        if (
          effectiveCtx?.messageProvider &&
          pluginConfig.excludeProviders?.includes(effectiveCtx.messageProvider)
        ) {
          debug(
            `[Hindsight] Skipping retain for excluded provider: ${effectiveCtx.messageProvider}`
          );
          return;
        }

        // Session pattern filtering
        const agentEndSessionKey = effectiveCtx?.sessionKey;
        if (agentEndSessionKey) {
          const ignorePatterns = compileSessionPatterns(pluginConfig.ignoreSessionPatterns ?? []);
          if (
            ignorePatterns.length > 0 &&
            matchesSessionPattern(agentEndSessionKey, ignorePatterns)
          ) {
            debug(
              `[Hindsight] Skipping retain: session '${agentEndSessionKey}' matches ignoreSessionPatterns`
            );
            return;
          }
          const statelessPatterns = compileSessionPatterns(
            pluginConfig.statelessSessionPatterns ?? []
          );
          if (
            statelessPatterns.length > 0 &&
            matchesSessionPattern(agentEndSessionKey, statelessPatterns)
          ) {
            debug(
              `[Hindsight] Skipping retain: session '${agentEndSessionKey}' matches statelessSessionPatterns`
            );
            return;
          }
        }

        const sessionKeyForLookup = effectiveCtx?.sessionKey;
        const skipTurnReason = sessionKeyForLookup
          ? skipHindsightTurnBySession.get(sessionKeyForLookup)
          : undefined;
        if (skipTurnReason && !isRetryableIdentitySkipReason(skipTurnReason)) {
          debug(
            `[Hindsight Hook] Skipping retain for session ${sessionKeyForLookup}: ${formatIdentitySkipReason(skipTurnReason)}`
          );
          logSkipOnce("retain", sessionKeyForLookup, skipTurnReason);
          if (sessionKeyForLookup) {
            skipHindsightTurnBySession.delete(sessionKeyForLookup);
          }
          return;
        }

        const {
          effectiveCtx: effectiveCtxForRetain,
          resolvedCtx: resolvedCtxForRetain,
          skipReason: identitySkipReason,
        } = resolveAndCacheIdentity({
          sessionKey: sessionKeyForLookup,
          ctx: effectiveCtx,
          pluginConfig,
        });

        if (identitySkipReason) {
          debug(
            `[Hindsight Hook] Skipping retain for session ${sessionKeyForLookup}: ${formatIdentitySkipReason(identitySkipReason)}`
          );
          logSkipOnce("retain", sessionKeyForLookup, identitySkipReason);
          if (sessionKeyForLookup) {
            skipHindsightTurnBySession.delete(sessionKeyForLookup);
          }
          return;
        }
        if (sessionKeyForLookup) {
          skipHindsightTurnBySession.delete(sessionKeyForLookup);
        }

        const bankId = deriveBankId(resolvedCtxForRetain, pluginConfig);
        debug(`[Hindsight Hook] agent_end triggered - bank: ${bankId}`);

        if (event.success === false) {
          debug("[Hindsight Hook] Agent run failed, skipping retention");
          return;
        }

        if (
          !Array.isArray(event.context?.sessionEntry?.messages ?? event.messages) ||
          (event.context?.sessionEntry?.messages ?? event.messages ?? []).length === 0
        ) {
          debug("[Hindsight Hook] No messages in event, skipping retention");
          return;
        }

        if (pluginConfig.autoRetain === false) {
          debug("[Hindsight Hook] autoRetain is disabled, skipping retention");
          return;
        }

        // Chunked retention: skip non-Nth turns and use a sliding window when firing
        const retainEveryN = pluginConfig.retainEveryNTurns ?? 1;
        const allMessages = event.context?.sessionEntry?.messages ?? event.messages ?? [];
        let messagesToRetain = allMessages;
        let retainFullWindow = false;

        if (retainEveryN > 1) {
          const sessionTrackingKey = `${bankId}:${effectiveCtx?.sessionKey || "session"}`;
          const turnCount = (turnCountBySession.get(sessionTrackingKey) || 0) + 1;
          setCappedMapValue(turnCountBySession, sessionTrackingKey, turnCount);

          if (turnCount % retainEveryN !== 0) {
            const nextRetainAt = Math.ceil(turnCount / retainEveryN) * retainEveryN;
            debug(
              `[Hindsight Hook] Turn ${turnCount}/${retainEveryN}, skipping retain (next at turn ${nextRetainAt})`
            );
            return;
          }

          // Sliding window in turns: N turns + configured overlap turns.
          // We slice by actual turn boundaries (user-role messages), so this
          // remains stable even when system/tool messages are present.
          const overlapTurns = pluginConfig.retainOverlapTurns ?? 0;
          const windowTurns = retainEveryN + overlapTurns;
          messagesToRetain = sliceLastTurnsByUserBoundary(allMessages, windowTurns);
          retainFullWindow = true;
          debug(
            `[Hindsight Hook] Turn ${turnCount}: chunked retain firing (window: ${windowTurns} turns, ${messagesToRetain.length} messages)`
          );
        }

        const inlineRetainTags = normalizeRetainTags(
          messagesToRetain.flatMap((msg: any) => {
            if (msg?.role !== "user") {
              return [];
            }

            const content =
              typeof msg?.content === "string"
                ? msg.content
                : Array.isArray(msg?.content)
                  ? msg.content
                      .filter(
                        (block: any) => block?.type === "text" && typeof block?.text === "string"
                      )
                      .map((block: any) => block.text)
                      .join("\n")
                  : "";

            return extractInlineRetainTags(content);
          })
        );

        const retention = prepareRetentionTranscript(
          messagesToRetain,
          pluginConfig,
          retainFullWindow
        );
        if (!retention) {
          debug("[Hindsight Hook] No messages to retain (filtered/short/no-user)");
          return;
        }
        const { transcript, messageCount } = retention;

        if (isEphemeralOperationalText(transcript)) {
          debug("[Hindsight Hook] Transcript is operational/ephemeral noise, skipping retention");
          return;
        }

        // Wait for client to be ready
        const clientGlobal = (global as any).__hindsightClient;
        if (!clientGlobal) {
          log.warn("client global not found, skipping retain");
          return;
        }

        await clientGlobal.waitForReady();

        // Get client configured for this context's bank (async to handle mission setup)
        const client = await clientGlobal.getClientForContext(resolvedCtxForRetain);
        if (!client) {
          log.warn("client not initialized, skipping retain");
          return;
        }

        const retainNow = Date.now();
        const retainRequest = buildRetainRequest(
          transcript,
          messageCount,
          effectiveCtxForRetain,
          pluginConfig,
          retainNow,
          {
            retentionScope: retainFullWindow ? "window" : "turn",
            windowTurns: retainFullWindow
              ? (pluginConfig.retainEveryNTurns ?? 1) + (pluginConfig.retainOverlapTurns ?? 0)
              : undefined,
            tags: inlineRetainTags,
            appendSupported: supportsUpdateModeAppend,
          }
        );

        // Retain to Hindsight
        debug(
          `[Hindsight] Retaining to bank ${bankId}, document: ${retainRequest.documentId}, chars: ${transcript.length}\n---\n${transcript.substring(0, 500)}${transcript.length > 500 ? "\n...(truncated)" : ""}\n---`
        );

        try {
          await client.retain(retainRequest);
          log.trackRetain(bankId, messageCount);
          debug(
            `[Hindsight] Retained ${messageCount} messages to bank ${bankId} for session ${retainRequest.documentId}`
          );

          // After a successful retain, try flushing any queued items
          if (retainQueue && retainQueue.size() > 0) {
            flushRetainQueue().catch(() => {});
          }
        } catch (retainError) {
          // Queue the failed retain for later delivery (external API mode only)
          if (retainQueue) {
            retainQueue.enqueue(bankId, retainRequest, retainRequest.metadata);
            const pending = retainQueue.size();
            log.warn(
              `API unreachable — retain queued (${pending} pending, bank: ${bankId}): ${retainError instanceof Error ? retainError.message : retainError}`
            );
          } else {
            log.error("error retaining messages", retainError);
          }
        }
      } catch (error) {
        log.error("error retaining messages", error);
      }
    });
    debug("[Hindsight] Hooks registered");
    log.info("agent hooks registered");

    // Register knowledge tools (opt-in via enableKnowledgeTools config flag)
    if (pluginConfig.enableKnowledgeTools && typeof api.registerTool === "function") {
      try {
        const apiUrl = (() => {
          const ext = detectExternalApi(pluginConfig);
          return ext?.apiUrl || `http://localhost:${pluginConfig.apiPort || 9077}`;
        })();
        const apiToken = pluginConfig.hindsightApiToken || undefined;

        // Factory: called per session with agent context, returns tools scoped to that bank
        const factory = (ctx: PluginToolContext) => {
          const bankId = deriveBankId(ctx as any, pluginConfig);
          const tools = createKnowledgeTools({ apiUrl, apiToken, bankId });
          return tools.map((t) => ({
            name: t.name,
            label: t.label,
            description: t.description,
            parameters: t.parameters,
            async execute(_id: string, params: Record<string, unknown>) {
              return { ...(await t.execute(params)), details: {} };
            },
          }));
        };

        api.registerTool(factory, {
          names: [...TOOL_NAMES],
          optional: false,
        });
        log.info("knowledge tools registered");
      } catch (err) {
        log.warn(`knowledge tools registration failed: ${err}`);
      }
    }
  } catch (error) {
    log.error("plugin loading error", error);
    if (error instanceof Error) {
      log.error("error stack", error.stack);
    }
    throw error;
  }
}

// Export client getter for tools

function sanitizeDocumentIdPart(value: string | undefined, fallback: string): string {
  const normalized = (value || "").trim();
  if (!normalized) return fallback;
  return (
    normalized
      .replace(/[^a-zA-Z0-9:_-]+/g, "_")
      .replace(/_+/g, "_")
      .replace(/^_+|_+$/g, "") || fallback
  );
}

function getSessionDocumentBase(effectiveCtx: PluginHookAgentContext | undefined): string {
  const sessionKeyPart = sanitizeDocumentIdPart(effectiveCtx?.sessionKey, "session");
  return `openclaw:${sessionKeyPart}`;
}

function nextDocumentSequence(effectiveCtx: PluginHookAgentContext | undefined): number {
  const sequenceKey = effectiveCtx?.sessionKey || "session";
  const next = (documentSequenceBySession.get(sequenceKey) || 0) + 1;
  setCappedMapValue(documentSequenceBySession, sequenceKey, next);
  return next;
}

function extractThreadId(channelId: string | undefined): string | undefined {
  if (!channelId) return undefined;
  const match = channelId.match(/(?:^|:)topic:([^:]+)$/);
  return match?.[1];
}

export function buildRetainRequest(
  transcript: string,
  messageCount: number,
  effectiveCtx: PluginHookAgentContext | undefined,
  pluginConfig: PluginConfig,
  now = Date.now(),
  options?: {
    retentionScope?: "turn" | "window" | "manual";
    windowTurns?: number;
    turnIndex?: number;
    tags?: string[];
    /**
     * Whether the live Hindsight API supports `update_mode: 'append'`. When
     * true with `retainDocumentScope: 'session'`, the request gets a stable
     * per-session document id and `updateMode: 'append'` so each retain
     * concatenates to the existing document. When false, falls back to a
     * unique per-turn document id so prior turns aren't overwritten.
     * Defaults to false (conservative).
     */
    appendSupported?: boolean;
  }
): RetainRequest {
  const resolvedCtx = resolveSessionIdentity(effectiveCtx);
  const parsedSession = resolvedCtx?.sessionKey ? parseSessionKey(resolvedCtx.sessionKey) : {};
  const turnIndex = options?.turnIndex ?? nextDocumentSequence(resolvedCtx);
  const retentionScope = options?.retentionScope || "turn";
  const documentBase = getSessionDocumentBase(resolvedCtx);
  const documentScope = pluginConfig.retainDocumentScope ?? "session";
  const documentKind = retentionScope === "window" ? "window" : "turn";
  // Session-scope only stays session-scope when the API can append; otherwise
  // every retain on the same id silently overwrites prior turns (behavior
  // pre-#932). Force per-turn ids on legacy APIs.
  const useSessionScopedDoc = documentScope === "session" && options?.appendSupported === true;
  const documentId = useSessionScopedDoc
    ? documentBase
    : `${documentBase}:${documentKind}:${String(turnIndex).padStart(6, "0")}`;
  const provider = effectiveCtx?.messageProvider || parsedSession.provider;
  const channelId = sanitizeChannelId(effectiveCtx?.channelId, provider) || parsedSession.channel;
  const channelType = effectiveCtx?.messageProvider;
  const threadId = extractThreadId(channelId);
  const mergedTags = normalizeRetainTags([
    ...(pluginConfig.retainTags ?? []),
    ...(options?.tags ?? []),
  ]);

  return {
    content: transcript,
    documentId: documentId,
    metadata: {
      retained_at: new Date(now).toISOString(),
      message_count: String(messageCount),
      source: pluginConfig.retainSource || "openclaw",
      retention_scope: retentionScope,
      turn_index: String(turnIndex),
      session_key: resolvedCtx?.sessionKey,
      agent_id: resolvedCtx?.agentId || parsedSession.agentId,
      provider,
      channel_type: channelType,
      channel_id: channelId,
      thread_id: threadId,
      sender_id: resolvedCtx?.senderId,
      ...(options?.windowTurns !== undefined ? { window_turns: String(options.windowTurns) } : {}),
    },
    tags: mergedTags.length > 0 ? mergedTags : undefined,
    updateMode: useSessionScopedDoc ? "append" : undefined,
  };
}

export function prepareRetentionTranscript(
  messages: any[],
  pluginConfig: PluginConfig,
  retainFullWindow = false
): { transcript: string; messageCount: number } | null {
  if (!messages || messages.length === 0) {
    return null;
  }

  let targetMessages: any[];
  if (retainFullWindow) {
    // Chunked retention: retain the full sliding window (already sliced by caller)
    targetMessages = messages;
  } else {
    // Default: retain only the last turn (user message + assistant responses)
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx === -1) {
      return null; // No user message found in turn
    }
    targetMessages = messages.slice(lastUserIdx);
  }

  const format = pluginConfig.retainFormat ?? "json";
  const includeToolCalls = format === "json" && pluginConfig.retainToolCalls !== false;

  if (includeToolCalls) {
    const structured = buildAnthropicStructuredMessages(targetMessages, pluginConfig);
    if (structured.length === 0) return null;
    const transcript = JSON.stringify(structured);
    if (!transcript.trim() || transcript.length < 10) return null;
    return { transcript, messageCount: structured.length };
  }

  // Role filtering (text-only path)
  const allowedRoles = new Set(pluginConfig.retainRoles || ["user", "assistant"]);
  const filteredMessages = targetMessages.filter((m: any) => allowedRoles.has(m.role));

  if (filteredMessages.length === 0) {
    return null; // No messages to retain
  }

  const normalized: Array<{ role: string; content: string; timestamp?: string }> = [];
  for (const msg of filteredMessages) {
    const role = msg.role || "unknown";
    let content = "";

    if (typeof msg.content === "string") {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      content = msg.content
        .filter((block: any) => block.type === "text")
        .map((block: any) => block.text)
        .join("\n");
    }

    content = stripMemoryTags(content);
    content = stripInlineRetainTags(content);
    content = stripMetadataEnvelopes(content);
    content = stripInlineTimestampPrefix(content);

    if (content.trim()) {
      const timestamp = normalizeMessageTimestamp(msg);
      normalized.push(timestamp ? { role, content, timestamp } : { role, content });
    }
  }

  if (normalized.length === 0) return null;

  const transcript =
    format === "text"
      ? normalized
          .map(({ role, content }) => `[role: ${role}]\n${content}\n[${role}:end]`)
          .join("\n\n")
      : JSON.stringify(normalized);

  if (!transcript.trim() || transcript.length < 10) return null;

  return { transcript, messageCount: normalized.length };
}

// MCP tool name suffixes that are operational (recall/retain/search/CRUD) and
// shouldn't be retained — preserves agent reasoning without creating feedback
// loops on Hindsight's own MCP surface. Mirrors the claude-code integration.
const OPERATIONAL_TOOL_PATTERN =
  /(?:recall|retain|reflect|search|extract|create_|delete_|update_|get_|list_)/i;
const TOOL_RESULT_MAX_CHARS = 2000;

/**
 * Build an Anthropic-shaped message array from OpenClaw's session messages.
 *
 * OpenClaw stores assistant content as a block array that may contain
 * `text`, `thinking`, and `toolCall` entries, and emits tool results as
 * separate messages with `role: "toolResult"`. The Anthropic wire format
 * expected by Hindsight's Claude Code integration (and downstream consumers)
 * is: assistant messages carry `text` and `tool_use` blocks, and tool
 * results live in a following `user` message as `tool_result` blocks.
 * We translate to that shape here so stored documents are consistent
 * across integrations.
 */
function buildAnthropicStructuredMessages(
  messages: any[],
  pluginConfig: PluginConfig
): Array<{ role: string; content: any[]; timestamp?: string }> {
  const allowedRoles = new Set(pluginConfig.retainRoles || ["user", "assistant"]);
  const out: Array<{ role: string; content: any[]; timestamp?: string }> = [];

  for (const msg of messages) {
    const rawRole = msg?.role;
    if (rawRole === "toolResult") {
      const toolResultBlock = buildToolResultBlock(msg);
      if (!toolResultBlock) continue;
      // Fold tool_result into a synthetic user message (Anthropic convention),
      // merging with an immediately-preceding synthetic user if one exists so
      // consecutive tool results stay together.
      const last = out[out.length - 1];
      if (
        last &&
        last.role === "user" &&
        last.content.every((b: any) => b.type === "tool_result")
      ) {
        last.content.push(toolResultBlock);
      } else {
        out.push({ role: "user", content: [toolResultBlock] });
      }
      continue;
    }

    if (!allowedRoles.has(rawRole)) continue;

    const blocks = extractStructuredBlocks(msg.content, rawRole);
    if (blocks.length > 0) {
      const timestamp = normalizeMessageTimestamp(msg);
      out.push(
        timestamp
          ? { role: rawRole, content: blocks, timestamp }
          : { role: rawRole, content: blocks }
      );
    }
  }

  return out;
}

function normalizeMessageTimestamp(msg: any): string | undefined {
  const raw = msg?.timestamp;
  if (raw === undefined || raw === null) return undefined;
  const date =
    typeof raw === "number" ? new Date(raw) : typeof raw === "string" ? new Date(raw) : undefined;
  if (!date || Number.isNaN(date.getTime())) return undefined;
  return date.toISOString();
}

function extractStructuredBlocks(content: any, role: string): any[] {
  if (typeof content === "string") {
    const cleaned = stripInlineTimestampPrefix(
      stripMetadataEnvelopes(stripInlineRetainTags(stripMemoryTags(content)))
    ).trim();
    return cleaned ? [{ type: "text", text: cleaned }] : [];
  }
  if (!Array.isArray(content)) return [];

  const blocks: any[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const blockType = block.type;

    if (blockType === "text") {
      const cleaned = stripInlineTimestampPrefix(
        stripMetadataEnvelopes(stripInlineRetainTags(stripMemoryTags(block.text ?? "")))
      ).trim();
      if (cleaned) blocks.push({ type: "text", text: cleaned });
    } else if (blockType === "toolCall" && role === "assistant") {
      const name = typeof block.name === "string" ? block.name : "unknown";
      // Skip Hindsight's own MCP operational tools to avoid feedback loops.
      if (name.startsWith("mcp__") && OPERATIONAL_TOOL_PATTERN.test(name.split("__").pop() ?? ""))
        continue;
      const input = block.arguments && typeof block.arguments === "object" ? block.arguments : {};
      const id = typeof block.id === "string" ? block.id : undefined;
      const toolUse: any = { type: "tool_use", name, input };
      if (id) toolUse.id = id;
      blocks.push(toolUse);
    }
    // thinking / unknown types are dropped
  }
  return blocks;
}

function buildToolResultBlock(msg: any): any | null {
  const toolUseId = typeof msg.toolCallId === "string" ? msg.toolCallId : "";
  const raw = msg.content;
  let text = "";
  if (typeof raw === "string") {
    text = raw;
  } else if (Array.isArray(raw)) {
    text = raw
      .filter((b: any) => b && b.type === "text" && typeof b.text === "string")
      .map((b: any) => b.text)
      .join("\n");
  }
  text = text.trim();
  if (!text) return null;
  if (text.length > TOOL_RESULT_MAX_CHARS) {
    text = text.slice(0, TOOL_RESULT_MAX_CHARS) + "... (truncated)";
  }
  const block: any = { type: "tool_result", content: text };
  if (toolUseId) block.tool_use_id = toolUseId;
  return block;
}

export function countUserTurns(messages: any[]): number {
  if (!Array.isArray(messages) || messages.length === 0) {
    return 0;
  }

  return messages.reduce(
    (count: number, message: any) => count + (message?.role === "user" ? 1 : 0),
    0
  );
}

export function getRetentionTurnIndex(
  conversationTurnCount: number,
  retainEveryN: number
): number | null {
  if (conversationTurnCount <= 0 || retainEveryN <= 0) {
    return null;
  }

  if (retainEveryN === 1) {
    return conversationTurnCount;
  }

  if (conversationTurnCount % retainEveryN !== 0) {
    return null;
  }

  return Math.floor(conversationTurnCount / retainEveryN);
}

export function sliceLastTurnsByUserBoundary(messages: any[], turns: number): any[] {
  if (!Array.isArray(messages) || messages.length === 0 || turns <= 0) {
    return [];
  }

  let userTurnsSeen = 0;
  let startIndex = -1;

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      userTurnsSeen += 1;
      if (userTurnsSeen >= turns) {
        startIndex = i;
        break;
      }
    }
  }

  if (startIndex === -1) {
    return messages;
  }

  return messages.slice(startIndex);
}

export function getClient() {
  return client;
}
