import type { MoltbotPluginAPI, PluginConfig, PluginHookAgentContext, MemoryResult } from './types.js';
import { HindsightEmbedManager } from './embed-manager.js';
import { HindsightClient, type HindsightClientOptions } from './client.js';
import { createHash } from 'crypto';
import { dirname } from 'path';
import { fileURLToPath } from 'url';
import * as log from './logger.js';
import { configureLogger, setApiLogger, stopLogger } from './logger.js';

// Debug logging: silent by default, enable with debug: true or logLevel: 'debug'
let debugEnabled = false;
const debug = (...args: unknown[]) => {
  if (debugEnabled) log.verbose(args.map(a => typeof a === 'string' ? a.replace(/^\[Hindsight\]\s*/, '') : String(a)).join(' '));
};

// Module-level state
let embedManager: HindsightEmbedManager | null = null;
let client: HindsightClient | null = null;
let clientOptions: HindsightClientOptions | null = null;
let initPromise: Promise<void> | null = null;
let isInitialized = false;
let usingExternalApi = false; // Track if using external API (skip daemon management)

// Store the current plugin config for bank ID derivation
let currentPluginConfig: PluginConfig | null = null;

// Track which banks have had their mission set (to avoid re-setting on every request)
const banksWithMissionSet = new Set<string>();
// Use dedicated client instances per bank to avoid cross-session bankId mutation races.
const clientsByBankId = new Map<string, HindsightClient>();
const MAX_TRACKED_BANK_CLIENTS = 10_000;

// In-flight recall deduplication: concurrent recalls for the same bank reuse one promise
import type { RecallResponse } from './types.js';
const inflightRecalls = new Map<string, Promise<RecallResponse>>();
const turnCountBySession = new Map<string, number>();
const MAX_TRACKED_SESSIONS = 10_000;
const DEFAULT_RECALL_TIMEOUT_MS = 10_000;

// Cache sender IDs discovered in before_prompt_build (where event.prompt has the metadata
// blocks) so agent_end can look them up — event.messages in agent_end is clean history.
const senderIdBySession = new Map<string, string>();

// Guard against double hook registration on the same api instance
// Uses a WeakSet so each api instance can only register hooks once
const registeredApis = new WeakSet<object>();

// Cooldown + guard to prevent concurrent reinit attempts
let lastReinitAttempt = 0;
let isReinitInProgress = false;
const REINIT_COOLDOWN_MS = 30_000;

const DEFAULT_RECALL_PROMPT_PREAMBLE =
  'Relevant memories from past conversations (prioritize recent when conflicting). Only use memories that are directly useful to continue this conversation; ignore the rest:';

function formatCurrentTimeForRecall(date = new Date()): string {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}`;
}

/**
 * Lazy re-initialization after startup failure.
 * Called by waitForReady when initPromise rejected but API may now be reachable.
 * Throttled to one attempt per 30s to avoid hammering a down service.
 * Only works if initialization was attempted at least once (isInitialized guard).
 */
async function lazyReinit(): Promise<void> {
  const now = Date.now();
  if (now - lastReinitAttempt < REINIT_COOLDOWN_MS || isReinitInProgress) {
    return;
  }

  // Only attempt lazy reinit if we've already done initial setup
  // (i.e., service.start() was called at least once)
  if (!currentPluginConfig) {
    debug('[Hindsight] lazyReinit skipped - no plugin config (service.start() not called yet)');
    return;
  }

  isReinitInProgress = true;
  lastReinitAttempt = now;

  const config = currentPluginConfig;
  const externalApi = detectExternalApi(config);
  if (!externalApi.apiUrl) {
    isReinitInProgress = false;
    return; // Only external API mode supports lazy reinit
  }

  debug('[Hindsight] Attempting lazy re-initialization...');
  try {
    await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);

    // Health check passed — set up env vars and create client
    process.env.HINDSIGHT_EMBED_API_URL = externalApi.apiUrl;
    if (externalApi.apiToken) {
      process.env.HINDSIGHT_EMBED_API_TOKEN = externalApi.apiToken;
    }

    const llmConfig = detectLLMConfig(config);
    clientOptions = buildClientOptions(llmConfig, config, externalApi);
    clientsByBankId.clear();
    banksWithMissionSet.clear();
    client = new HindsightClient(clientOptions);
    const defaultBankId = deriveBankId(undefined, config);
    client.setBankId(defaultBankId);

    if (config.bankMission && !config.dynamicBankId) {
      await client.setBankMission(config.bankMission);
    }

    usingExternalApi = true;
    isInitialized = true;
    // Replace the rejected initPromise with a resolved one
    initPromise = Promise.resolve();
    debug('[Hindsight] ✓ Lazy re-initialization succeeded');
  } catch (error) {
    log.warn(`lazy re-init failed (retry in ${REINIT_COOLDOWN_MS / 1000}s): ${error instanceof Error ? error.message : error}`);
  } finally {
    isReinitInProgress = false;
  }
}

// Global access for hooks (Moltbot loads hooks separately)
if (typeof global !== 'undefined') {
  (global as any).__hindsightClient = {
    getClient: () => client,
    waitForReady: async () => {
      if (isInitialized) {return;}
      // If initPromise is null, it means service.start() hasn't been called yet
      // (CLI mode, not gateway mode). Hooks should gracefully no-op.
      if (!initPromise) {
        debug('[Hindsight] waitForReady called but initPromise is null (gateway not started)');
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
     * Get a client configured for a specific agent context.
     * Derives the bank ID from the context for per-channel isolation.
     * Also ensures the bank mission is set on first use.
     */
    getClientForContext: async (ctx: PluginHookAgentContext | undefined) => {
      if (!client) {return null;}
      const config = currentPluginConfig || {};
      if (config.dynamicBankId === false) {
        return client;
      }
      const bankId = deriveBankId(ctx, config);
      let bankClient = clientsByBankId.get(bankId);
      if (!bankClient) {
        if (!clientOptions) {
          return null;
        }
        bankClient = new HindsightClient(clientOptions);
        bankClient.setBankId(bankId);
        clientsByBankId.set(bankId, bankClient);
        if (clientsByBankId.size > MAX_TRACKED_BANK_CLIENTS) {
          const oldestKey = clientsByBankId.keys().next().value;
          if (oldestKey) {
            clientsByBankId.delete(oldestKey);
            banksWithMissionSet.delete(oldestKey);
          }
        }
      }

      // Set bank mission on first use of this bank (if configured)
      if (config.bankMission && config.dynamicBankId && !banksWithMissionSet.has(bankId)) {
        try {
          await bankClient.setBankMission(config.bankMission);
          banksWithMissionSet.add(bankId);
          debug(`[Hindsight] Set mission for new bank: ${bankId}`);
        } catch (error) {
          // Log but don't fail - bank mission is not critical
          log.warn(`could not set bank mission for ${bankId}: ${error}`);
        }
      }

      return bankClient;
    },
    getPluginConfig: () => currentPluginConfig,
  };
}

// Get directory of current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Default bank name (fallback when channel context not available)
const DEFAULT_BANK_NAME = 'openclaw';

/**
 * Strip plugin-injected memory tags from content to prevent retain feedback loop.
 * Removes <hindsight_memories> and <relevant_memories> blocks that were injected
 * during before_agent_start so they don't get re-stored into the memory bank.
 */
export function stripMemoryTags(content: string): string {
  content = content.replace(/<hindsight_memories>[\s\S]*?<\/hindsight_memories>/g, '');
  content = content.replace(/<relevant_memories>[\s\S]*?<\/relevant_memories>/g, '');
  return content;
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
      if (id && typeof id === 'string') return id;
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
  content = content.replace(/^---\n[\w\s]+\(untrusted metadata\)[^\n]*\n```json[\s\S]*?```\n\n?/im, '').replace(/\n---$/, '');
  // Strip: <Label> (untrusted metadata):\n```json\n{...}\n```  (without --- wrapper)
  content = content.replace(/[\w\s]+\(untrusted metadata\)[^\n]*\n```json[\s\S]*?```\n?/gim, '');
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
  prompt: string | undefined,
): string | null {
  // Reject known metadata/system message patterns — these are not user queries
  const METADATA_PATTERNS = [
    /^\s*conversation info\s*\(untrusted metadata\)/i,
    /^\s*\(untrusted metadata\)/i,
    /^\s*system:/i,
  ];
  const isMetadata = (s: string) => METADATA_PATTERNS.some(p => p.test(s));

  let recallQuery = rawMessage;
  // Strip sender metadata envelope before any checks
  if (recallQuery) {
    recallQuery = stripMetadataEnvelopes(recallQuery);
  }
  if (!recallQuery || typeof recallQuery !== 'string' || recallQuery.trim().length < 5 || isMetadata(recallQuery)) {
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
    cleaned = cleaned.replace(/^(?:System:.*\n)+\n?/, '');

    // Remove session abort hint
    cleaned = cleaned.replace(
      /^Note: The previous agent run was aborted[^\n]*\n\n/,
      '',
    );

    // Extract message after [ChannelName ...] envelope header
    const envelopeMatch = cleaned.match(
      /\[[A-Z][A-Za-z]*(?:\s[^\]]+)?\]\s*([\s\S]+)$/,
    );
    if (envelopeMatch) {
      cleaned = envelopeMatch[1];
    }

    // Remove trailing [from: SenderName] metadata (group chats)
    cleaned = cleaned.replace(/\n\[from:[^\]]*\]\s*$/, '');

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
  recallRoles: Array<'user' | 'assistant' | 'system' | 'tool'> = ['user', 'assistant'],
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

      let content = '';
      if (typeof msg?.content === 'string') {
        content = msg.content;
      } else if (Array.isArray(msg?.content)) {
        content = msg.content
          .filter((block: any) => block?.type === 'text' && typeof block?.text === 'string')
          .map((block: any) => block.text)
          .join('\n');
      }

      content = stripMemoryTags(content).trim();
      content = stripMetadataEnvelopes(content);
      if (!content) {
        return null;
      }
      if (role === 'user' && content === latest) {
        return null;
      }
      return `${role}: ${content}`;
    })
    .filter((line: string | null): line is string => Boolean(line));

  if (contextLines.length === 0) {
    return latest;
  }

  return [
    'Prior context:',
    contextLines.join('\n'),
    latest,
  ].join('\n\n');
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

  if (!query.includes('Prior context:')) {
    return latestOnly;
  }

  // New order: Prior context at top, latest user message at bottom.
  // Truncate by dropping oldest context lines first to preserve the suffix.
  const contextMarker = 'Prior context:\n\n';
  const markerIndex = query.indexOf(contextMarker);
  if (markerIndex === -1) {
    return latestOnly;
  }

  const suffixMarker = '\n\n' + latest;
  const suffixIndex = query.lastIndexOf(suffixMarker);
  if (suffixIndex === -1) {
    return latestOnly;
  }

  const suffix = query.slice(suffixIndex); // \n\n<latest>
  if (suffix.length >= maxChars) {
    return latestOnly;
  }

  const contextBody = query.slice(markerIndex + contextMarker.length, suffixIndex);
  const contextLines = contextBody.split('\n').filter(Boolean);
  const keptContextLines: string[] = [];

  // Add context lines from newest (bottom) to oldest (top), stopping when we exceed maxChars
  for (let i = contextLines.length - 1; i >= 0; i--) {
    keptContextLines.unshift(contextLines[i]);
    const candidate = `${contextMarker}${keptContextLines.join('\n')}${suffix}`;
    if (candidate.length > maxChars) {
      keptContextLines.shift();
      break;
    }
  }

  if (keptContextLines.length > 0) {
    return `${contextMarker}${keptContextLines.join('\n')}${suffix}`;
  }

  return latestOnly;
}

/**
 * Derive a bank ID from the agent context.
 * Uses configurable dynamicBankGranularity to determine bank segmentation.
 * Falls back to default bank when context is unavailable.
 */
/**
 * Parse the OpenClaw sessionKey to extract context fields.
 * Format: "agent:{agentId}:{provider}:{channelType}:{channelId}[:{extra}]"
 * Example: "agent:c0der:telegram:group:-1003825475854:topic:42"
 */
function parseSessionKey(sessionKey: string): { agentId?: string; provider?: string; channel?: string } {
  const parts = sessionKey.split(':');
  if (parts.length < 5 || parts[0] !== 'agent') return {};
  // parts[1] = agentId, parts[2] = provider, parts[3] = channelType, parts[4..] = channelId + extras
  return {
    agentId: parts[1],
    provider: parts[2],
    // Rejoin from channelType onward as the channel identifier (e.g. "group:-1003825475854:topic:42")
    channel: parts.slice(3).join(':'),
  };
}

export function deriveBankId(ctx: PluginHookAgentContext | undefined, pluginConfig: PluginConfig): string {
  if (pluginConfig.dynamicBankId === false) {
    return pluginConfig.bankIdPrefix ? `${pluginConfig.bankIdPrefix}-openclaw` : 'openclaw';
  }

  // When no context is available, fall back to the static default bank.
  if (!ctx) {
    return pluginConfig.bankIdPrefix ? `${pluginConfig.bankIdPrefix}-openclaw` : 'openclaw';
  }

  const fields = pluginConfig.dynamicBankGranularity?.length ? pluginConfig.dynamicBankGranularity : ['agent', 'channel', 'user'];

  // Validate field names at runtime — typos silently produce 'unknown' segments
  const validFields = new Set(['agent', 'channel', 'user', 'provider']);
  for (const f of fields) {
    if (!validFields.has(f)) {
      log.warn(`unknown dynamicBankGranularity field "${f}" — will resolve to "unknown". Valid: agent, channel, user, provider`);
    }
  }

  // Parse sessionKey as fallback when direct context fields are missing
  const sessionParsed = ctx?.sessionKey ? parseSessionKey(ctx.sessionKey) : {};

  // Warn when 'user' is in active fields but senderId is missing — bank ID will contain "anonymous"
  if (fields.includes('user') && ctx && !ctx.senderId) {
    debug('[Hindsight] senderId not available in context — bank ID will use "anonymous". Ensure your OpenClaw provider passes senderId.');
  }

  const fieldMap: Record<string, string> = {
    agent: ctx?.agentId || sessionParsed.agentId || 'default',
    channel: ctx?.channelId || sessionParsed.channel || 'unknown',
    user: ctx?.senderId || 'anonymous',
    provider: ctx?.messageProvider || sessionParsed.provider || 'unknown',
  };

  const baseBankId = fields
    .map(f => encodeURIComponent(fieldMap[f] || 'unknown'))
    .join('::');

  return pluginConfig.bankIdPrefix
    ? `${pluginConfig.bankIdPrefix}-${baseBankId}`
    : baseBankId;
}


export function formatMemories(results: MemoryResult[]): string {
  if (!results || results.length === 0) return '';
  return results
    .map(r => {
      const type = r.type ? ` [${r.type}]` : '';
      const date = r.mentioned_at ? ` (${r.mentioned_at})` : '';
      return `- ${r.text}${type}${date}`;
    })
    .join('\n\n');
}


// Provider detection from standard env vars
const PROVIDER_DETECTION = [
  { name: 'openai', keyEnv: 'OPENAI_API_KEY' },
  { name: 'anthropic', keyEnv: 'ANTHROPIC_API_KEY' },
  { name: 'gemini', keyEnv: 'GEMINI_API_KEY' },
  { name: 'groq', keyEnv: 'GROQ_API_KEY' },
  { name: 'ollama', keyEnv: '' },
  { name: 'openai-codex', keyEnv: '' },
  { name: 'claude-code', keyEnv: '' },
];

function detectLLMConfig(pluginConfig?: PluginConfig): {
  provider?: string;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  source: string;
} {
  // Override values from HINDSIGHT_API_LLM_* env vars (highest priority)
  const overrideProvider = process.env.HINDSIGHT_API_LLM_PROVIDER;
  const overrideModel = process.env.HINDSIGHT_API_LLM_MODEL;
  const overrideKey = process.env.HINDSIGHT_API_LLM_API_KEY;
  const overrideBaseUrl = process.env.HINDSIGHT_API_LLM_BASE_URL;

  // Priority 1: If provider is explicitly set via env var, use that
  if (overrideProvider) {
    // Providers that don't require an API key (use OAuth or local models)
    const noKeyRequired = ['ollama', 'openai-codex', 'claude-code'];
    if (!overrideKey && !noKeyRequired.includes(overrideProvider)) {
      throw new Error(
        `HINDSIGHT_API_LLM_PROVIDER is set to "${overrideProvider}" but HINDSIGHT_API_LLM_API_KEY is not set.\n` +
        `Please set: export HINDSIGHT_API_LLM_API_KEY=your-api-key`
      );
    }

    return {
      provider: overrideProvider,
      apiKey: overrideKey || '',
      model: overrideModel,
      baseUrl: overrideBaseUrl,
      source: 'HINDSIGHT_API_LLM_PROVIDER override',
    };
  }

  // Priority 2: Plugin config llmProvider/llmModel
  if (pluginConfig?.llmProvider) {
    const providerInfo = PROVIDER_DETECTION.find(p => p.name === pluginConfig.llmProvider);

    // Resolve API key: llmApiKeyEnv > provider's standard keyEnv
    let apiKey = '';
    if (pluginConfig.llmApiKeyEnv) {
      apiKey = process.env[pluginConfig.llmApiKeyEnv] || '';
    } else if (providerInfo?.keyEnv) {
      apiKey = process.env[providerInfo.keyEnv] || '';
    }

    // Providers that don't require an API key (use OAuth or local models)
    const noKeyRequired = ['ollama', 'openai-codex', 'claude-code'];
    if (!apiKey && !noKeyRequired.includes(pluginConfig.llmProvider)) {
      const keySource = pluginConfig.llmApiKeyEnv || providerInfo?.keyEnv || 'unknown';
      throw new Error(
        `Plugin config llmProvider is set to "${pluginConfig.llmProvider}" but no API key found.\n` +
        `Expected env var: ${keySource}\n` +
        `Set the env var or use llmApiKeyEnv in plugin config to specify a custom env var name.`
      );
    }

    return {
      provider: pluginConfig.llmProvider,
      apiKey,
      model: pluginConfig.llmModel || overrideModel,
      baseUrl: overrideBaseUrl,
      source: 'plugin config',
    };
  }

  // Priority 3: Auto-detect from standard provider env vars
  for (const providerInfo of PROVIDER_DETECTION) {
    const apiKey = providerInfo.keyEnv ? process.env[providerInfo.keyEnv] : '';

    // Skip providers that don't use API keys in auto-detection (must be explicitly requested)
    const noKeyRequired = ['ollama', 'openai-codex', 'claude-code'];
    if (noKeyRequired.includes(providerInfo.name)) {
      continue;
    }

    if (apiKey) {
      return {
        provider: providerInfo.name,
        apiKey,
        model: overrideModel,
        baseUrl: overrideBaseUrl,
        source: `auto-detected from ${providerInfo.keyEnv}`,
      };
    }
  }

  // No configuration found - show helpful error

  // Allow empty LLM config if using external Hindsight API (server handles LLM)
  const externalApiCheck = detectExternalApi(pluginConfig);
  if (externalApiCheck.apiUrl) {
    return {
      provider: undefined,
      apiKey: undefined,
      model: undefined,
      baseUrl: undefined,
      source: 'external-api-mode-no-llm',
    };
  }

  throw new Error(
    `No LLM configuration found for Hindsight memory plugin.\n\n` +
    `Option 1: Set a standard provider API key (auto-detect):\n` +
    `  export OPENAI_API_KEY=sk-your-key\n` +
    `  export ANTHROPIC_API_KEY=your-key\n` +
    `  export GEMINI_API_KEY=your-key\n` +
    `  export GROQ_API_KEY=your-key\n\n` +
    `Option 2: Use Codex or Claude Code (no API key needed):\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=openai-codex    # Requires 'codex auth login'\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=claude-code     # Requires Claude Code CLI\n\n` +
    `Option 3: Set llmProvider in openclaw.json plugin config:\n` +
    `  "llmProvider": "openai"\n\n` +
    `Option 4: Override with Hindsight-specific env vars:\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=openai\n` +
    `  export HINDSIGHT_API_LLM_API_KEY=sk-your-key\n` +
    `  export HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1  # Optional\n\n` +
    `The model will be selected automatically by Hindsight. To override: export HINDSIGHT_API_LLM_MODEL=your-model`
  );
}

/**
 * Detect external Hindsight API configuration.
 * Priority: env vars > plugin config
 */
function detectExternalApi(pluginConfig?: PluginConfig): {
  apiUrl: string | null;
  apiToken: string | null;
} {
  const apiUrl = process.env.HINDSIGHT_EMBED_API_URL || pluginConfig?.hindsightApiUrl || null;
  const apiToken = process.env.HINDSIGHT_EMBED_API_TOKEN || pluginConfig?.hindsightApiToken || null;
  return { apiUrl, apiToken };
}

/**
 * Build HindsightClientOptions from LLM config, plugin config, and external API settings.
 */
function buildClientOptions(
  llmConfig: { provider?: string; apiKey?: string; model?: string },
  pluginCfg: PluginConfig,
  externalApi: { apiUrl: string | null; apiToken: string | null },
): HindsightClientOptions {
  return {
    llmModel: llmConfig.model,
    embedVersion: pluginCfg.embedVersion,
    embedPackagePath: pluginCfg.embedPackagePath,
    apiUrl: externalApi.apiUrl ?? undefined,
    apiToken: externalApi.apiToken ?? undefined,
  };
}

/**
 * Health check for external Hindsight API.
 * Retries up to 3 times with 2s delay — container DNS may not be ready on first boot.
 */
async function checkExternalApiHealth(apiUrl: string, apiToken?: string | null): Promise<void> {
  const healthUrl = `${apiUrl.replace(/\/$/, '')}/health`;
  const maxRetries = 3;
  const retryDelay = 2000;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      debug(`[Hindsight] Checking external API health at ${healthUrl}... (attempt ${attempt}/${maxRetries})`);
      const headers: Record<string, string> = {};
      if (apiToken) {
        headers['Authorization'] = `Bearer ${apiToken}`;
      }
      const response = await fetch(healthUrl, { signal: AbortSignal.timeout(10000), headers });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json() as { status?: string };
      debug(`[Hindsight] External API health: ${JSON.stringify(data)}`);
      return;
    } catch (error) {
      if (attempt < maxRetries) {
        debug(`[Hindsight] Health check attempt ${attempt} failed, retrying in ${retryDelay}ms...`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      } else {
        throw new Error(`Cannot connect to external Hindsight API at ${apiUrl}: ${error}`, { cause: error });
      }
    }
  }
}

function getPluginConfig(api: MoltbotPluginAPI): PluginConfig {
  const config = api.config.plugins?.entries?.['hindsight-openclaw']?.config || {};
  const defaultMission = 'You are an AI assistant helping users across multiple communication channels (Telegram, Slack, Discord, etc.). Remember user preferences, instructions, and important context from conversations to provide personalized assistance.';

  return {
    bankMission: config.bankMission || defaultMission,
    embedPort: config.embedPort || 0,
    daemonIdleTimeout: config.daemonIdleTimeout !== undefined ? config.daemonIdleTimeout : 0,
    embedVersion: config.embedVersion || 'latest',
    embedPackagePath: config.embedPackagePath,
    llmProvider: config.llmProvider,
    llmModel: config.llmModel,
    llmApiKeyEnv: config.llmApiKeyEnv,
    hindsightApiUrl: config.hindsightApiUrl,
    hindsightApiToken: config.hindsightApiToken,
    apiPort: config.apiPort || 9077,
    // Dynamic bank ID options (default: enabled)
    dynamicBankId: config.dynamicBankId !== false,
    bankIdPrefix: config.bankIdPrefix,
    excludeProviders: Array.isArray(config.excludeProviders) ? config.excludeProviders : [],
    autoRecall: config.autoRecall !== false, // Default: true (on) — backward compatible
    dynamicBankGranularity: Array.isArray(config.dynamicBankGranularity) ? config.dynamicBankGranularity : undefined,
    autoRetain: config.autoRetain !== false, // Default: true
    retainRoles: Array.isArray(config.retainRoles) ? config.retainRoles : undefined,
    recallBudget: config.recallBudget || 'mid',
    recallMaxTokens: config.recallMaxTokens || 1024,
    recallTypes: Array.isArray(config.recallTypes) ? config.recallTypes : ['world', 'experience'],
    recallRoles: Array.isArray(config.recallRoles) ? config.recallRoles : ['user', 'assistant'],
    retainEveryNTurns: typeof config.retainEveryNTurns === 'number' && config.retainEveryNTurns >= 1 ? config.retainEveryNTurns : 1,
    retainOverlapTurns: typeof config.retainOverlapTurns === 'number' && config.retainOverlapTurns >= 0 ? config.retainOverlapTurns : 0,
    recallTopK: typeof config.recallTopK === 'number' ? config.recallTopK : undefined,
    recallContextTurns: typeof config.recallContextTurns === 'number' && config.recallContextTurns >= 1 ? config.recallContextTurns : 1,
    recallMaxQueryChars: typeof config.recallMaxQueryChars === 'number' && config.recallMaxQueryChars >= 1 ? config.recallMaxQueryChars : 800,
    recallPromptPreamble:
      typeof config.recallPromptPreamble === 'string' && config.recallPromptPreamble.trim().length > 0
        ? config.recallPromptPreamble
        : DEFAULT_RECALL_PROMPT_PREAMBLE,
    recallInjectionPosition: typeof config.recallInjectionPosition === 'string' && ['prepend', 'append', 'user'].includes(config.recallInjectionPosition) ? config.recallInjectionPosition as PluginConfig['recallInjectionPosition'] : undefined,
    recallTimeoutMs: typeof config.recallTimeoutMs === 'number' && config.recallTimeoutMs >= 1000 ? config.recallTimeoutMs : undefined,
    debug: config.debug ?? false,
  };
}

export default function (api: MoltbotPluginAPI) {
  try {
    debug('[Hindsight] Plugin loading...');

    // Get plugin config first (needed for debug flag and service registration)
    const pluginConfig = getPluginConfig(api);
    // If logLevel is 'debug', also enable legacy debug flag
    debugEnabled = pluginConfig.debug ?? (pluginConfig.logLevel === 'debug');

    // Configure structured logger — route through OpenClaw's api.logger for consistent formatting
    if (api.logger) setApiLogger(api.logger);
    configureLogger({
      logLevel: pluginConfig.logLevel ?? (pluginConfig.debug ? 'debug' : 'info'),
      logSummaryIntervalMs: pluginConfig.logSummaryIntervalMs,
    });

    // Store config globally for bank ID derivation in hooks
    currentPluginConfig = pluginConfig;

    debug('[Hindsight] Plugin loaded successfully (deferred heavy init to gateway start)');

    // Register background service for cleanup
    // IMPORTANT: Heavy initialization (LLM detection, daemon start, API health checks)
    // happens in service.start() which is ONLY called on gateway start,
    // not on every CLI command.
    debug('[Hindsight] Registering service...');
    api.registerService({
      id: 'hindsight-memory',
      async start() {
        debug('[Hindsight] Service start called - beginning heavy initialization...');

        // Detect LLM configuration (env vars > plugin config > auto-detect)
        debug('[Hindsight] Detecting LLM config...');
        const llmConfig = detectLLMConfig(pluginConfig);

        const baseUrlInfo = llmConfig.baseUrl ? `, base URL: ${llmConfig.baseUrl}` : '';
        const modelInfo = llmConfig.model || 'default';

        if (llmConfig.provider === 'ollama') {
          debug(`[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${modelInfo} (${llmConfig.source})`);
        } else {
          debug(`[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${modelInfo} (${llmConfig.source}${baseUrlInfo})`);
        }
        if (pluginConfig.bankMission) {
          debug(`[Hindsight] Custom bank mission configured: "${pluginConfig.bankMission.substring(0, 50)}..."`);
        }

        // Log dynamic bank ID mode
        if (pluginConfig.dynamicBankId) {
          const prefixInfo = pluginConfig.bankIdPrefix ? ` (prefix: ${pluginConfig.bankIdPrefix})` : '';
          debug(`[Hindsight] ✓ Dynamic bank IDs enabled${prefixInfo} - each channel gets isolated memory`);
        } else {
          debug(`[Hindsight] Dynamic bank IDs disabled - using static bank: ${DEFAULT_BANK_NAME}`);
        }

        // Detect external API mode
        const externalApi = detectExternalApi(pluginConfig);

        // Get API port from config (default: 9077)
        const apiPort = pluginConfig.apiPort || 9077;

        if (externalApi.apiUrl) {
          // External API mode - skip local daemon
          usingExternalApi = true;
          debug(`[Hindsight] ✓ Using external API: ${externalApi.apiUrl}`);

          // Set env vars so CLI commands (uvx hindsight-embed) use external API
          process.env.HINDSIGHT_EMBED_API_URL = externalApi.apiUrl;
          if (externalApi.apiToken) {
            process.env.HINDSIGHT_EMBED_API_TOKEN = externalApi.apiToken;
            debug('[Hindsight] API token configured');
          }
        } else {
          debug(`[Hindsight] Daemon idle timeout: ${pluginConfig.daemonIdleTimeout}s (0 = never timeout)`);
          debug(`[Hindsight] API Port: ${apiPort}`);
        }

        // Initialize (runs synchronously in service.start())
        debug('[Hindsight] Starting initialization...');
        initPromise = (async () => {
          try {
            if (usingExternalApi && externalApi.apiUrl) {
              // External API mode - check health, skip daemon startup
              debug('[Hindsight] External API mode - skipping local daemon...');
              await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);

              // Initialize client with direct HTTP mode
              debug('[Hindsight] Creating HindsightClient (HTTP mode)...');
              clientOptions = buildClientOptions(llmConfig, pluginConfig, externalApi);
              clientsByBankId.clear();
              banksWithMissionSet.clear();
              client = new HindsightClient(clientOptions);

              // Set default bank (will be overridden per-request when dynamic bank IDs are enabled)
              const defaultBankId = deriveBankId(undefined, pluginConfig);
              debug(`[Hindsight] Default bank: ${defaultBankId}`);
              client.setBankId(defaultBankId);

              // Note: Bank mission will be set per-bank when dynamic bank IDs are enabled
              // For now, set it on the default bank
              if (pluginConfig.bankMission && !pluginConfig.dynamicBankId) {
                debug(`[Hindsight] Setting bank mission...`);
                await client.setBankMission(pluginConfig.bankMission);
              }

              if (!isInitialized) {
                const mode = 'external API';
                const autoRecall = pluginConfig.autoRecall !== false;
                const autoRetain = pluginConfig.autoRetain !== false;
                log.info(`initialized (mode: ${mode}, bank: ${defaultBankId}, autoRecall: ${autoRecall}, autoRetain: ${autoRetain})`);
              }
              isInitialized = true;
              debug('[Hindsight] ✓ Ready (external API mode)');
            } else {
              // Local daemon mode - start hindsight-embed daemon
              debug('[Hindsight] Creating HindsightEmbedManager...');
              embedManager = new HindsightEmbedManager(
                apiPort,
                llmConfig.provider || "",
                llmConfig.apiKey || "",
                llmConfig.model,
                llmConfig.baseUrl,
                pluginConfig.daemonIdleTimeout,
                pluginConfig.embedVersion,
                pluginConfig.embedPackagePath
              );

              // Start the embedded server
              debug('[Hindsight] Starting embedded server...');
              await embedManager.start();

              // Initialize client (local daemon mode — no apiUrl)
              debug('[Hindsight] Creating HindsightClient (subprocess mode)...');
              clientOptions = buildClientOptions(llmConfig, pluginConfig, { apiUrl: null, apiToken: null });
              clientsByBankId.clear();
              banksWithMissionSet.clear();
              client = new HindsightClient(clientOptions);

              // Set default bank (will be overridden per-request when dynamic bank IDs are enabled)
              const defaultBankId = deriveBankId(undefined, pluginConfig);
              debug(`[Hindsight] Default bank: ${defaultBankId}`);
              client.setBankId(defaultBankId);

              // Note: Bank mission will be set per-bank when dynamic bank IDs are enabled
              // For now, set it on the default bank
              if (pluginConfig.bankMission && !pluginConfig.dynamicBankId) {
                debug(`[Hindsight] Setting bank mission...`);
                await client.setBankMission(pluginConfig.bankMission);
              }

              if (!isInitialized) {
                const mode = 'local daemon';
                const autoRecall = pluginConfig.autoRecall !== false;
                const autoRetain = pluginConfig.autoRetain !== false;
                log.info(`initialized (mode: ${mode}, bank: ${defaultBankId}, autoRecall: ${autoRecall}, autoRetain: ${autoRetain})`);
              }
              isInitialized = true;
              debug('[Hindsight] ✓ Ready');
            }
          } catch (error) {
            log.error('initialization error', error);
            throw error;
          }
        })();

        // Wait for initialization to complete
        try {
          await initPromise;
        } catch (error) {
          log.error('initial initialization failed', error);
          // Continue to health check below
        }

        // External API mode: check external API health
        if (usingExternalApi) {
          const externalApi = detectExternalApi(pluginConfig);
          if (externalApi.apiUrl && isInitialized) {
            try {
              await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);
              debug('[Hindsight] External API is healthy');
              return;
            } catch (error) {
              log.error('external API health check failed', error);
              // Reset state for reinitialization attempt
              client = null;
              clientOptions = null;
              clientsByBankId.clear();
              banksWithMissionSet.clear();
              isInitialized = false;
            }
          }
        } else {
          // Local daemon mode: check daemon health (handles SIGUSR1 restart case)
          if (embedManager && isInitialized) {
            const healthy = await embedManager.checkHealth();
            if (healthy) {
              debug('[Hindsight] Daemon is healthy');
              return;
            }

            debug('[Hindsight] Daemon is not responding - reinitializing...');
            // Reset state for reinitialization
            embedManager = null;
            client = null;
            clientOptions = null;
            clientsByBankId.clear();
            banksWithMissionSet.clear();
            isInitialized = false;
          }
        }

        // Reinitialize if needed (fresh start or recovery)
        if (!isInitialized) {
          debug('[Hindsight] Reinitializing...');
          const reinitPluginConfig = getPluginConfig(api);
          currentPluginConfig = reinitPluginConfig;
          const llmConfig = detectLLMConfig(reinitPluginConfig);
          const externalApi = detectExternalApi(reinitPluginConfig);
          const apiPort = reinitPluginConfig.apiPort || 9077;

          if (externalApi.apiUrl) {
            // External API mode
            usingExternalApi = true;
            process.env.HINDSIGHT_EMBED_API_URL = externalApi.apiUrl;
            if (externalApi.apiToken) {
              process.env.HINDSIGHT_EMBED_API_TOKEN = externalApi.apiToken;
            }

            await checkExternalApiHealth(externalApi.apiUrl, externalApi.apiToken);

            clientOptions = buildClientOptions(llmConfig, reinitPluginConfig, externalApi);
            clientsByBankId.clear();
            banksWithMissionSet.clear();
            client = new HindsightClient(clientOptions);
            const defaultBankId = deriveBankId(undefined, reinitPluginConfig);
            client.setBankId(defaultBankId);

            if (reinitPluginConfig.bankMission && !reinitPluginConfig.dynamicBankId) {
              await client.setBankMission(reinitPluginConfig.bankMission);
            }

            isInitialized = true;
            debug('[Hindsight] Reinitialization complete (external API mode)');
          } else {
            // Local daemon mode
            embedManager = new HindsightEmbedManager(
              apiPort,
              llmConfig.provider || "",
              llmConfig.apiKey || "",
              llmConfig.model,
              llmConfig.baseUrl,
              reinitPluginConfig.daemonIdleTimeout,
              reinitPluginConfig.embedVersion,
              reinitPluginConfig.embedPackagePath
            );

            await embedManager.start();

            clientOptions = buildClientOptions(llmConfig, reinitPluginConfig, { apiUrl: null, apiToken: null });
            clientsByBankId.clear();
            banksWithMissionSet.clear();
            client = new HindsightClient(clientOptions);
            const defaultBankId = deriveBankId(undefined, reinitPluginConfig);
            client.setBankId(defaultBankId);

            if (reinitPluginConfig.bankMission && !reinitPluginConfig.dynamicBankId) {
              await client.setBankMission(reinitPluginConfig.bankMission);
            }

            isInitialized = true;
            debug('[Hindsight] Reinitialization complete');
          }
        }
      },

      async stop() {
        try {
          debug('[Hindsight] Service stopping...');

          // Only stop daemon if in local mode
          if (!usingExternalApi && embedManager) {
            await embedManager.stop();
            embedManager = null;
          }

          client = null;
          clientOptions = null;
          clientsByBankId.clear();
          banksWithMissionSet.clear();
          isInitialized = false;

          stopLogger();
          debug('[Hindsight] Service stopped');
        } catch (error) {
          log.error('service stop error', error);
          throw error;
        }
      },
    });

    debug('[Hindsight] Plugin loaded successfully');

    // Register agent hooks for auto-recall and auto-retention
    if (registeredApis.has(api)) {
      debug('[Hindsight] Hooks already registered for this api instance, skipping duplicate registration');
      return;
    }
    registeredApis.add(api);
    debug('[Hindsight] Registering agent hooks...');

    // Auto-recall: Inject relevant memories before agent processes the message
    // Hook signature: (event, ctx) where event has {prompt, messages?} and ctx has agent context
    api.on('before_prompt_build', async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        // Check if this provider is excluded
        if (ctx?.messageProvider && pluginConfig.excludeProviders?.includes(ctx.messageProvider)) {
          debug(`[Hindsight] Skipping recall for excluded provider: ${ctx.messageProvider}`);
          return;
        }

        // Skip auto-recall when disabled (agent has its own recall tool)
        if (!pluginConfig.autoRecall) {
          debug('[Hindsight] Auto-recall disabled via config, skipping');
          return;
        }

        // Derive bank ID from context — enrich ctx.senderId from the inbound metadata
        // block when it's missing (agent-phase hooks don't carry senderId in ctx directly).
        const senderIdFromPrompt = !ctx?.senderId ? extractSenderIdFromText(event.prompt ?? event.rawMessage ?? '') : undefined;
        const effectiveCtxForRecall = senderIdFromPrompt ? { ...ctx, senderId: senderIdFromPrompt } : ctx;

        // Cache the resolved sender ID keyed by sessionKey so agent_end can use it.
        // event.messages in agent_end is clean history without the metadata blocks.
        const resolvedSenderId = effectiveCtxForRecall?.senderId;
        const sessionKeyForCache = ctx?.sessionKey;
        if (resolvedSenderId && sessionKeyForCache) {
          senderIdBySession.set(sessionKeyForCache, resolvedSenderId);
          if (senderIdBySession.size > MAX_TRACKED_SESSIONS) {
            const oldest = senderIdBySession.keys().next().value;
            if (oldest) senderIdBySession.delete(oldest);
          }
        }

        const bankId = deriveBankId(effectiveCtxForRecall, pluginConfig);
        debug(`[Hindsight] before_prompt_build - bank: ${bankId}, channel: ${ctx?.messageProvider}/${ctx?.channelId}`);
        debug(`[Hindsight] event keys: ${Object.keys(event).join(', ')}`);
        debug(`[Hindsight] event.context keys: ${Object.keys(event.context ?? {}).join(', ')}`);

        // Get the user's latest message for recall — only the raw user text, not the full prompt
        // rawMessage is clean user text; prompt includes envelope, system events, media notes, etc.
        debug(`[Hindsight] extractRecallQuery input lengths - raw: ${event.rawMessage?.length ?? 0}, prompt: ${event.prompt?.length ?? 0}`);
        const extracted = extractRecallQuery(event.rawMessage, event.prompt);
        if (!extracted) {
          debug('[Hindsight] extractRecallQuery returned null, skipping recall');
          return;
        }
        debug(`[Hindsight] extractRecallQuery result length: ${extracted.length}`);
        const recallContextTurns = pluginConfig.recallContextTurns ?? 1;
        const recallMaxQueryChars = pluginConfig.recallMaxQueryChars ?? 800;
        const sessionMessages = event.context?.sessionEntry?.messages ?? event.messages ?? [];
        const messageCount = sessionMessages.length;
        debug(`[Hindsight] event.messages count: ${messageCount}, roles: ${sessionMessages.map((m: any) => m.role).join(',')}`);
        if (recallContextTurns > 1 && messageCount === 0) {
          debug('[Hindsight] recallContextTurns > 1 but event.messages is empty — prior context unavailable at before_agent_start for this provider');
        }
        const recallRoles = pluginConfig.recallRoles ?? ['user', 'assistant'];
        const composedPrompt = composeRecallQuery(extracted, sessionMessages, recallContextTurns, recallRoles);
        let prompt = truncateRecallQuery(composedPrompt, extracted, recallMaxQueryChars);

        // Final defensive cap
        if (prompt.length > recallMaxQueryChars) {
          prompt = prompt.substring(0, recallMaxQueryChars);
        }

        // Wait for client to be ready
        const clientGlobal = (global as any).__hindsightClient;
        if (!clientGlobal) {
          debug('[Hindsight] Client global not available, skipping auto-recall');
          return;
        }

        await clientGlobal.waitForReady();

        // Get client configured for this context's bank (async to handle mission setup)
        const client = await clientGlobal.getClientForContext(effectiveCtxForRecall);
        if (!client) {
          debug('[Hindsight] Client not initialized, skipping auto-recall');
          return;
        }

        debug(`[Hindsight] Auto-recall for bank ${bankId}, full query:\n---\n${prompt}\n---`);

        // Recall with deduplication: reuse in-flight request for same bank
        const normalizedPrompt = prompt.trim().toLowerCase().replace(/\s+/g, ' ');
        const queryHash = createHash('sha256').update(normalizedPrompt).digest('hex').slice(0, 16);
        const recallKey = `${bankId}::${queryHash}`;
        const existing = inflightRecalls.get(recallKey);
        let recallPromise: Promise<RecallResponse>;
        if (existing) {
          debug(`[Hindsight] Reusing in-flight recall for bank ${bankId}`);
          recallPromise = existing;
        } else {
          const recallTimeoutMs = pluginConfig.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS;
          recallPromise = client.recall({ query: prompt, max_tokens: pluginConfig.recallMaxTokens || 1024, budget: pluginConfig.recallBudget, types: pluginConfig.recallTypes }, recallTimeoutMs);
          inflightRecalls.set(recallKey, recallPromise);
          void recallPromise.catch(() => {}).finally(() => inflightRecalls.delete(recallKey));
        }

        const response = await recallPromise;

        if (!response.results || response.results.length === 0) {
          debug('[Hindsight] No memories found for auto-recall');
          return;
        }

        debug(`[Hindsight] Raw recall response (${response.results.length} results before topK):\n${response.results.map((r: any, i: number) => `  [${i}] score=${r.score?.toFixed(3) ?? 'n/a'} type=${r.type ?? 'n/a'}: ${JSON.stringify(r.content ?? r.text ?? r).substring(0, 200)}`).join('\n')}`);

        const results = pluginConfig.recallTopK ? response.results.slice(0, pluginConfig.recallTopK) : response.results;

        debug(`[Hindsight] After topK (${pluginConfig.recallTopK ?? 'unlimited'}): ${results.length} results injected`);

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
        const position = pluginConfig.recallInjectionPosition || 'prepend';
        switch (position) {
          case 'append':
            return { appendSystemContext: contextMessage };
          case 'user':
            return { prependContext: contextMessage };
          case 'prepend':
          default:
            return { prependSystemContext: contextMessage };
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === 'TimeoutError') {
          log.warn(`[Hindsight] Auto-recall timed out after ${pluginConfig.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS}ms, skipping memory injection`);
        } else if (error instanceof Error && error.name === 'AbortError') {
          log.warn(`[Hindsight] Auto-recall aborted after ${pluginConfig.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS}ms, skipping memory injection`);
        } else {
          log.error('auto-recall error', error);
        }
        return;
      }
    });

    // Hook signature: (event, ctx) where event has {messages, success, error?, durationMs?}
    api.on('agent_end', async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        // Avoid cross-session contamination: only use context carried by this event.
        const eventSessionKey = typeof event?.sessionKey === 'string' ? event.sessionKey : undefined;
        const effectiveCtx = ctx || (eventSessionKey ? ({ sessionKey: eventSessionKey } as PluginHookAgentContext) : undefined);

        // Check if this provider is excluded
        if (effectiveCtx?.messageProvider && pluginConfig.excludeProviders?.includes(effectiveCtx.messageProvider)) {
          debug(`[Hindsight] Skipping retain for excluded provider: ${effectiveCtx.messageProvider}`);
          return;
        }

        // Derive bank ID from context — enrich ctx.senderId from the session cache.
        // event.messages in agent_end is clean history without OpenClaw's metadata blocks;
        // the sender ID was captured during before_prompt_build where event.prompt has them.
        const sessionKeyForLookup = effectiveCtx?.sessionKey;
        const senderIdFromCache = !effectiveCtx?.senderId && sessionKeyForLookup
          ? senderIdBySession.get(sessionKeyForLookup)
          : undefined;
        const effectiveCtxForRetain = senderIdFromCache ? { ...effectiveCtx, senderId: senderIdFromCache } : effectiveCtx;
        const bankId = deriveBankId(effectiveCtxForRetain, pluginConfig);
        debug(`[Hindsight Hook] agent_end triggered - bank: ${bankId}`);

        if (event.success === false) {
          debug('[Hindsight Hook] Agent run failed, skipping retention');
          return;
        }

        if (!Array.isArray(event.context?.sessionEntry?.messages ?? event.messages) || (event.context?.sessionEntry?.messages ?? event.messages ?? []).length === 0) {
          debug('[Hindsight Hook] No messages in event, skipping retention');
          return;
        }

        if (pluginConfig.autoRetain === false) {
          debug('[Hindsight Hook] autoRetain is disabled, skipping retention');
          return;
        }

        // Chunked retention: skip non-Nth turns and use a sliding window when firing
        const retainEveryN = pluginConfig.retainEveryNTurns ?? 1;
        const allMessages = event.context?.sessionEntry?.messages ?? event.messages ?? [];
        let messagesToRetain = allMessages;
        let retainFullWindow = false;

        if (retainEveryN > 1) {
          const sessionTrackingKey = `${bankId}:${effectiveCtx?.sessionKey || 'session'}`;
          const turnCount = (turnCountBySession.get(sessionTrackingKey) || 0) + 1;
          turnCountBySession.set(sessionTrackingKey, turnCount);
          if (turnCountBySession.size > MAX_TRACKED_SESSIONS) {
            const oldestKey = turnCountBySession.keys().next().value;
            if (oldestKey) {
              turnCountBySession.delete(oldestKey);
            }
          }

          if (turnCount % retainEveryN !== 0) {
            const nextRetainAt = Math.ceil(turnCount / retainEveryN) * retainEveryN;
            debug(`[Hindsight Hook] Turn ${turnCount}/${retainEveryN}, skipping retain (next at turn ${nextRetainAt})`);
            return;
          }

          // Sliding window in turns: N turns + configured overlap turns.
          // We slice by actual turn boundaries (user-role messages), so this
          // remains stable even when system/tool messages are present.
          const overlapTurns = pluginConfig.retainOverlapTurns ?? 0;
          const windowTurns = retainEveryN + overlapTurns;
          messagesToRetain = sliceLastTurnsByUserBoundary(allMessages, windowTurns);
          retainFullWindow = true;
          debug(`[Hindsight Hook] Turn ${turnCount}: chunked retain firing (window: ${windowTurns} turns, ${messagesToRetain.length} messages)`);
        }

        const retention = prepareRetentionTranscript(messagesToRetain, pluginConfig, retainFullWindow);
        if (!retention) {
          debug('[Hindsight Hook] No messages to retain (filtered/short/no-user)');
          return;
        }
        const { transcript, messageCount } = retention;

        // Wait for client to be ready
        const clientGlobal = (global as any).__hindsightClient;
        if (!clientGlobal) {
          log.warn('client global not found, skipping retain');
          return;
        }

        await clientGlobal.waitForReady();

        // Get client configured for this context's bank (async to handle mission setup)
        const client = await clientGlobal.getClientForContext(effectiveCtxForRetain);
        if (!client) {
          log.warn('client not initialized, skipping retain');
          return;
        }


        // Use unique document ID per conversation (sessionKey + timestamp)
        // Static sessionKey (e.g. "agent:main:main") causes CASCADE delete of old memories
        const documentId = `${effectiveCtx?.sessionKey || 'session'}-${Date.now()}`;

        // Retain to Hindsight
        debug(`[Hindsight] Retaining to bank ${bankId}, document: ${documentId}, chars: ${transcript.length}\n---\n${transcript.substring(0, 500)}${transcript.length > 500 ? '\n...(truncated)' : ''}\n---`);
        await client.retain({
          content: transcript,
          document_id: documentId,
          metadata: {
            retained_at: new Date().toISOString(),
            message_count: String(messageCount),
            channel_type: effectiveCtx?.messageProvider,
            channel_id: effectiveCtx?.channelId,
            sender_id: effectiveCtx?.senderId,
          },
        });

        log.trackRetain(bankId, messageCount);
        debug(`[Hindsight] Retained ${messageCount} messages to bank ${bankId} for session ${documentId}`);
      } catch (error) {
        log.error('error retaining messages', error);
      }
    });
    debug('[Hindsight] Hooks registered');
  } catch (error) {
    log.error('plugin loading error', error);
    if (error instanceof Error) {
      log.error('error stack', error.stack);
    }
    throw error;
  }
}

// Export client getter for tools

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
      if (messages[i].role === 'user') {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx === -1) {
      return null; // No user message found in turn
    }
    targetMessages = messages.slice(lastUserIdx);
  }

  // Role filtering
  const allowedRoles = new Set(pluginConfig.retainRoles || ['user', 'assistant']);
  const filteredMessages = targetMessages.filter((m: any) => allowedRoles.has(m.role));

  if (filteredMessages.length === 0) {
    return null; // No messages to retain
  }

  // Format messages into a transcript
  const transcriptParts = filteredMessages
    .map((msg: any) => {
      const role = msg.role || 'unknown';
      let content = '';

      // Handle different content formats
      if (typeof msg.content === 'string') {
        content = msg.content;
      } else if (Array.isArray(msg.content)) {
        content = msg.content
          .filter((block: any) => block.type === 'text')
          .map((block: any) => block.text)
          .join('\n');
      }

      // Strip plugin-injected memory tags and metadata envelopes to prevent feedback loop
      content = stripMemoryTags(content);
      content = stripMetadataEnvelopes(content);

      return content.trim() ? `[role: ${role}]\n${content}\n[${role}:end]` : null;
    })
    .filter(Boolean);

  const transcript = transcriptParts.join('\n\n');

  if (!transcript.trim() || transcript.length < 10) {
    return null; // Transcript too short
  }

  return { transcript, messageCount: transcriptParts.length };
}

export function sliceLastTurnsByUserBoundary(messages: any[], turns: number): any[] {
  if (!Array.isArray(messages) || messages.length === 0 || turns <= 0) {
    return [];
  }

  let userTurnsSeen = 0;
  let startIndex = -1;

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') {
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
