// Moltbot plugin API types (minimal subset needed for this plugin)

export interface PluginPromptHookResult {
  prependContext?: string;
  prependSystemContext?: string;
  appendSystemContext?: string;
}

export interface MoltbotPluginAPI {
  config: MoltbotConfig;
  registerService(config: ServiceConfig): void;
  // OpenClaw hook handler signature: (event, ctx?) where ctx contains channel/sender info
  on(event: string, handler: (event: any, ctx?: any) => void | Promise<void | PluginPromptHookResult>): void;
  // OpenClaw framework logger — handles coloring/formatting consistently across plugins
  logger: {
    info(msg: string): void;
    warn(msg: string): void;
    error(msg: string): void;
  };
  // Add more as needed
}

export interface MoltbotConfig {
  agents?: {
    defaults?: {
      models?: {
        [modelName: string]: {
          alias?: string;
        };
      };
    };
  };
  plugins?: {
    entries?: {
      [pluginId: string]: {
        enabled?: boolean;
        config?: PluginConfig;
      };
    };
  };
}

export interface PluginHookAgentContext {
  agentId?: string;
  sessionKey?: string;
  workspaceDir?: string;
  messageProvider?: string;
  channelId?: string;
  senderId?: string;
}

export interface PluginConfig {
  bankMission?: string;
  embedPort?: number;
  daemonIdleTimeout?: number; // Seconds before daemon shuts down (0 = never)
  embedVersion?: string; // hindsight-embed version (default: "latest")
  embedPackagePath?: string; // Local path to hindsight package (e.g. '/path/to/hindsight')
  llmProvider?: string; // LLM provider (e.g. 'openai', 'anthropic', 'gemini', 'groq', 'ollama', 'openai-codex', 'claude-code')
  llmModel?: string; // LLM model (e.g. 'gpt-4o-mini', 'claude-3-5-haiku-20241022')
  llmApiKey?: string; // LLM provider API key. Configure via SecretRef: openclaw config set ... --ref-source env --ref-id OPENAI_API_KEY
  llmBaseUrl?: string; // Optional base URL override for OpenAI-compatible providers (e.g. OpenRouter)
  apiPort?: number; // Port for openclaw profile daemon (default: 9077)
  hindsightApiUrl?: string; // External Hindsight API URL (skips local daemon when set)
  hindsightApiToken?: string; // API token for external Hindsight API. Configure via SecretRef.
  dynamicBankId?: boolean; // Enable per-channel memory banks (default: true)
  bankId?: string; // Static bank ID used when dynamicBankId is false.
  bankIdPrefix?: string; // Prefix for bank IDs (e.g. 'prod' -> 'prod-slack-C123')
  retainTags?: string[]; // Tags applied to all retained documents after trimming and deduplication; auto-retain merges these with inline per-message retain-tag directives (e.g. ['source_system:openclaw', 'agent:agentname'])
  retainSource?: string; // Source written into retained document metadata (default: 'openclaw')
  excludeProviders?: string[]; // Message providers to exclude from recall/retain (e.g. ['telegram', 'discord'])
  autoRecall?: boolean; // Auto-recall memories on every prompt (default: true). Set to false when agent has its own recall tool.
  dynamicBankGranularity?: Array<'agent' | 'provider' | 'channel' | 'user'>; // Fields for bank ID derivation. Default: ['agent', 'channel', 'user']
  autoRetain?: boolean; // Default: true
  retainRoles?: Array<'user' | 'assistant' | 'system' | 'tool'>; // Roles to include in retained transcript. Default: ['user', 'assistant']
  retainFormat?: 'json' | 'text'; // Serialization format for retained conversation content. Default: 'json' (structured array of {role, content}); 'text' emits legacy '[role: x] ... [x:end]' markers.
  retainToolCalls?: boolean; // When true (default) and retainFormat='json', each message's content is an Anthropic-shaped array of typed blocks (text, tool_use, tool_result) including the agent's tool calls and their results. When false, content is a flat string with only text.
  recallBudget?: 'low' | 'mid' | 'high'; // Recall effort. Default: 'mid'
  recallMaxTokens?: number; // Max tokens for recall response. Default: 1024
  recallTypes?: Array<'world' | 'experience' | 'observation'>; // Memory types to recall. Default: ['world', 'experience']
  recallRoles?: Array<'user' | 'assistant' | 'system' | 'tool'>; // Roles to include when composing contextual recall query. Default: ['user', 'assistant']
  retainEveryNTurns?: number; // Retain every Nth turn (1 = every turn, default: 1). Values > 1 enable chunked retention.
  retainOverlapTurns?: number; // Extra prior turns included when chunked retention fires (default: 0). Window = retainEveryNTurns + retainOverlapTurns.
  recallTopK?: number; // Max number of memories to inject. Default: unlimited
  recallContextTurns?: number; // Number of user turns to include in recall query context. Default: 1 (latest only)
  recallTimeoutMs?: number; // Timeout for auto-recall in milliseconds. Default: 10000
  recallMaxQueryChars?: number; // Max chars for composed recall query. Default: 800
  recallPromptPreamble?: string; // Prompt preamble placed above recalled memories. Default: built-in guidance text.
  recallInjectionPosition?: 'prepend' | 'append' | 'user'; // Where to inject recalled memories. 'prepend' = start of system prompt (default), 'append' = end of system prompt (preserves prompt cache), 'user' = before user message.
  ignoreSessionPatterns?: string[]; // Session key glob patterns to skip entirely (no recall, no retain). E.g. ["agent:main:**", "agent:*:cron:**"]
  statelessSessionPatterns?: string[]; // Session key glob patterns for read-only sessions (recall allowed, retain skipped). E.g. ["agent:*:subagent:**"]
  skipStatelessSessions?: boolean; // When true (default), stateless sessions also skip recall. When false, they recall but never retain.
  debug?: boolean; // Enable debug logging (default: false)
  logLevel?: 'off' | 'error' | 'warning' | 'info' | 'debug'; // Console log verbosity (default: 'info').
  logSummaryIntervalMs?: number; // Batch retain/recall log summaries over this interval in ms. 0 = log every event. Default: 300000 (5 min).
  retainQueuePath?: string; // Path to JSONL file for buffering failed retains. Default: ~/.openclaw/data/hindsight-retain-queue.jsonl
  retainQueueMaxAgeMs?: number; // Max age in ms for queued items. -1 = keep forever (default: -1)
  retainQueueFlushIntervalMs?: number; // How often to attempt flushing the queue in ms. Default: 60000 (1 min)
}

export interface ServiceConfig {
  id: string;
  start(): Promise<void>;
  stop(): Promise<void>;
}

// -----------------------------------------------------------------------------
// Hindsight API types
// -----------------------------------------------------------------------------

// MemoryResult / RecallResponse / ReflectResponse come from the generated
// hindsight-client SDK. We alias MemoryResult → RecallResult so existing code
// paths (formatMemories, etc.) keep the old name.
export type { RecallResult as MemoryResult, RecallResponse, ReflectResponse } from '@vectorize-io/hindsight-client';

/**
 * Internal retain payload shape built by `buildRetainRequest`. Not a
 * re-export from the generated client — the generated client's retain()
 * takes bankId + content + options as positional args, whereas we build up a
 * single object inside the plugin and translate it at the call site. Keeping
 * this type local means tests can assert the shape without pulling in
 * generated types.
 */
export interface RetainRequest {
  content: string;
  documentId?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
}

/**
 * Stats returned by `GET /v1/default/banks/{bank_id}/stats`. The generated
 * high-level client does not expose this endpoint yet; backfill calls it
 * directly via `fetch`.
 */
export interface BankStats {
  bank_id: string;
  total_nodes: number;
  total_links: number;
  total_documents: number;
  pending_operations: number;
  failed_operations: number;
  pending_consolidation: number;
  last_consolidated_at: string | null;
  total_observations: number;
  nodes_by_fact_type?: Record<string, number>;
  links_by_link_type?: Record<string, number>;
  links_by_fact_type?: Record<string, number>;
  links_breakdown?: Record<string, unknown>;
}
