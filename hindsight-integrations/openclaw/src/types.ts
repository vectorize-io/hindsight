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
  llmProvider?: string; // LLM provider override (e.g. 'openai', 'anthropic', 'gemini', 'groq', 'ollama')
  llmModel?: string; // LLM model override (e.g. 'gpt-4o-mini', 'claude-3-5-haiku-20241022')
  llmApiKeyEnv?: string; // Env var name holding the API key (e.g. 'MY_CUSTOM_KEY')
  apiPort?: number; // Port for openclaw profile daemon (default: 9077)
  hindsightApiUrl?: string; // External Hindsight API URL (skips local daemon when set)
  hindsightApiToken?: string; // API token for external Hindsight API authentication
  dynamicBankId?: boolean; // Enable per-channel memory banks (default: true)
  bankIdPrefix?: string; // Prefix for bank IDs (e.g. 'prod' -> 'prod-slack-C123')
  excludeProviders?: string[]; // Message providers to exclude from recall/retain (e.g. ['telegram', 'discord'])
  autoRecall?: boolean; // Auto-recall memories on every prompt (default: true). Set to false when agent has its own recall tool.
  dynamicBankGranularity?: Array<'agent' | 'provider' | 'channel' | 'user'>; // Fields for bank ID derivation. Default: ['agent', 'channel', 'user']
  autoRetain?: boolean; // Default: true
  retainRoles?: Array<'user' | 'assistant' | 'system' | 'tool'>; // Roles to include in retained transcript. Default: ['user', 'assistant']
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
  debug?: boolean; // Enable debug logging (default: false)
}

export interface ServiceConfig {
  id: string;
  start(): Promise<void>;
  stop(): Promise<void>;
}

// Hindsight API types

export interface RetainRequest {
  content: string;
  document_id?: string;
  metadata?: Record<string, unknown>;
}

export interface RetainResponse {
  message: string;
  document_id: string;
  memory_unit_ids: string[];
}

export interface RecallRequest {
  query: string;
  max_tokens?: number;
  budget?: 'low' | 'mid' | 'high';
  types?: Array<'world' | 'experience' | 'observation'>;
}

export interface RecallResponse {
  results: MemoryResult[];
  entities: Record<string, unknown> | null;
  trace: unknown | null;
  chunks: unknown | null;
}

export interface MemoryResult {
  id: string;
  text: string;
  type: string;
  entities: string[];
  context: string;
  occurred_start: string | null;
  occurred_end: string | null;
  mentioned_at: string | null;
  document_id: string | null;
  metadata: Record<string, unknown> | null;
  chunk_id: string | null;
  tags: string[];
}

export interface CreateBankRequest {
  name: string;
  background_context?: string;
}

export interface CreateBankResponse {
  bank_id: string;
  name: string;
  created_at: string;
}
