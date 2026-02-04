// Moltbot plugin API types (minimal subset needed for this plugin)

export interface MoltbotPluginAPI {
  config: MoltbotConfig;
  registerService(config: ServiceConfig): void;
  // OpenClaw hook handler signature: (event, ctx?) where ctx contains channel/sender info
  on(event: string, handler: (event: any, ctx?: any) => void | Promise<void | { prependContext?: string }>): void;
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
}

export interface RecallResponse {
  results: MemoryResult[];
}

export interface MemoryResult {
  content: string;
  score: number;
  metadata?: {
    document_id?: string;
    created_at?: string;
    source?: string;
  };
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
