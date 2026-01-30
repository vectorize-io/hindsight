// Moltbot plugin API types (minimal subset needed for this plugin)

export interface MoltbotPluginAPI {
  config: MoltbotConfig;
  registerService(config: ServiceConfig): void;
  on(event: string, handler: (context: any) => void | Promise<void | { prependContext?: string }>): void;
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
