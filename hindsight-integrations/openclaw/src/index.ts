import type { MoltbotPluginAPI, PluginConfig } from './types.js';
import { HindsightEmbedManager } from './embed-manager.js';
import { HindsightClient } from './client.js';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

// Module-level state
let embedManager: HindsightEmbedManager | null = null;
let client: HindsightClient | null = null;
let initPromise: Promise<void> | null = null;
let isInitialized = false;
let usingExternalApi = false; // Track if using external API (skip daemon management)

// Store the current plugin config for bank ID derivation
let currentPluginConfig: PluginConfig | null = null;

// Track which banks have had their mission set (to avoid re-setting on every request)
const banksWithMissionSet = new Set<string>();

// Global access for hooks (Moltbot loads hooks separately)
if (typeof global !== 'undefined') {
  (global as any).__hindsightClient = {
    getClient: () => client,
    waitForReady: async () => {
      if (isInitialized) return;
      if (initPromise) await initPromise;
    },
    /**
     * Get a client configured for a specific agent context.
     * Derives the bank ID from the context for per-channel isolation.
     * Also ensures the bank mission is set on first use.
     */
    getClientForContext: async (ctx: PluginHookAgentContext | undefined) => {
      if (!client) return null;
      const config = currentPluginConfig || {};
      const bankId = deriveBankId(ctx, config);
      client.setBankId(bankId);

      // Set bank mission on first use of this bank (if configured)
      if (config.bankMission && config.dynamicBankId && !banksWithMissionSet.has(bankId)) {
        try {
          await client.setBankMission(config.bankMission);
          banksWithMissionSet.add(bankId);
          console.log(`[Hindsight] Set mission for new bank: ${bankId}`);
        } catch (error) {
          // Log but don't fail - bank mission is not critical
          console.warn(`[Hindsight] Could not set bank mission for ${bankId}: ${error}`);
        }
      }

      return client;
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
 * Agent context passed to plugin hooks.
 * These fields are populated by OpenClaw when invoking hooks.
 */
interface PluginHookAgentContext {
  agentId?: string;
  sessionKey?: string;
  workspaceDir?: string;
  messageProvider?: string;
  channelId?: string;
  senderId?: string;
}

/**
 * Derive a bank ID from the agent context.
 * Creates channel-specific banks: {messageProvider}-{channelId}
 * Falls back to default bank when context is unavailable.
 */
function deriveBankId(
  ctx: PluginHookAgentContext | undefined,
  pluginConfig: PluginConfig
): string {
  // If dynamic bank ID is disabled, use static bank
  if (pluginConfig.dynamicBankId === false) {
    return pluginConfig.bankIdPrefix
      ? `${pluginConfig.bankIdPrefix}-${DEFAULT_BANK_NAME}`
      : DEFAULT_BANK_NAME;
  }

  const channelType = ctx?.messageProvider || 'unknown';
  const channelId = ctx?.channelId || 'default';

  // Build bank ID: {prefix?}-{channelType}-{channelId}
  const baseBankId = `${channelType}-${channelId}`;
  return pluginConfig.bankIdPrefix
    ? `${pluginConfig.bankIdPrefix}-${baseBankId}`
    : baseBankId;
}

// Provider detection from standard env vars
const PROVIDER_DETECTION = [
  { name: 'openai', keyEnv: 'OPENAI_API_KEY', defaultModel: 'gpt-4o-mini' },
  { name: 'anthropic', keyEnv: 'ANTHROPIC_API_KEY', defaultModel: 'claude-3-5-haiku-20241022' },
  { name: 'gemini', keyEnv: 'GEMINI_API_KEY', defaultModel: 'gemini-2.5-flash' },
  { name: 'groq', keyEnv: 'GROQ_API_KEY', defaultModel: 'openai/gpt-oss-20b' },
  { name: 'ollama', keyEnv: '', defaultModel: 'llama3.2' },
  { name: 'openai-codex', keyEnv: '', defaultModel: 'gpt-5.2-codex' },
  { name: 'claude-code', keyEnv: '', defaultModel: 'claude-sonnet-4-5-20250929' },
];

function detectLLMConfig(pluginConfig?: PluginConfig): {
  provider: string;
  apiKey: string;
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

    const providerInfo = PROVIDER_DETECTION.find(p => p.name === overrideProvider);
    return {
      provider: overrideProvider,
      apiKey: overrideKey || '',
      model: overrideModel || (providerInfo?.defaultModel),
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
      model: pluginConfig.llmModel || overrideModel || providerInfo?.defaultModel,
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
        model: overrideModel || providerInfo.defaultModel,
        baseUrl: overrideBaseUrl, // Only use explicit HINDSIGHT_API_LLM_BASE_URL
        source: `auto-detected from ${providerInfo.keyEnv}`,
      };
    }
  }

  // No configuration found - show helpful error
  throw new Error(
    `No LLM configuration found for Hindsight memory plugin.\n\n` +
    `Option 1: Set a standard provider API key (auto-detect):\n` +
    `  export OPENAI_API_KEY=sk-your-key        # Uses gpt-4o-mini\n` +
    `  export ANTHROPIC_API_KEY=your-key       # Uses claude-3-5-haiku\n` +
    `  export GEMINI_API_KEY=your-key          # Uses gemini-2.5-flash\n` +
    `  export GROQ_API_KEY=your-key            # Uses openai/gpt-oss-20b\n\n` +
    `Option 2: Use Codex or Claude Code (no API key needed):\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=openai-codex    # Requires 'codex auth login'\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=claude-code     # Requires Claude Code CLI\n\n` +
    `Option 3: Set llmProvider in openclaw.json plugin config:\n` +
    `  "llmProvider": "openai", "llmModel": "gpt-4o-mini"\n\n` +
    `Option 4: Override with Hindsight-specific env vars:\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=openai\n` +
    `  export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n` +
    `  export HINDSIGHT_API_LLM_API_KEY=sk-your-key\n` +
    `  export HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1  # Optional\n\n` +
    `Tip: Use a cheap/fast model for memory extraction (e.g., gpt-4o-mini, claude-3-5-haiku, or free models on OpenRouter)`
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
 * Health check for external Hindsight API.
 */
async function checkExternalApiHealth(apiUrl: string): Promise<void> {
  const healthUrl = `${apiUrl.replace(/\/$/, '')}/health`;
  console.log(`[Hindsight] Checking external API health at ${healthUrl}...`);
  try {
    const response = await fetch(healthUrl, { signal: AbortSignal.timeout(10000) });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json() as { status?: string };
    console.log(`[Hindsight] External API health: ${JSON.stringify(data)}`);
  } catch (error) {
    throw new Error(`Cannot connect to external Hindsight API at ${apiUrl}: ${error}`);
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
  };
}

export default function (api: MoltbotPluginAPI) {
  try {
    console.log('[Hindsight] Plugin loading...');

    // Get plugin config first (needed for LLM detection)
    console.log('[Hindsight] Getting plugin config...');
    const pluginConfig = getPluginConfig(api);

    // Store config globally for bank ID derivation in hooks
    currentPluginConfig = pluginConfig;

    // Detect LLM configuration (env vars > plugin config > auto-detect)
    console.log('[Hindsight] Detecting LLM config...');
    const llmConfig = detectLLMConfig(pluginConfig);

    const baseUrlInfo = llmConfig.baseUrl ? `, base URL: ${llmConfig.baseUrl}` : '';
    const modelInfo = llmConfig.model || 'default';

    if (llmConfig.provider === 'ollama') {
      console.log(`[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${modelInfo} (${llmConfig.source})`);
    } else {
      console.log(`[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${modelInfo} (${llmConfig.source}${baseUrlInfo})`);
    }
    if (pluginConfig.bankMission) {
      console.log(`[Hindsight] Custom bank mission configured: "${pluginConfig.bankMission.substring(0, 50)}..."`);
    }

    // Log dynamic bank ID mode
    if (pluginConfig.dynamicBankId) {
      const prefixInfo = pluginConfig.bankIdPrefix ? ` (prefix: ${pluginConfig.bankIdPrefix})` : '';
      console.log(`[Hindsight] ✓ Dynamic bank IDs enabled${prefixInfo} - each channel gets isolated memory`);
    } else {
      console.log(`[Hindsight] Dynamic bank IDs disabled - using static bank: ${DEFAULT_BANK_NAME}`);
    }

    // Detect external API mode
    const externalApi = detectExternalApi(pluginConfig);

    // Get API port from config (default: 9077)
    const apiPort = pluginConfig.apiPort || 9077;

    if (externalApi.apiUrl) {
      // External API mode - skip local daemon
      usingExternalApi = true;
      console.log(`[Hindsight] ✓ Using external API: ${externalApi.apiUrl}`);

      // Set env vars so CLI commands (uvx hindsight-embed) use external API
      process.env.HINDSIGHT_EMBED_API_URL = externalApi.apiUrl;
      if (externalApi.apiToken) {
        process.env.HINDSIGHT_EMBED_API_TOKEN = externalApi.apiToken;
        console.log('[Hindsight] API token configured');
      }
    } else {
      console.log(`[Hindsight] Daemon idle timeout: ${pluginConfig.daemonIdleTimeout}s (0 = never timeout)`);
      console.log(`[Hindsight] API Port: ${apiPort}`);
    }

    // Initialize in background (non-blocking)
    console.log('[Hindsight] Starting initialization in background...');
    initPromise = (async () => {
      try {
        if (usingExternalApi && externalApi.apiUrl) {
          // External API mode - check health, skip daemon startup
          console.log('[Hindsight] External API mode - skipping local daemon...');
          await checkExternalApiHealth(externalApi.apiUrl);

          // Initialize client (CLI commands will use external API via env vars)
          console.log('[Hindsight] Creating HindsightClient...');
          client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model, pluginConfig.embedVersion, pluginConfig.embedPackagePath);

          // Set default bank (will be overridden per-request when dynamic bank IDs are enabled)
          const defaultBankId = deriveBankId(undefined, pluginConfig);
          console.log(`[Hindsight] Default bank: ${defaultBankId}`);
          client.setBankId(defaultBankId);

          // Note: Bank mission will be set per-bank when dynamic bank IDs are enabled
          // For now, set it on the default bank
          if (pluginConfig.bankMission && !pluginConfig.dynamicBankId) {
            console.log(`[Hindsight] Setting bank mission...`);
            await client.setBankMission(pluginConfig.bankMission);
          }

          isInitialized = true;
          console.log('[Hindsight] ✓ Ready (external API mode)');
        } else {
          // Local daemon mode - start hindsight-embed daemon
          console.log('[Hindsight] Creating HindsightEmbedManager...');
          embedManager = new HindsightEmbedManager(
            apiPort,
            llmConfig.provider,
            llmConfig.apiKey,
            llmConfig.model,
            llmConfig.baseUrl,
            pluginConfig.daemonIdleTimeout,
            pluginConfig.embedVersion,
            pluginConfig.embedPackagePath
          );

          // Start the embedded server
          console.log('[Hindsight] Starting embedded server...');
          await embedManager.start();

          // Initialize client
          console.log('[Hindsight] Creating HindsightClient...');
          client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model, pluginConfig.embedVersion, pluginConfig.embedPackagePath);

          // Set default bank (will be overridden per-request when dynamic bank IDs are enabled)
          const defaultBankId = deriveBankId(undefined, pluginConfig);
          console.log(`[Hindsight] Default bank: ${defaultBankId}`);
          client.setBankId(defaultBankId);

          // Note: Bank mission will be set per-bank when dynamic bank IDs are enabled
          // For now, set it on the default bank
          if (pluginConfig.bankMission && !pluginConfig.dynamicBankId) {
            console.log(`[Hindsight] Setting bank mission...`);
            await client.setBankMission(pluginConfig.bankMission);
          }

          isInitialized = true;
          console.log('[Hindsight] ✓ Ready');
        }
      } catch (error) {
        console.error('[Hindsight] Initialization error:', error);
        throw error;
      }
    })();

    // Don't await - let it initialize in background

    // Register background service for cleanup
    console.log('[Hindsight] Registering service...');
    api.registerService({
      id: 'hindsight-memory',
      async start() {
        console.log('[Hindsight] Service start called...');

        // Wait for background init if still pending
        if (initPromise) {
          try {
            await initPromise;
          } catch (error) {
            console.error('[Hindsight] Initial initialization failed:', error);
            // Continue to health check below
          }
        }

        // External API mode: check external API health
        if (usingExternalApi) {
          const externalApi = detectExternalApi(pluginConfig);
          if (externalApi.apiUrl && isInitialized) {
            try {
              await checkExternalApiHealth(externalApi.apiUrl);
              console.log('[Hindsight] External API is healthy');
              return;
            } catch (error) {
              console.error('[Hindsight] External API health check failed:', error);
              // Reset state for reinitialization attempt
              client = null;
              isInitialized = false;
            }
          }
        } else {
          // Local daemon mode: check daemon health (handles SIGUSR1 restart case)
          if (embedManager && isInitialized) {
            const healthy = await embedManager.checkHealth();
            if (healthy) {
              console.log('[Hindsight] Daemon is healthy');
              return;
            }

            console.log('[Hindsight] Daemon is not responding - reinitializing...');
            // Reset state for reinitialization
            embedManager = null;
            client = null;
            isInitialized = false;
          }
        }

        // Reinitialize if needed (fresh start or recovery)
        if (!isInitialized) {
          console.log('[Hindsight] Reinitializing...');
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

            await checkExternalApiHealth(externalApi.apiUrl);

            client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model, reinitPluginConfig.embedVersion, reinitPluginConfig.embedPackagePath);
            const defaultBankId = deriveBankId(undefined, reinitPluginConfig);
            client.setBankId(defaultBankId);

            if (reinitPluginConfig.bankMission && !reinitPluginConfig.dynamicBankId) {
              await client.setBankMission(reinitPluginConfig.bankMission);
            }

            isInitialized = true;
            console.log('[Hindsight] Reinitialization complete (external API mode)');
          } else {
            // Local daemon mode
            embedManager = new HindsightEmbedManager(
              apiPort,
              llmConfig.provider,
              llmConfig.apiKey,
              llmConfig.model,
              llmConfig.baseUrl,
              reinitPluginConfig.daemonIdleTimeout,
              reinitPluginConfig.embedVersion,
              reinitPluginConfig.embedPackagePath
            );

            await embedManager.start();

            client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model, reinitPluginConfig.embedVersion, reinitPluginConfig.embedPackagePath);
            const defaultBankId = deriveBankId(undefined, reinitPluginConfig);
            client.setBankId(defaultBankId);

            if (reinitPluginConfig.bankMission && !reinitPluginConfig.dynamicBankId) {
              await client.setBankMission(reinitPluginConfig.bankMission);
            }

            isInitialized = true;
            console.log('[Hindsight] Reinitialization complete');
          }
        }
      },

      async stop() {
        try {
          console.log('[Hindsight] Service stopping...');

          // Only stop daemon if in local mode
          if (!usingExternalApi && embedManager) {
            await embedManager.stop();
            embedManager = null;
          }

          client = null;
          isInitialized = false;

          console.log('[Hindsight] Service stopped');
        } catch (error) {
          console.error('[Hindsight] Service stop error:', error);
          throw error;
        }
      },
    });

    console.log('[Hindsight] Plugin loaded successfully');

    // Register agent hooks for auto-recall and auto-retention
    console.log('[Hindsight] Registering agent hooks...');

    // Store session key and context for retention
    let currentSessionKey: string | undefined;
    let currentAgentContext: PluginHookAgentContext | undefined;

    // Auto-recall: Inject relevant memories before agent processes the message
    // Hook signature: (event, ctx) where event has {prompt, messages?} and ctx has agent context
    api.on('before_agent_start', async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        // Capture session key and context for use in agent_end
        if (ctx?.sessionKey) {
          currentSessionKey = ctx.sessionKey;
        }
        currentAgentContext = ctx;

        // Derive bank ID from context
        const bankId = deriveBankId(ctx, pluginConfig);
        console.log(`[Hindsight] before_agent_start - bank: ${bankId}, channel: ${ctx?.messageProvider}/${ctx?.channelId}`);

        // Get the user's latest message for recall
        let prompt = event.prompt;
        if (!prompt || typeof prompt !== 'string' || prompt.length < 5) {
          return; // Skip very short messages
        }

        // Extract actual message from Telegram format: [Telegram ... GMT+1] actual message
        const telegramMatch = prompt.match(/\[Telegram[^\]]+\]\s*(.+)$/);
        if (telegramMatch) {
          prompt = telegramMatch[1].trim();
        }

        if (prompt.length < 5) {
          return; // Skip very short messages after extraction
        }

        // Wait for client to be ready
        const clientGlobal = (global as any).__hindsightClient;
        if (!clientGlobal) {
          console.log('[Hindsight] Client global not available, skipping auto-recall');
          return;
        }

        await clientGlobal.waitForReady();

        // Get client configured for this context's bank (async to handle mission setup)
        const client = await clientGlobal.getClientForContext(ctx);
        if (!client) {
          console.log('[Hindsight] Client not initialized, skipping auto-recall');
          return;
        }

        console.log(`[Hindsight] Auto-recall for bank ${bankId}, prompt: ${prompt.substring(0, 50)}`);

        // Recall relevant memories (up to 512 tokens)
        const response = await client.recall({
          query: prompt,
          max_tokens: 512,
        });

        if (!response.results || response.results.length === 0) {
          console.log('[Hindsight] No memories found for auto-recall');
          return;
        }

        // Format memories as JSON with all fields from recall
        const memoriesJson = JSON.stringify(response.results, null, 2);

        const contextMessage = `<hindsight_memories>
Relevant memories from past conversations (score 1=highest, prioritize recent when conflicting):
${memoriesJson}

User message: ${prompt}
</hindsight_memories>`;

        console.log(`[Hindsight] Auto-recall: Injecting ${response.results.length} memories from bank ${bankId}`);

        // Inject context before the user message
        return { prependContext: contextMessage };
      } catch (error) {
        console.error('[Hindsight] Auto-recall error:', error);
        return;
      }
    });

    // Hook signature: (event, ctx) where event has {messages, success, error?, durationMs?}
    api.on('agent_end', async (event: any, ctx?: PluginHookAgentContext) => {
      try {
        // Use context from this hook, or fall back to context captured in before_agent_start
        const effectiveCtx = ctx || currentAgentContext;

        // Derive bank ID from context
        const bankId = deriveBankId(effectiveCtx, pluginConfig);
        console.log(`[Hindsight Hook] agent_end triggered - bank: ${bankId}`);

        // Check event success and messages
        if (!event.success || !Array.isArray(event.messages) || event.messages.length === 0) {
          console.log('[Hindsight Hook] Skipping: success:', event.success, 'messages:', event.messages?.length);
          return;
        }

        // Wait for client to be ready
        const clientGlobal = (global as any).__hindsightClient;
        if (!clientGlobal) {
          console.warn('[Hindsight] Client global not found, skipping retain');
          return;
        }

        await clientGlobal.waitForReady();

        // Get client configured for this context's bank (async to handle mission setup)
        const client = await clientGlobal.getClientForContext(effectiveCtx);
        if (!client) {
          console.warn('[Hindsight] Client not initialized, skipping retain');
          return;
        }

        // Format messages into a transcript
        const transcript = event.messages
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

            return `[role: ${role}]\n${content}\n[${role}:end]`;
          })
          .join('\n\n');

        if (!transcript.trim() || transcript.length < 10) {
          console.log('[Hindsight Hook] Transcript too short, skipping');
          return;
        }

        // Use session key as document ID (prefer context over captured value)
        const documentId = effectiveCtx?.sessionKey || currentSessionKey || 'default-session';

        // Retain to Hindsight
        await client.retain({
          content: transcript,
          document_id: documentId,
          metadata: {
            retained_at: new Date().toISOString(),
            message_count: event.messages.length,
            channel_type: effectiveCtx?.messageProvider,
            channel_id: effectiveCtx?.channelId,
            sender_id: effectiveCtx?.senderId,
          },
        });

        console.log(`[Hindsight] Retained ${event.messages.length} messages to bank ${bankId} for session ${documentId}`);
      } catch (error) {
        console.error('[Hindsight] Error retaining messages:', error);
      }
    });
    console.log('[Hindsight] Hooks registered');
  } catch (error) {
    console.error('[Hindsight] Plugin loading error:', error);
    if (error instanceof Error) {
      console.error('[Hindsight] Error stack:', error.stack);
    }
    throw error;
  }
}

// Export client getter for tools
export function getClient() {
  return client;
}
