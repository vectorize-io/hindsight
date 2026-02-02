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

// Global access for hooks (Moltbot loads hooks separately)
if (typeof global !== 'undefined') {
  (global as any).__hindsightClient = {
    getClient: () => client,
    waitForReady: async () => {
      if (isInitialized) return;
      if (initPromise) await initPromise;
    },
  };
}

// Get directory of current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Default bank name
const BANK_NAME = 'openclaw';

// Provider detection from standard env vars
const PROVIDER_DETECTION = [
  { name: 'openai', keyEnv: 'OPENAI_API_KEY', defaultModel: 'gpt-4o-mini' },
  { name: 'anthropic', keyEnv: 'ANTHROPIC_API_KEY', defaultModel: 'claude-3-5-haiku-20241022' },
  { name: 'gemini', keyEnv: 'GEMINI_API_KEY', defaultModel: 'gemini-2.5-flash' },
  { name: 'groq', keyEnv: 'GROQ_API_KEY', defaultModel: 'openai/gpt-oss-20b' },
  { name: 'ollama', keyEnv: '', defaultModel: 'llama3.2' },
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
    if (!overrideKey && overrideProvider !== 'ollama') {
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

    if (!apiKey && pluginConfig.llmProvider !== 'ollama') {
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

    // Skip ollama in auto-detection (must be explicitly requested)
    if (providerInfo.name === 'ollama') {
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
    `  export GEMINI_API_KEY=your-key          # Uses gemini-2.0-flash-exp\n` +
    `  export GROQ_API_KEY=your-key            # Uses llama-3.3-70b-versatile\n\n` +
    `Option 2: Set llmProvider in openclaw.json plugin config:\n` +
    `  "llmProvider": "openai", "llmModel": "gpt-4o-mini"\n\n` +
    `Option 3: Override with Hindsight-specific env vars:\n` +
    `  export HINDSIGHT_API_LLM_PROVIDER=openai\n` +
    `  export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n` +
    `  export HINDSIGHT_API_LLM_API_KEY=sk-your-key\n` +
    `  export HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1  # Optional\n\n` +
    `Tip: Use a cheap/fast model for memory extraction (e.g., gpt-4o-mini, claude-3-5-haiku, or free models on OpenRouter)`
  );
}

function getPluginConfig(api: MoltbotPluginAPI): PluginConfig {
  const config = api.config.plugins?.entries?.['hindsight-openclaw']?.config || {};
  const defaultMission = 'You are an AI assistant helping users across multiple communication channels (Telegram, Slack, Discord, etc.). Remember user preferences, instructions, and important context from conversations to provide personalized assistance.';

  return {
    bankMission: config.bankMission || defaultMission,
    embedPort: config.embedPort || 0,
    daemonIdleTimeout: config.daemonIdleTimeout !== undefined ? config.daemonIdleTimeout : 0,
    embedVersion: config.embedVersion || 'latest',
    llmProvider: config.llmProvider,
    llmModel: config.llmModel,
    llmApiKeyEnv: config.llmApiKeyEnv,
  };
}

export default function (api: MoltbotPluginAPI) {
  try {
    console.log('[Hindsight] Plugin loading...');

    // Get plugin config first (needed for LLM detection)
    console.log('[Hindsight] Getting plugin config...');
    const pluginConfig = getPluginConfig(api);

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
    console.log(`[Hindsight] Daemon idle timeout: ${pluginConfig.daemonIdleTimeout}s (0 = never timeout)`);

    // Determine port
    const port = pluginConfig.embedPort || Math.floor(Math.random() * 10000) + 10000;
    console.log(`[Hindsight] Port: ${port}`);

    // Initialize in background (non-blocking)
    console.log('[Hindsight] Starting initialization in background...');
    initPromise = (async () => {
      try {
        // Initialize embed manager
        console.log('[Hindsight] Creating HindsightEmbedManager...');
        embedManager = new HindsightEmbedManager(
          port,
          llmConfig.provider,
          llmConfig.apiKey,
          llmConfig.model,
          llmConfig.baseUrl,
          pluginConfig.daemonIdleTimeout,
          pluginConfig.embedVersion
        );

        // Start the embedded server
        console.log('[Hindsight] Starting embedded server...');
        await embedManager.start();

        // Initialize client
        console.log('[Hindsight] Creating HindsightClient...');
        client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model, pluginConfig.embedVersion);

        // Use openclaw bank
        console.log(`[Hindsight] Using bank: ${BANK_NAME}`);
        client.setBankId(BANK_NAME);

        // Set bank mission
        if (pluginConfig.bankMission) {
          console.log(`[Hindsight] Setting bank mission...`);
          await client.setBankMission(pluginConfig.bankMission);
        }

        isInitialized = true;
        console.log('[Hindsight] ✓ Ready');
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
        console.log('[Hindsight] Service start called - checking daemon health...');

        // Wait for background init if still pending
        if (initPromise) {
          try {
            await initPromise;
          } catch (error) {
            console.error('[Hindsight] Initial initialization failed:', error);
            // Continue to health check below
          }
        }

        // Check if daemon is actually healthy (handles SIGUSR1 restart case)
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

        // Reinitialize if needed (fresh start or recovery from dead daemon)
        if (!isInitialized) {
          console.log('[Hindsight] Reinitializing daemon...');
          const pluginConfig = getPluginConfig(api);
          const llmConfig = detectLLMConfig(pluginConfig);
          const port = pluginConfig.embedPort || Math.floor(Math.random() * 10000) + 10000;

          embedManager = new HindsightEmbedManager(
            port,
            llmConfig.provider,
            llmConfig.apiKey,
            llmConfig.model,
            llmConfig.baseUrl,
            pluginConfig.daemonIdleTimeout,
            pluginConfig.embedVersion
          );

          await embedManager.start();

          client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model, pluginConfig.embedVersion);
          client.setBankId(BANK_NAME);

          if (pluginConfig.bankMission) {
            await client.setBankMission(pluginConfig.bankMission);
          }

          isInitialized = true;
          console.log('[Hindsight] Reinitialization complete');
        }
      },

      async stop() {
        try {
          console.log('[Hindsight] Service stopping...');

          if (embedManager) {
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

    // Register agent_end hook for auto-retention
    console.log('[Hindsight] Registering agent_end hook...');
    // Store session key for retention
    let currentSessionKey: string | undefined;

    // Auto-recall: Inject relevant memories before agent processes the message
    api.on('before_agent_start', async (context: any) => {
      try {
        // Capture session key
        if (context.sessionKey) {
          currentSessionKey = context.sessionKey as string;
          console.log('[Hindsight] Captured session key:', currentSessionKey);
        }

        // Get the user's latest message for recall
        let prompt = context.prompt;
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

        const client = clientGlobal.getClient();
        if (!client) {
          console.log('[Hindsight] Client not initialized, skipping auto-recall');
          return;
        }

        console.log('[Hindsight] Auto-recall for prompt:', prompt.substring(0, 50));

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

        console.log(`[Hindsight] Auto-recall: Injecting ${response.results.length} memories`);

        // Inject context before the user message
        return { prependContext: contextMessage };
      } catch (error) {
        console.error('[Hindsight] Auto-recall error:', error);
        return;
      }
    });

    api.on('agent_end', async (event: any) => {
      try {
        console.log('[Hindsight Hook] agent_end triggered');

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

        const client = clientGlobal.getClient();
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

        // Use session key as document ID
        const documentId = currentSessionKey || 'default-session';

        // Retain to Hindsight
        await client.retain({
          content: transcript,
          document_id: documentId,
          metadata: {
            retained_at: new Date().toISOString(),
            message_count: event.messages.length,
          },
        });

        console.log(`[Hindsight] Retained ${event.messages.length} messages for session ${documentId}`);
      } catch (error) {
        console.error('[Hindsight] Error retaining messages:', error);
      }
    });
    console.log('[Hindsight] Hook registered');
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
