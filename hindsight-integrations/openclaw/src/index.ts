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

// Provider mapping: moltbot provider name -> hindsight provider name
const PROVIDER_MAP: Record<string, string> = {
  anthropic: 'anthropic',
  openai: 'openai',
  'openai-codex': 'openai',
  gemini: 'gemini',
  groq: 'groq',
  ollama: 'ollama',
};

// Environment variable mapping
const ENV_KEY_MAP: Record<string, string> = {
  anthropic: 'ANTHROPIC_API_KEY',
  openai: 'OPENAI_API_KEY',
  'openai-codex': 'OPENAI_API_KEY',
  gemini: 'GEMINI_API_KEY',
  groq: 'GROQ_API_KEY',
  ollama: '', // No key needed for local ollama
};

function detectLLMConfig(api: MoltbotPluginAPI): {
  provider: string;
  apiKey: string;
  model?: string;
  envKey?: string;
} {
  // Get models from config (agents.defaults.models is a dictionary of models)
  const models = api.config.agents?.defaults?.models;
  if (!models || Object.keys(models).length === 0) {
    throw new Error(
      'No models configured in Moltbot. Please configure at least one model in agents.defaults.models'
    );
  }

  // Try all configured models to find one with an available API key
  const configuredModels = Object.keys(models);

  for (const modelKey of configuredModels) {
    const [moltbotProvider, ...modelParts] = modelKey.split('/');
    const model = modelParts.join('/');
    const hindsightProvider = PROVIDER_MAP[moltbotProvider];

    if (!hindsightProvider) {
      continue; // Skip unsupported providers
    }

    const envKey = ENV_KEY_MAP[moltbotProvider];
    const apiKey = envKey ? process.env[envKey] || '' : '';

    // For ollama, no key is needed
    if (hindsightProvider === 'ollama') {
      return { provider: hindsightProvider, apiKey: '', model, envKey: '' };
    }

    // If we found a key, use this provider
    if (apiKey) {
      return { provider: hindsightProvider, apiKey, model, envKey };
    }
  }

  // No API keys found for any provider - show helpful error
  const configuredProviders = configuredModels
    .map(m => m.split('/')[0])
    .filter(p => PROVIDER_MAP[p]);

  const keyInstructions = configuredProviders
    .map(p => {
      const envVar = ENV_KEY_MAP[p];
      return envVar ? `  • ${envVar} (for ${p})` : null;
    })
    .filter(Boolean)
    .join('\n');

  throw new Error(
    `No API keys found for Hindsight memory plugin.\n\n` +
    `Configured providers in Moltbot: ${configuredProviders.join(', ')}\n\n` +
    `Please set one of these environment variables:\n${keyInstructions}\n\n` +
    `You can set them in your shell profile (~/.zshrc or ~/.bashrc):\n` +
    `  export ANTHROPIC_API_KEY="your-key-here"\n\n` +
    `Or run OpenClaw with the environment variable:\n` +
    `  ANTHROPIC_API_KEY="your-key" openclaw gateway\n\n` +
    `Alternatively, configure ollama provider which doesn't require an API key.`
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
  };
}

export default function (api: MoltbotPluginAPI) {
  try {
    console.log('[Hindsight] Plugin loading...');

    // Detect LLM configuration from Moltbot
    console.log('[Hindsight] Detecting LLM config...');
    const llmConfig = detectLLMConfig(api);

    if (llmConfig.provider === 'ollama') {
      console.log(`[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${llmConfig.model || 'default'} (no API key required)`);
    } else {
      console.log(`[Hindsight] ✓ Using provider: ${llmConfig.provider}, model: ${llmConfig.model || 'default'} (API key: ${llmConfig.envKey})`);
    }

    console.log('[Hindsight] Getting plugin config...');
    const pluginConfig = getPluginConfig(api);
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
        // Wait for background init if still pending
        console.log('[Hindsight] Service start called - ensuring initialization complete...');
        if (initPromise) await initPromise;
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
