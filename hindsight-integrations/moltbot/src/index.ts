import type { MoltbotPluginAPI, PluginConfig } from './types.js';
import { HindsightEmbedManager } from './embed-manager.js';
import { HindsightClient } from './client.js';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

// Module-level state
let embedManager: HindsightEmbedManager | null = null;
let client: HindsightClient | null = null;

// Get directory of current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Default bank name
const BANK_NAME = 'moltbot-agent';

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
    `Or run Moltbot with the environment variable:\n` +
    `  ANTHROPIC_API_KEY="your-key" clawdbot start\n\n` +
    `Alternatively, configure ollama provider which doesn't require an API key.`
  );
}

function getPluginConfig(api: MoltbotPluginAPI): PluginConfig {
  const config = api.config.plugins?.entries?.['hindsight-memory']?.config || {};
  return {
    bankMission: config.bankMission,
    embedPort: config.embedPort || 0,
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

    // Determine port
    const port = pluginConfig.embedPort || Math.floor(Math.random() * 10000) + 10000;
    console.log(`[Hindsight] Port: ${port}`);

    // Register background service
    console.log('[Hindsight] Registering service...');
    api.registerService({
      id: 'hindsight-memory',
      async start() {
        try {
          console.log('[Hindsight] Service starting...');

          // Initialize embed manager
          console.log('[Hindsight] Creating HindsightEmbedManager...');
          embedManager = new HindsightEmbedManager(
            port,
            llmConfig.provider,
            llmConfig.apiKey,
            llmConfig.model
          );

          // Start the embedded server
          console.log('[Hindsight] Starting embedded server...');
          await embedManager.start();

          // Initialize client
          console.log('[Hindsight] Creating HindsightClient...');
          client = new HindsightClient(llmConfig.provider, llmConfig.apiKey, llmConfig.model);

          // Use default bank
          console.log('[Hindsight] Using default bank');
          client.setBankId('default');

          console.log('[Hindsight] Service ready');
        } catch (error) {
          console.error('[Hindsight] Service start error:', error);
          throw error;
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

          console.log('[Hindsight] Service stopped');
        } catch (error) {
          console.error('[Hindsight] Service stop error:', error);
          throw error;
        }
      },
    });

    console.log('[Hindsight] Plugin loaded successfully');

    // Note: Hooks and skills are registered separately by Moltbot via the manifest
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
