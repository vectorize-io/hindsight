/**
 * Hindsight OpenCode Plugin — persistent long-term memory for OpenCode agents.
 *
 * Provides:
 *   - Custom tools: hindsight_retain, hindsight_recall, hindsight_reflect
 *   - Auto-retain on session.idle
 *   - Memory injection on session.created via system transform
 *   - Memory preservation during context compaction
 *
 * @example
 * ```json
 * // opencode.json
 * { "plugin": ["@vectorize-io/opencode-hindsight"] }
 *
 * // With options:
 * { "plugin": [["@vectorize-io/opencode-hindsight", { "bankId": "my-bank" }]] }
 * ```
 */

import type { Plugin } from "@opencode-ai/plugin";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { loadConfig } from "./config.js";
import { deriveBankId } from "./bank.js";
import { createTools } from "./tools.js";
import { createHooks, type PluginState } from "./hooks.js";
import { Logger, type OpencodeLogClient } from "./logger.js";

// Module-level state persists across sessions (plugin is instantiated per session,
// but the module is loaded once per OpenCode server process).
const state: PluginState = {
  turnCount: 0,
  missionsSet: new Set(),
  recalledSessions: new Set(),
  lastRetainedTurn: new Map(),
};

const HindsightPlugin: Plugin = async (input, options) => {
  const config = loadConfig(options);

  // Route logs through OpenCode's server log stream (TUI-safe). error/warn/info
  // are always emitted; debug is gated on config.debug.
  const logger = new Logger({
    client: input.client as unknown as OpencodeLogClient,
    debug: config.debug,
  });

  // hindsightApiUrl always resolves to a value (DEFAULT_HINDSIGHT_API_URL by default),
  // so plugin instantiation never fails just because the URL is unset.
  // Requests fail at call time if no API key is configured for a Cloud URL —
  // that surfaces a clear, actionable error from the server rather than silently
  // disabling the plugin.
  const client = new HindsightClient({
    baseUrl: config.hindsightApiUrl!,
    apiKey: config.hindsightApiToken || undefined,
  });

  const bankId = deriveBankId(config, input.directory);
  // Always log the resolved endpoint + bank so users can see which instance the
  // plugin is talking to (a common source of "memories aren't saving" confusion).
  logger.info("Hindsight plugin initialized", {
    api: config.hindsightApiUrl,
    bank: bankId,
    authenticated: Boolean(config.hindsightApiToken),
    autoRecall: config.autoRecall,
    autoRetain: config.autoRetain,
  });

  const tools = createTools(client, bankId, config, state.missionsSet, logger);
  const hooks = createHooks(
    client,
    bankId,
    config,
    state,
    input.client as unknown as Parameters<typeof createHooks>[4],
    logger
  );

  return {
    tool: tools,
    ...hooks,
  };
};

// Named export for direct import
export { HindsightPlugin };

// Default export is the Plugin function itself — OpenCode's legacy loader
// iterates Object.values(mod) and calls every function export as a Plugin
// factory. Keep the entry surface free of utility re-exports (loadConfig,
// deriveBankId) so they are not invoked as plugins. The plugin itself imports
// those utilities directly from their modules.
export default HindsightPlugin;

// Re-export types for consumers
export type { HindsightConfig } from "./config.js";
export type { PluginState } from "./hooks.js";
