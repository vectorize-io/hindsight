/**
 * opencode-hindsight: Persistent long-term memory for OpenCode.
 *
 * An OpenCode plugin that automatically captures session conversations to
 * Hindsight and injects recalled memories into compaction context. For
 * on-demand recall during sessions, users configure Hindsight's built-in
 * MCP server in OpenCode's config.
 *
 * Hooks:
 *   session.idle   -> auto-retain the completed conversation
 *   session.compacting -> inject recalled memories into compaction context
 *
 * MCP (user-configured, not plugin code):
 *   Hindsight's /mcp/{bank_id}/ endpoint provides retain/recall/reflect
 *   tools directly to the agent.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { loadConfig, debugLog } from "./config";
import { HindsightClient } from "./client";
import { resolveApiUrl, stopDaemon } from "./daemon";
import { deriveBankId, ensureBankMission } from "./bank";
import { retainSession } from "./retain";

export const HindsightMemory: Plugin = async ({ directory }) => {
  const config = loadConfig();
  let client: HindsightClient | null = null;
  let apiUrl: string | null = null;

  // Attempt initial connection (non-blocking, no daemon start)
  try {
    apiUrl = await resolveApiUrl(config, false);
    if (apiUrl) {
      client = new HindsightClient(apiUrl, config.hindsightApiToken);
      const bankId = deriveBankId(config, directory);
      await ensureBankMission(client, bankId, config);
      debugLog(config, `Connected to Hindsight at ${apiUrl}, bank: ${bankId}`);
    } else {
      debugLog(config, "No Hindsight server available at startup (will retry on retain)");
    }
  } catch (err) {
    debugLog(config, `Startup connection failed: ${err}`);
  }

  return {
    event: async ({ event }) => {
      // Auto-retain on session completion
      if (event.type === "session.idle" && config.autoRetain) {
        try {
          // Resolve API URL (allow daemon start for retain)
          if (!apiUrl) {
            apiUrl = await resolveApiUrl(config, true);
            if (!apiUrl) {
              debugLog(config, "Cannot retain: no Hindsight server available");
              return;
            }
            client = new HindsightClient(apiUrl, config.hindsightApiToken);
          }

          await retainSession(event as Record<string, unknown>, config, client!, apiUrl);
        } catch (err) {
          debugLog(config, `Retain failed: ${err}`);
        }
      }
    },

    // Inject recalled memories into compaction context
    "experimental.session.compacting": async (input, output) => {
      if (!config.autoRecall || !client) return;

      try {
        const bankId = deriveBankId(config, directory);

        // Build a recall query from the compaction input when available.
        // The input may contain a summary or the conversation being compacted.
        const inputContext = typeof input === "object" && input !== null
          ? (input as Record<string, unknown>)
          : {};
        const sessionSummary = typeof inputContext.summary === "string"
          ? inputContext.summary.slice(0, 500)
          : "";
        const sessionTitle = typeof inputContext.title === "string"
          ? inputContext.title
          : "";

        const queryParts = [sessionTitle, sessionSummary].filter(Boolean);
        const query = queryParts.length > 0
          ? `Context from current session: ${queryParts.join(". ")}. What related decisions, preferences, and knowledge from past sessions are relevant?`
          : "What are the key decisions, preferences, and context from recent coding sessions?";

        const response = await client.recall({
          bankId,
          query,
          budget: config.recallBudget,
          maxTokens: config.recallMaxTokens,
          types: config.recallTypes,
        });

        if (response.results.length > 0) {
          const memories = response.results
            .map((r) => `- ${r.text} [${r.type}]${r.mentioned_at ? ` (${r.mentioned_at})` : ""}`)
            .join("\n\n");

          output.context.push(
            `## ${config.recallPromptPreamble}\n\n${memories}`,
          );

          debugLog(config, `Injected ${response.results.length} memories into compaction context`);
        }
      } catch (err) {
        debugLog(config, `Recall for compaction failed: ${err}`);
      }
    },
  };
};

export default HindsightMemory;
