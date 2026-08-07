/**
 * Custom tool definitions for the Hindsight OpenCode plugin.
 *
 * Registers retain, operation status, recall, and reflect tools the agent can
 * call explicitly.
 */

import { tool } from "@opencode-ai/plugin/tool";
import type { ToolDefinition } from "@opencode-ai/plugin/tool";
import type { HindsightClient } from "@vectorize-io/hindsight-client";
import type { HindsightConfig } from "./config.js";
import { formatMemories, formatCurrentTime } from "./content.js";
import { ensureBankMission } from "./bank.js";
import { Logger } from "./logger.js";

export interface HindsightTools {
  hindsight_retain: ToolDefinition;
  hindsight_operation_status: ToolDefinition;
  hindsight_recall: ToolDefinition;
  hindsight_reflect: ToolDefinition;
  // Index signature so the object is assignable to OpenCode's Hooks.tool
  // (Record<string, ToolDefinition>) without losing the specific keys above.
  [key: string]: ToolDefinition;
}

export function createTools(
  client: HindsightClient,
  bankId: string,
  config: HindsightConfig,
  missionsSet?: Set<string>,
  logger: Logger = new Logger({ silent: true })
): HindsightTools {
  const hindsight_retain = tool({
    description:
      "Store information in long-term memory. Use this to remember important facts, " +
      "user preferences, project context, decisions, and anything worth recalling in future sessions. " +
      "Be specific — include who, what, when, and why.",
    args: {
      content: tool.schema
        .string()
        .describe("The information to remember. Be specific and self-contained."),
      context: tool.schema
        .string()
        .optional()
        .describe("Optional context about where this information came from."),
    },
    async execute(args) {
      if (missionsSet) {
        await ensureBankMission(client, bankId, config, missionsSet, logger);
      }
      const response = await client.retain(bankId, args.content, {
        context: args.context || config.retainContext,
        tags: config.retainTags.length ? config.retainTags : undefined,
        metadata: Object.keys(config.retainMetadata).length ? config.retainMetadata : undefined,
        async: true,
      });
      if (!response.operation_id) {
        throw new Error("Async retain did not return an operation ID.");
      }
      return (
        `Memory queued successfully. Operation ID: ${response.operation_id}. ` +
        "Use hindsight_operation_status to poll for completion."
      );
    },
  });

  const hindsight_operation_status = tool({
    description:
      "Check the status of an asynchronous Hindsight operation returned by hindsight_retain.",
    args: {
      operationId: tool.schema.string().describe("The operation ID returned by hindsight_retain."),
    },
    async execute(args) {
      // The high-level client supports async retain but does not expose operation status.
      const baseUrl = config.hindsightApiUrl!.replace(/\/$/, "");
      const response = await fetch(
        `${baseUrl}/v1/default/banks/${encodeURIComponent(bankId)}/operations/${encodeURIComponent(args.operationId)}`,
        {
          headers: config.hindsightApiToken
            ? { Authorization: `Bearer ${config.hindsightApiToken}` }
            : undefined,
        }
      );
      if (!response.ok) {
        const details = await response.text();
        throw new Error(`Operation status failed (${response.status}): ${details.slice(0, 500)}`);
      }

      return JSON.stringify(await response.json(), null, 2);
    },
  });

  const hindsight_recall = tool({
    description:
      "Search long-term memory for relevant information. Use this proactively before " +
      "answering questions about past conversations, user preferences, project history, " +
      "or any topic where prior context would help. When in doubt, recall first.",
    args: {
      query: tool.schema
        .string()
        .describe("Natural language search query. Be specific about what you need to know."),
    },
    async execute(args) {
      const response = await client.recall(bankId, args.query, {
        budget: config.recallBudget as "low" | "mid" | "high",
        maxTokens: config.recallMaxTokens,
        types: config.recallTypes,
        tags: config.recallTags.length ? config.recallTags : undefined,
        tagsMatch: config.recallTags.length ? config.recallTagsMatch : undefined,
      });

      const results = response.results || [];
      if (!results.length) return "No relevant memories found.";

      const formatted = formatMemories(results);
      return `Found ${results.length} relevant memories (as of ${formatCurrentTime()} UTC):\n\n${formatted}`;
    },
  });

  const hindsight_reflect = tool({
    description:
      "Generate a thoughtful answer using long-term memory. Unlike recall (which returns " +
      "raw memories), reflect synthesizes memories into a coherent answer. Use for questions " +
      'like "What do you know about this user?" or "Summarize our project decisions."',
    args: {
      query: tool.schema.string().describe("The question to answer using long-term memory."),
      context: tool.schema
        .string()
        .optional()
        .describe("Optional additional context to guide the reflection."),
    },
    async execute(args) {
      if (missionsSet) {
        await ensureBankMission(client, bankId, config, missionsSet, logger);
      }
      const response = await client.reflect(bankId, args.query, {
        context: args.context,
        budget: config.recallBudget as "low" | "mid" | "high",
      });

      return response.text || "No relevant information found to reflect on.";
    },
  });

  return { hindsight_retain, hindsight_operation_status, hindsight_recall, hindsight_reflect };
}
