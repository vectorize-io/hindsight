import type { AgentMemoryProvider } from "@framers/agentos";
import type { HindsightClient, RecallResult } from "./client.js";
import { DEFAULT_BANK, type HindsightMemoryOptions } from "./options.js";

const DEFAULT_HEADING = "# Relevant long-term memories";

/** Render recalled memories as a markdown bullet list for context injection. */
function formatMemories(
  results: RecallResult[],
  heading: string,
  labelTypes: boolean
): string {
  const lines = results
    .map((r) => {
      const text = r.text?.trim();
      if (!text) return undefined;
      const label = labelTypes && r.type ? `[${r.type}] ` : "";
      return `- ${label}${text}`;
    })
    .filter((line): line is string => Boolean(line));
  if (lines.length === 0) return "";
  return `${heading}\n${lines.join("\n")}`;
}

/**
 * Creates an {@link AgentMemoryProvider} that gives an AgentOS agent long-term
 * memory backed by Hindsight.
 *
 * Wire the returned provider into an agent via `memoryProvider`; AgentOS then
 * auto-invokes it on every call path:
 *
 * - `getContext` runs **before** each model call — it recalls relevant memories
 *   from Hindsight and returns them as a context block that AgentOS injects into
 *   the system prompt.
 * - `observe` runs **after** each turn (once for the user turn, once for the
 *   assistant reply) — it retains the turn to Hindsight so entity extraction and
 *   consolidation happen server-side.
 *
 * Both sides fail safe: a recall failure returns no context (the agent still
 * responds) and retains are fire-and-forget by default so they never add turn
 * latency. Disable either side with `recall.enabled: false` /
 * `retain.enabled: false`.
 *
 * @example
 * ```ts
 * import { agent } from "@framers/agentos";
 * import { createHindsightMemory } from "@vectorize-io/hindsight-agentos";
 * import { Hindsight } from "@vectorize-io/hindsight-client";
 *
 * const memory = createHindsightMemory({
 *   client: new Hindsight({ apiKey: process.env.HINDSIGHT_API_KEY }),
 *   bank: "ada",
 *   recall: { budget: "high", includeEntities: true },
 *   retain: { tags: ["source:agentos"] },
 * });
 *
 * const ada = agent({ name: "Ada", memoryProvider: memory });
 * ```
 */
export function createHindsightMemory(
  options: HindsightMemoryOptions
): AgentMemoryProvider {
  const { client, bank = DEFAULT_BANK, recall = {}, retain = {} } = options;

  const provider: AgentMemoryProvider = {};

  if (recall.enabled !== false) {
    const heading = recall.heading ?? DEFAULT_HEADING;
    provider.getContext = async (text, opts) => {
      const query = text?.trim();
      if (!query) return null;
      try {
        const response = await client.recall(bank, query, {
          types: recall.types,
          maxTokens: recall.maxTokens ?? opts?.tokenBudget,
          budget: recall.budget,
          includeEntities: recall.includeEntities,
        });
        const results = response.results ?? [];
        const contextText = formatMemories(results, heading, recall.labelTypes ?? false);
        return contextText ? { contextText } : null;
      } catch {
        // Recall failure is non-fatal; the agent proceeds without memory.
        return null;
      }
    };
  }

  if (retain.enabled !== false) {
    const isAsync = retain.async ?? true;
    provider.observe = async (role, text) => {
      const content = text?.trim();
      if (!content) return;
      if (role === "assistant" && !retain.includeAgentMessages) return;

      const write = client
        .retain(bank, content, {
          async: isAsync,
          tags: retain.tags,
          metadata: retain.metadata,
        })
        .catch(() => undefined);

      // Fire-and-forget in async mode so retain never adds turn latency;
      // await only when the caller opted into synchronous writes.
      if (!isAsync) await write;
    };
  }

  return provider;
}
