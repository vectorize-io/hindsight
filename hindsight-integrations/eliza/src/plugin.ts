import type { Plugin } from "@elizaos/core";
import { createHindsightEvaluator } from "./evaluator.js";
import type { HindsightPluginOptions } from "./options.js";
import { createHindsightProvider } from "./provider.js";

/**
 * Creates an elizaOS plugin that gives an agent long-term memory backed by
 * Hindsight.
 *
 * - A **provider** recalls relevant memories and injects them into the prompt
 *   before each model call.
 * - An **evaluator** retains conversation messages after each turn.
 *
 * Both sides default to enabled. Disable either via `recall.enabled` /
 * `retain.enabled`.
 *
 * @example
 * ```ts
 * import { createHindsightPlugin } from "@vectorize-io/hindsight-eliza";
 * import { Hindsight } from "@vectorize-io/hindsight-client";
 *
 * const plugin = createHindsightPlugin({
 *   client: new Hindsight({ apiKey: process.env.HINDSIGHT_API_KEY }),
 *   recall: { budget: "high", includeEntities: true },
 *   retain: { tags: ["source:eliza"] },
 * });
 * ```
 */
export function createHindsightPlugin(options: HindsightPluginOptions): Plugin {
  const { client, bank, recall = {}, retain = {} } = options;

  const providers = recall.enabled === false ? [] : [createHindsightProvider(client, bank, recall)];
  const evaluators =
    retain.enabled === false ? [] : [createHindsightEvaluator(client, bank, retain)];

  return {
    name: "@vectorize-io/hindsight-eliza",
    description: "Hindsight long-term memory: recall relevant memories and retain conversations.",
    providers,
    evaluators,
  };
}
