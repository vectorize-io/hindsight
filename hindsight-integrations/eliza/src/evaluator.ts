import type { Evaluator, IAgentRuntime, Memory } from "@elizaos/core";
import type { HindsightClient } from "./client.js";
import { type BankResolver, type RetainOptions, resolveBank } from "./options.js";

/**
 * Builds the retain evaluator. Evaluators run after the agent has processed a
 * turn, which is the natural moment to persist new memories. By default it
 * stores the incoming message; set `includeAgentMessages` to also store the
 * agent's replies. Retain is fire-and-forget by default so it never adds
 * latency to the conversation.
 */
export function createHindsightEvaluator(
  client: HindsightClient,
  bank: BankResolver | undefined,
  options: RetainOptions = {}
): Evaluator {
  const isAsync = options.async ?? true;

  const retain = (bankId: string, text: string): Promise<unknown> => {
    const promise = client
      .retain(bankId, text, {
        async: isAsync,
        tags: options.tags,
        metadata: options.metadata,
      })
      .catch(() => undefined);
    return isAsync ? Promise.resolve() : promise;
  };

  return {
    name: "HINDSIGHT_RETAIN",
    description: "Persists conversation messages to Hindsight long-term memory.",
    alwaysRun: true,
    examples: [],
    validate: async (_runtime: IAgentRuntime, message: Memory) => {
      return Boolean(message.content?.text?.trim());
    },
    handler: async (
      runtime: IAgentRuntime,
      message: Memory,
      _state,
      _options,
      _callback,
      responses?: Memory[]
    ) => {
      const bankId = resolveBank(bank, message);
      const fromAgent = message.entityId === runtime.agentId;

      // Store the triggering message unless it is the agent's own and the
      // caller opted out of retaining agent messages.
      const text = message.content?.text?.trim();
      if (text && (!fromAgent || options.includeAgentMessages)) {
        await retain(bankId, text);
      }

      // Optionally store the agent's replies for this turn.
      if (options.includeAgentMessages && responses?.length) {
        for (const response of responses) {
          const responseText = response.content?.text?.trim();
          if (responseText) await retain(bankId, responseText);
        }
      }
    },
  };
}
