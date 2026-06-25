/**
 * Hook implementations for the Hindsight OpenCode plugin.
 *
 * Hooks:
 *   - experimental.chat.system.transform → recall memories on **every turn** and
 *     inject them into the system prompt with contextual, query-based recall
 *   - event (session.idle) → auto-retain conversation transcript (fires after
 *     each agent response, upserting the same document so the latest state wins)
 *   - experimental.session.compacting → retain + inject memories into compaction context
 *
 * Key design decisions:
 *
 * 1. **Recall every turn** — we recall on every `system.transform`, keyed on
 *    the latest user message. This ensures injected memories are always relevant
 *    to the current question.
 *
 * 2. **Retain on session idle** — `session.idle` fires after each agent
 *    response. We always retain the full conversation (upsert with the same
 *    documentId), so the latest state is stored reliably, including for
 *    one-shot prompts. A pre-compaction retain serves as a backup before
 *    context is compressed.
 *
 * 3. **System prompt instructions** — we inject memory usage instructions
 *    into the system prompt so the LLM knows when and how to use memory tools.
 */

import type { HindsightClient } from "@vectorize-io/hindsight-client";
import type { HindsightConfig } from "./config.js";
import { Logger } from "./logger.js";
import {
  formatMemories,
  formatCurrentTime,
  composeRecallQuery,
  truncateRecallQuery,
  prepareRetentionTranscript,
  stripMemoryTags,
  type Message,
} from "./content.js";
import { ensureBankMission } from "./bank.js";

export interface PluginState {
  turnCount: number;
  missionsSet: Set<string>;
  /** Per-session turn counter — incremented on each system.transform */
  sessionTurnCount: Map<string, number>;
}

interface EventInput {
  event: {
    type: string;
    properties: Record<string, unknown>;
  };
}

interface CompactingInput {
  sessionID: string;
}

interface CompactingOutput {
  context: string[];
  prompt?: string;
}

interface SystemTransformInput {
  sessionID?: string;
  model: unknown;
}

interface SystemTransformOutput {
  system: string[];
}

type OpencodeClient = {
  session: {
    messages: (params: { path: { id: string } }) => Promise<{
      data?: Array<{
        info: { role: string };
        parts: Array<{ type: string; text?: string }>;
      }>;
      error?: unknown;
      request?: unknown;
      response?: unknown;
    }>;
  };
};

export interface HindsightHooks {
  event: (input: EventInput) => Promise<void>;
  "experimental.session.compacting": (
    input: CompactingInput,
    output: CompactingOutput
  ) => Promise<void>;
  "experimental.chat.system.transform": (
    input: SystemTransformInput,
    output: SystemTransformOutput
  ) => Promise<void>;
}

/**
 * System prompt instructions that teach the LLM to use memory tools
 * proactively. Combines hermes-agent's memory tool description style
 * (motivational, concrete examples) with Hindsight-specific tool names.
 */
function buildMemoryInstructions(bankId: string): string {
  return [
    "# Hindsight Memory",
    `Active. Bank: ${bankId}.`,
    "Relevant memories are automatically injected into context.",
    "",
    "Save durable information to persistent memory that survives across sessions. " +
      "Memory is injected into future turns, so keep it compact and focused on facts " +
      "that will still matter later.",
    "",
    "WHEN TO SAVE (do this proactively, don't wait to be asked):",
    "- User corrects you or says 'remember this' / 'don't do that again'",
    "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)",
    "- You discover something about the environment (OS, installed tools, project structure)",
    "- You learn a convention, API quirk, or workflow specific to this user's setup",
    "- You identify a stable fact that will be useful again in future sessions",
    "- After completing a task or making a decision",
    "",
    "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. " +
      "The most valuable memory prevents the user from having to repeat themselves.",
    "",
    "Use hindsight_recall to search for relevant memories before answering questions about " +
      "past work, user preferences, or project context. When in doubt, recall first.",
    "Use hindsight_reflect to synthesize a reasoned answer from all stored memories.",
    "Use hindsight_retain to store facts, decisions, and preferences.",
  ].join("\n");
}

export function createHooks(
  hindsightClient: HindsightClient,
  bankId: string,
  config: HindsightConfig,
  state: PluginState,
  opencodeClient: OpencodeClient,
  logger: Logger = new Logger({ silent: true })
): HindsightHooks {
  interface RecallOutcome {
    /** formatted context string, or null if no results */
    context: string | null;
    /** true if the API call succeeded (even with 0 results) */
    ok: boolean;
  }

  /** Recall memories and format as context string */
  async function recallForContext(query: string): Promise<RecallOutcome> {
    try {
      const response = await hindsightClient.recall(bankId, query, {
        budget: config.recallBudget as "low" | "mid" | "high",
        maxTokens: config.recallMaxTokens,
        types: config.recallTypes,
        tags: config.recallTags.length ? config.recallTags : undefined,
        tagsMatch: config.recallTags.length ? config.recallTagsMatch : undefined,
      });

      const results = response.results || [];
      if (!results.length) return { context: null, ok: true };

      const formatted = formatMemories(results);
      const context =
        `<hindsight_memories>\n` +
        `Hindsight Memory (persistent cross-session context)\n` +
        `Use this to answer questions about the user and prior sessions. ` +
        `Do not call tools to look up information that is already present here.\n` +
        `Current time: ${formatCurrentTime()} UTC\n\n` +
        `${formatted}\n` +
        `</hindsight_memories>`;
      return { context, ok: true };
    } catch (e) {
      logger.error("Recall failed", e);
      return { context: null, ok: false };
    }
  }

  /** Extract plain-text messages from an OpenCode session */
  async function getSessionMessages(sessionId: string): Promise<Message[]> {
    try {
      logger.debug(`getSessionMessages: fetching messages for session ${sessionId}`);
      const response = await opencodeClient.session.messages({
        path: { id: sessionId },
      });
      if (response.error) {
        logger.warn("getSessionMessages: OpenCode returned an error", {
          error: JSON.stringify(response.error)?.substring(0, 500),
        });
      }
      const rawMessages = response.data || [];
      const messages: Message[] = [];
      for (const msg of rawMessages) {
        const role = msg.info.role;
        if (role !== "user" && role !== "assistant") continue;
        const textParts = msg.parts.filter((p) => p.type === "text" && p.text).map((p) => p.text!);
        if (textParts.length) {
          messages.push({ role, content: textParts.join("\n") });
        }
      }
      logger.debug(`getSessionMessages: raw=${rawMessages.length}, parsed=${messages.length}`);
      return messages;
    } catch (e) {
      logger.error("Failed to get session messages", e);
      return [];
    }
  }

  /**
   * Retain the full conversation for a session as a single document (upsert).
   * Used by both idle-retain and pre-compaction retain.
   */
  async function retainSession(sessionId: string, messages: Message[]): Promise<void> {
    const transcript = prepareRetentionTranscript(messages, true);
    if (!transcript) return;

    await ensureBankMission(hindsightClient, bankId, config, state.missionsSet, logger);
    await hindsightClient.retain(bankId, transcript, {
      documentId: sessionId,
      context: config.retainContext,
      tags: config.retainTags.length ? config.retainTags : undefined,
      metadata: Object.keys(config.retainMetadata).length
        ? { ...config.retainMetadata, session_id: sessionId }
        : { session_id: sessionId },
      async: true,
    });
  }

  /** Auto-retain conversation transcript on session idle */
  async function handleSessionIdle(sessionId: string): Promise<void> {
    logger.debug(`handleSessionIdle called for session ${sessionId}`);
    if (!config.autoRetain) return;

    const messages = await getSessionMessages(sessionId);
    if (!messages.length) return;

    try {
      await retainSession(sessionId, messages);
      logger.info(`Auto-retained ${messages.length} messages`, {
        session: sessionId,
        bank: bankId,
      });
    } catch (e) {
      logger.error("Auto-retain failed", e);
    }
  }

  const event = async (input: EventInput): Promise<void> => {
    try {
      const { event: evt } = input;
      logger.debug(`event hook fired: type=${evt.type}`);

      if (evt.type === "session.idle") {
        const sessionId = (evt.properties as { sessionID?: string }).sessionID;
        if (sessionId) {
          await handleSessionIdle(sessionId);
        }
      }
    } catch (e) {
      logger.error("Event hook error", e);
    }
  };

  const compacting = async (input: CompactingInput, output: CompactingOutput): Promise<void> => {
    try {
      // First, retain what we have before compaction
      const messages = await getSessionMessages(input.sessionID);
      if (messages.length && config.autoRetain) {
        try {
          await retainSession(input.sessionID, messages);
          // Reset turn tracking — after compaction the message list shrinks.
          state.sessionTurnCount.delete(input.sessionID);
          logger.debug("Pre-compaction retain completed");
        } catch (e) {
          logger.error("Pre-compaction retain failed", e);
        }
      }

      // Then recall relevant memories to inject into compaction context
      if (messages.length) {
        const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
        if (lastUserMsg) {
          const query = composeRecallQuery(
            lastUserMsg.content,
            messages,
            config.recallContextTurns
          );
          const truncated = truncateRecallQuery(
            query,
            lastUserMsg.content,
            config.recallMaxQueryChars
          );
          const { context } = await recallForContext(truncated);
          if (context) {
            output.context.push(context);
          }
        }
      }
    } catch (e) {
      logger.error("Compaction hook error", e);
    }
  };

  const systemTransform = async (
    input: SystemTransformInput,
    output: SystemTransformOutput
  ): Promise<void> => {
    try {
      const sessionId = input.sessionID;
      if (!sessionId) return;

      if (!config.autoRecall) {
        // Still count turns for state consistency, but skip all recall work
        const turnNum = (state.sessionTurnCount.get(sessionId) || 0) + 1;
        state.sessionTurnCount.set(sessionId, turnNum);
        return;
      }

      // Increment per-session turn counter
      const turnNum = (state.sessionTurnCount.get(sessionId) || 0) + 1;
      state.sessionTurnCount.set(sessionId, turnNum);

      // Inject memory instructions into system prompt on the first turn only
      if (turnNum === 1) {
        const instructions = buildMemoryInstructions(bankId);
        output.system[0] = output.system[0]
          ? `${output.system[0]}\n\n${instructions}`
          : instructions;
        logger.debug(`Injected memory instructions for session ${sessionId}`);
      }

      await ensureBankMission(hindsightClient, bankId, config, state.missionsSet, logger);

      // Build a contextual recall query from the current conversation
      const messages = await getSessionMessages(sessionId);
      if (!messages.length) return;

      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      if (!lastUserMsg) return;

      // Strip injected memory tags from the query to avoid feedback loops
      const cleanUserMsg = stripMemoryTags(lastUserMsg.content);
      const query = composeRecallQuery(cleanUserMsg, messages, config.recallContextTurns);
      const truncated = truncateRecallQuery(query, cleanUserMsg, config.recallMaxQueryChars);

      // Always do a live recall
      const { context } = await recallForContext(truncated);
      if (context) {
        output.system[0] = output.system[0] ? `${output.system[0]}\n\n${context}` : context;
        logger.debug(`Injected recall context for session ${sessionId} turn ${turnNum}`);
      }
    } catch (e) {
      logger.error("System transform hook error", e);
    }
  };

  return {
    event,
    "experimental.session.compacting": compacting,
    "experimental.chat.system.transform": systemTransform,
  };
}
