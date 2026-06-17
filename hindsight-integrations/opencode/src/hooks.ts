/**
 * Hook implementations for the Hindsight OpenCode plugin.
 *
 * Hooks:
 *   - experimental.chat.system.transform → recall memories once per session and
 *     inject them into the system prompt (order-independent; see #1758)
 *   - event (session.idle) → auto-retain conversation transcript
 *   - experimental.session.compacting → inject memories into compaction context
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
  sliceLastTurnsByUserBoundary,
  type Message,
} from "./content.js";
import { ensureBankMission } from "./bank.js";

export interface PluginState {
  turnCount: number;
  missionsSet: Set<string>;
  /** Track sessions we've already injected recall into */
  recalledSessions: Set<string>;
  /** Track last retained turn count per session to avoid duplicates */
  lastRetainedTurn: Map<string, number>;
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
        `${config.recallPromptPreamble}\n` +
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
   * Retain messages for a session, respecting retainMode and documentId semantics.
   * Used by both idle-retain and pre-compaction retain.
   */
  async function retainSession(sessionId: string, messages: Message[]): Promise<void> {
    const retainFullWindow = config.retainMode === "full-session";
    let targetMessages: Message[];
    let documentId: string;

    if (retainFullWindow) {
      targetMessages = messages;
      // Full-session upserts the same document each time
      documentId = sessionId;
    } else {
      // Sliding window: retainEveryNTurns + overlap
      const windowTurns = config.retainEveryNTurns + config.retainOverlapTurns;
      targetMessages = sliceLastTurnsByUserBoundary(messages, windowTurns);
      // Chunked mode: unique document per chunk
      documentId = `${sessionId}-${Date.now()}`;
    }

    const { transcript } = prepareRetentionTranscript(targetMessages, true);
    if (!transcript) return;

    await ensureBankMission(hindsightClient, bankId, config, state.missionsSet, logger);
    await hindsightClient.retain(bankId, transcript, {
      documentId,
      context: config.retainContext,
      tags: config.retainTags.length ? config.retainTags : undefined,
      metadata: Object.keys(config.retainMetadata).length
        ? { ...config.retainMetadata, session_id: sessionId }
        : { session_id: sessionId },
      async: true,
    });
  }

  /** Auto-retain conversation transcript */
  async function handleSessionIdle(sessionId: string): Promise<void> {
    logger.debug(`handleSessionIdle called for session ${sessionId}`);
    if (!config.autoRetain) return;

    const messages = await getSessionMessages(sessionId);
    if (!messages.length) return;

    // Count user turns
    const userTurns = messages.filter((m) => m.role === "user").length;
    const lastRetained = state.lastRetainedTurn.get(sessionId) || 0;
    logger.debug(
      `handleSessionIdle: userTurns=${userTurns}, lastRetained=${lastRetained}, retainEveryNTurns=${config.retainEveryNTurns}`
    );

    // Only retain if enough new turns since last retain
    if (userTurns - lastRetained < config.retainEveryNTurns) return;

    try {
      await retainSession(sessionId, messages);
      state.lastRetainedTurn.set(sessionId, userTurns);
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
      // NOTE: autoRecall is driven entirely by `experimental.chat.system.transform`
      // (see below). We deliberately do NOT key it off `session.created` — the
      // relative firing order of `session.created` vs `system.transform` is an
      // undocumented OpenCode implementation detail that has differed between
      // versions, and relying on it silently disabled recall (see #1758).
    } catch (e) {
      logger.error("Event hook error", e);
    }
  };

  const compacting = async (input: CompactingInput, output: CompactingOutput): Promise<void> => {
    try {
      // First, retain what we have before compaction (using shared retention logic)
      const messages = await getSessionMessages(input.sessionID);
      if (messages.length && config.autoRetain) {
        try {
          await retainSession(input.sessionID, messages);
          // Reset turn tracking — after compaction the message list shrinks,
          // so the old lastRetainedTurn value would block future idle retains.
          state.lastRetainedTurn.delete(input.sessionID);
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
      if (!config.autoRecall) return;
      const sessionId = input.sessionID;
      if (!sessionId) return;

      // Recall once per session, on the first system.transform we see for it.
      // `recalledSessions` is a dedup marker for sessions we've ALREADY recalled
      // into — not an event-ordering gate. This makes autoRecall independent of
      // whether session.created fired first (see #1758).
      if (state.recalledSessions.has(sessionId)) return;

      await ensureBankMission(hindsightClient, bankId, config, state.missionsSet, logger);

      // Use a generic project-context query for session start
      const query = `project context and recent work`;
      const { context, ok } = await recallForContext(query);

      // Mark as recalled only after a successful API round-trip (even with 0
      // results), so transient failures retry on the next message.
      if (ok) {
        state.recalledSessions.add(sessionId);
        // Cap tracked sessions
        if (state.recalledSessions.size > 1000) {
          const first = state.recalledSessions.values().next().value;
          if (first) state.recalledSessions.delete(first);
        }
      }

      if (context) {
        // Fold recall into the FIRST system section rather than pushing a new
        // one. OpenCode emits each system[] entry as a separate system message,
        // and some providers/LLMs only honor the first — a pushed section can be
        // silently dropped. Appending to system[0] guarantees recall is seen.
        // (Original approach from #1988 by @sdrobov.)
        output.system[0] = output.system[0] ? `${output.system[0]}\n\n${context}` : context;
        logger.debug(`Injected recall context for session ${sessionId}`);
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
