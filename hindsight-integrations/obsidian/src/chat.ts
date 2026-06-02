/**
 * Chat turn logic, kept UI-free so it can be unit-tested.
 *
 * DESIGN.md §0.5: conversation memory is OFF by default. When
 * `rememberConversations` is false we ONLY call `reflect` (read) and never
 * `retain` (write) — so no knowledge is created that doesn't trace back to a
 * vault note. The tests assert this guarantee explicitly.
 */

import type { HindsightClient } from "./client";
import type { Budget, ReflectResponse } from "./types";

export interface ChatTurnDeps {
  client: HindsightClient;
  bankId: string;
  budget: Budget;
  rememberConversations: boolean;
  /** Document id factory for retained turns (only used when remembering). */
  newConversationDocId?: (role: "user" | "assistant") => string;
}

function defaultDocId(role: "user" | "assistant"): string {
  return `conversation/${new Date().toISOString()}-${role}`;
}

/**
 * Run one chat turn: reflect over the whole bank and return the grounded
 * response. Retains the user/assistant turns only when conversation memory is
 * explicitly enabled.
 */
export async function runChatTurn(deps: ChatTurnDeps, message: string): Promise<ReflectResponse> {
  const genId = deps.newConversationDocId ?? defaultDocId;

  if (deps.rememberConversations) {
    await deps.client.retain(deps.bankId, genId("user"), message, {
      tags: ["conversation", "user"],
      context: "obsidian-chat",
    });
  }

  const response = await deps.client.reflect(deps.bankId, message, {
    budget: deps.budget,
    includeCitations: true,
  });

  if (deps.rememberConversations) {
    await deps.client.retain(deps.bankId, genId("assistant"), response.text, {
      tags: ["conversation", "assistant"],
      context: "obsidian-chat",
    });
  }

  return response;
}
