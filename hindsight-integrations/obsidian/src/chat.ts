/**
 * Chat turn logic, kept UI-free so it can be unit-tested.
 *
 * DESIGN.md §0.5: conversation memory is OFF by default. When
 * `rememberConversations` is false we ONLY call `reflect` (read) and never
 * `retain` (write) — so no knowledge is created that doesn't trace back to a
 * vault note. The tests assert this guarantee explicitly.
 */

import type { HindsightClient } from "./client";
import { retrievedNotes } from "./reflect-util";
import type { Budget, ReflectResponse, TagGroup } from "./types";

export interface ChatTurnDeps {
  client: HindsightClient;
  bankId: string;
  budget: Budget;
  rememberConversations: boolean;
  /** Scope filter (vault/folder) applied to reflect; undefined = whole bank. */
  tagGroups?: TagGroup[];
  /** When true, log the reflect request/response to the console for debugging. */
  debug?: boolean;
  /** Document id factory for retained turns (only used when remembering). */
  newConversationDocId?: (role: "user" | "assistant") => string;
}

function defaultDocId(role: "user" | "assistant"): string {
  return `conversation/${new Date().toISOString()}-${role}`;
}

/**
 * Run one chat turn: reflect over the (optionally scoped) bank and return the
 * grounded response. Retains the user/assistant turns only when conversation
 * memory is explicitly enabled.
 */
export async function runChatTurn(deps: ChatTurnDeps, message: string): Promise<ReflectResponse> {
  const genId = deps.newConversationDocId ?? defaultDocId;

  if (deps.rememberConversations) {
    await deps.client.retain(deps.bankId, genId("user"), message, {
      tags: ["conversation", "user"],
      context: "obsidian-chat",
    });
  }

  if (deps.debug) {
    console.log("[hindsight] reflect →", {
      bank: deps.bankId,
      query: message,
      budget: deps.budget,
      tag_groups: deps.tagGroups ?? "(none — whole bank)",
    });
  }

  const response = await deps.client.reflect(deps.bankId, message, {
    budget: deps.budget,
    includeCitations: true,
    tagGroups: deps.tagGroups,
  });

  if (deps.debug) {
    // Retrieved notes should all fall within the selected scope — this is how
    // you verify the filter actually applied.
    console.log("[hindsight] reflect ←", {
      notes_retrieved: retrievedNotes(response),
      mental_models: (response.based_on?.mental_models ?? []).length,
    });
  }

  if (deps.rememberConversations) {
    await deps.client.retain(deps.bankId, genId("assistant"), response.text, {
      tags: ["conversation", "assistant"],
      context: "obsidian-chat",
    });
  }

  return response;
}
