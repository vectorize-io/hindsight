/**
 * Recall memories for a Paperclip agent heartbeat.
 *
 * Call this before the agent processes a task to inject relevant context
 * from prior heartbeats and sessions.
 */

import { HindsightClient } from './client.js';
import type { PaperclipMemoryConfig } from './config.js';
import { deriveBankId } from './bank.js';

export interface RecallInput {
  /** Paperclip company ID — used to derive the bank ID. */
  companyId: string;
  /** Paperclip agent ID — used to derive the bank ID. */
  agentId: string;
  /**
   * Query string for memory retrieval. Typically the task title + description.
   * e.g. `${issue.title}\n${issue.description}`
   */
  query: string;
}

/**
 * Retrieve relevant memories for the current Paperclip task.
 *
 * Returns a formatted string of memories to inject into the agent's prompt,
 * or an empty string if no relevant memories are found.
 *
 * @example
 * ```typescript
 * const memories = await recall(
 *   { companyId, agentId, query: `${task.title}\n${task.description}` },
 *   loadConfig()
 * )
 * if (memories) {
 *   systemPrompt = `Past context:\n${memories}\n\n${systemPrompt}`
 * }
 * ```
 */
export async function recall(
  input: RecallInput,
  config: PaperclipMemoryConfig,
): Promise<string> {
  const { companyId, agentId, query } = input;

  if (!query.trim()) return '';

  const bankId = deriveBankId({ companyId, agentId }, config);
  const client = new HindsightClient(config);

  let results: Array<{ text: string; type?: string; mentionedAt?: string }>;
  try {
    const response = await client.recall(bankId, query, {
      budget: config.recallBudget,
      maxTokens: config.recallMaxTokens,
    });
    results = response.results;
  } catch (err) {
    console.warn('[hindsight-paperclip] recall failed:', (err as Error).message);
    return '';
  }

  if (!results.length) return '';

  return formatMemories(results);
}

function formatMemories(
  results: Array<{ text: string; type?: string; mentionedAt?: string }>,
): string {
  return results
    .map((r) => {
      const typeStr = r.type ? ` [${r.type}]` : '';
      const dateStr = r.mentionedAt ? ` (${r.mentionedAt})` : '';
      return `- ${r.text}${typeStr}${dateStr}`;
    })
    .join('\n\n');
}
