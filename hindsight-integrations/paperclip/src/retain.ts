/**
 * Retain memories after a Paperclip agent heartbeat.
 *
 * Call this after the agent completes a task to store what it did
 * so future heartbeats can recall the context.
 */

import { HindsightClient } from './client.js';
import type { PaperclipMemoryConfig } from './config.js';
import { deriveBankId } from './bank.js';

export interface RetainInput {
  /** Paperclip company ID — used to derive the bank ID. */
  companyId: string;
  /** Paperclip agent ID — used to derive the bank ID. */
  agentId: string;
  /** The agent's output or summary of what it did during the heartbeat. */
  content: string;
  /** Paperclip run ID — used as document ID to prevent duplicate storage. */
  documentId: string;
  /** Additional metadata to store with the memory. */
  metadata?: Record<string, string>;
}

/**
 * Store the agent's output as a memory after a Paperclip task heartbeat.
 *
 * Fails silently — memory retention is an enhancement, not a requirement.
 *
 * @example
 * ```typescript
 * await retain(
 *   { companyId, agentId, content: agentOutput, documentId: runId },
 *   loadConfig()
 * )
 * ```
 */
export async function retain(
  input: RetainInput,
  config: PaperclipMemoryConfig,
): Promise<void> {
  const { companyId, agentId, content, documentId, metadata } = input;

  if (!content.trim()) return;

  const bankId = deriveBankId({ companyId, agentId }, config);
  const client = new HindsightClient(config);

  try {
    await client.retain(bankId, content, {
      documentId,
      context: config.retainContext,
      metadata: { companyId, agentId, ...metadata },
    });
  } catch (err) {
    console.warn('[hindsight-paperclip] retain failed:', (err as Error).message);
  }
}
