/**
 * @vectorize-io/hindsight-paperclip
 *
 * Persistent memory for Paperclip AI agents using Hindsight.
 *
 * @example
 * ```typescript
 * import { recall, retain, loadConfig } from '@vectorize-io/hindsight-paperclip'
 *
 * const config = loadConfig()
 *
 * // Before heartbeat
 * const memories = await recall({ companyId, agentId, query }, config)
 *
 * // After heartbeat
 * await retain({ companyId, agentId, content: output, documentId: runId }, config)
 * ```
 */

export { recall } from './recall.js';
export type { RecallInput } from './recall.js';

export { retain } from './retain.js';
export type { RetainInput } from './retain.js';

export { createMemoryMiddleware } from './middleware.js';
export type { HindsightRequest } from './middleware.js';

export { deriveBankId } from './bank.js';
export type { BankContext } from './bank.js';

export { loadConfig } from './config.js';
export type { PaperclipMemoryConfig, BankGranularity } from './config.js';

export { HindsightClient } from './client.js';
export type { Memory, RecallResponse, RetainResponse } from './client.js';
