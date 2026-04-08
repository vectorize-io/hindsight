/**
 * Configuration for @vectorize-io/hindsight-paperclip.
 *
 * Loaded from explicit options first, then environment variables.
 */

export type BankGranularity = 'company' | 'agent';

export interface PaperclipMemoryConfig {
  /** Hindsight server URL. Required. env: HINDSIGHT_API_URL */
  hindsightApiUrl: string;
  /** API token for Hindsight Cloud. env: HINDSIGHT_API_TOKEN */
  hindsightApiToken?: string;
  /**
   * Which dimensions to include in the bank ID.
   * Default: ['company', 'agent'] → "paperclip::{companyId}::{agentId}"
   */
  bankGranularity?: BankGranularity[];
  /** Prefix prepended to all bank IDs. Default: "paperclip" */
  bankIdPrefix?: string;
  /** Recall search depth. Default: "mid" */
  recallBudget?: 'low' | 'mid' | 'high';
  /** Max tokens in the recalled memory block. Default: 1024 */
  recallMaxTokens?: number;
  /** Provenance label stored with each retained document. Default: "paperclip" */
  retainContext?: string;
  /** Request timeout in milliseconds. Default: 15000 */
  timeoutMs?: number;
}

export function loadConfig(overrides?: Partial<PaperclipMemoryConfig>): PaperclipMemoryConfig {
  const config: PaperclipMemoryConfig = {
    hindsightApiUrl: process.env['HINDSIGHT_API_URL'] ?? '',
    hindsightApiToken: process.env['HINDSIGHT_API_TOKEN'],
    bankGranularity: ['company', 'agent'],
    bankIdPrefix: 'paperclip',
    recallBudget: 'mid',
    recallMaxTokens: 1024,
    retainContext: 'paperclip',
    timeoutMs: 15_000,
    ...overrides,
  };
  if (!config.hindsightApiUrl) {
    throw new Error(
      'hindsightApiUrl is required — set HINDSIGHT_API_URL or pass hindsightApiUrl to loadConfig()',
    );
  }
  return config;
}
