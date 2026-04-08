/**
 * Bank ID derivation for Paperclip agents.
 *
 * Aligns Hindsight's memory bank model with Paperclip's company/agent isolation.
 */

import type { PaperclipMemoryConfig } from './config.js';

export interface BankContext {
  companyId: string;
  agentId: string;
}

/**
 * Derive a Hindsight bank ID from Paperclip context.
 *
 * Default output: "paperclip::{companyId}::{agentId}"
 *
 * With bankGranularity: ['company'] → "paperclip::{companyId}"
 * With bankGranularity: ['agent']   → "paperclip::{agentId}"
 * With bankIdPrefix: ''              → "{companyId}::{agentId}"
 */
export function deriveBankId(context: BankContext, config: PaperclipMemoryConfig): string {
  const parts: string[] = [];

  if (config.bankIdPrefix) {
    parts.push(config.bankIdPrefix);
  }

  for (const field of config.bankGranularity ?? ['company', 'agent']) {
    if (field === 'company') parts.push(context.companyId);
    if (field === 'agent') parts.push(context.agentId);
  }

  if (parts.length === 0) {
    throw new Error('Bank ID cannot be empty — bankGranularity or bankIdPrefix must be set');
  }

  return parts.join('::');
}
