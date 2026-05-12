/**
 * Bank ID derivation — maps Paperclip company/agent identity onto Hindsight bank IDs.
 *
 * Default format: "paperclip::{companyId}::{agentId}"
 *
 * bankGranularity: ['company']        → "paperclip::{companyId}"
 * bankGranularity: ['agent']          → "paperclip::{agentId}"
 * bankGranularity: ['company','agent'] → "paperclip::{companyId}::{agentId}"
 *
 * sharedBankName overrides both granularity-based derivation and the
 * "paperclip" prefix entirely — every agent in the company points at the
 * named bank. Useful when multiple agents must collaborate on a single
 * pool of memories (e.g. a CEO/CTO/Staff cohort sharing project context).
 */

export interface BankContext {
  companyId: string;
  agentId: string;
}

export interface BankConfig {
  /**
   * Fixed bank name override. When set (non-empty after trim), `deriveBankId`
   * returns this value verbatim and ignores `bankGranularity` + identity.
   */
  sharedBankName?: string;
  bankGranularity?: Array<"company" | "agent">;
}

export function deriveBankId(context: BankContext, config: BankConfig): string {
  if (typeof config.sharedBankName === "string" && config.sharedBankName.trim()) {
    return config.sharedBankName.trim();
  }

  const granularity = config.bankGranularity ?? ["company", "agent"];
  const parts: string[] = ["paperclip"];

  for (const field of granularity) {
    if (field === "company") parts.push(context.companyId);
    if (field === "agent") parts.push(context.agentId);
  }

  return parts.join("::");
}
