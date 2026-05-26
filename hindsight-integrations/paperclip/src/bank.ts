/**
 * Bank ID derivation — maps Paperclip company/agent identity onto Hindsight bank IDs.
 *
 * Default format: "paperclip::{companyId}::{agentId}"
 *
 * bankGranularity: ['company']               → "paperclip::{companyId}"
 * bankGranularity: ['agent']                 → "paperclip::{agentId}"
 * bankGranularity: ['company','agent']       → "paperclip::{companyId}::{agentId}"
 * bankGranularity: ['company','agent','user'] → "paperclip::{companyId}::{agentId}::user::{userId}"
 */

export interface BankContext {
  companyId: string;
  agentId: string;
  userId?: string;
}

export interface BankConfig {
  bankGranularity?: Array<"company" | "agent" | "user">;
}

export function deriveBankId(context: BankContext, config: BankConfig): string {
  const granularity = config.bankGranularity ?? ["company", "agent"];
  const parts: string[] = ["paperclip"];

  for (const field of granularity) {
    if (field === "company") parts.push(context.companyId);
    if (field === "agent") parts.push(context.agentId);
    if (field === "user" && context.userId) {
      parts.push("user");
      parts.push(context.userId);
    }
  }

  return parts.join("::");
}

/**
 * Extract a user identifier from a Paperclip issue.
 *
 * Paperclip issues carry an `originId` field with the format
 * "channel-key::user-email" (e.g. "slack::alice@acme.com"). We scan the
 * segments backwards for one that looks like an email. Falls back to
 * `creatorEmail` if the issue object carries it.
 *
 * Returns undefined when no user can be identified — callers treat this as
 * "user granularity not applicable for this event" (the bank ID just omits
 * the user segment).
 */
export function extractUserFromIssue(issue: {
  originId?: string | null;
  creatorEmail?: string | null;
}): string | undefined {
  // Prefer explicit creatorEmail when available
  if (issue.creatorEmail) return issue.creatorEmail;

  if (!issue.originId) return undefined;

  // originId format: "channel-key::user-email" — scan backwards for an email
  const parts = issue.originId.split("::");
  for (let i = parts.length - 1; i >= 0; i--) {
    if (parts[i].includes("@")) return parts[i];
  }
  return undefined;
}
