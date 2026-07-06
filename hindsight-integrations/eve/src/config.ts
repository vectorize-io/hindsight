/**
 * Pure config resolution + turn-pairing helpers for auto-memory. No `eve`
 * import, so precedence rules and the retain buffer are unit-testable without
 * the framework.
 */
import { stripSentinelBlocks, type RecallBudget } from "./client.js";

/** Hindsight Cloud REST base, used when no API URL is configured. */
export const HINDSIGHT_CLOUD_API_URL = "https://api.hindsight.vectorize.io";

/** Default broad recall query — surfaces the user's ambient profile/context. */
export const DEFAULT_RECALL_QUERY = "user preferences, identity, and working context";

/** Default bank when none is configured (Hindsight auto-creates it). */
export const DEFAULT_BANK_ID = "default";

export interface AutoMemoryOptions {
  /** Hindsight REST base URL. Defaults to `HINDSIGHT_API_URL`, then Cloud. */
  apiUrl?: string;
  /**
   * API key sent as `Authorization: Bearer <key>`. Defaults to `HINDSIGHT_API_KEY`.
   * Pass `null` for a no-auth self-hosted server.
   */
  apiKey?: string | null;
  /** Bank to scope memory to (REST path). Defaults to `HINDSIGHT_BANK_ID`, then `"default"`. */
  bankId?: string;
  /** Broad query used for each turn's recall injection. */
  recallQuery?: string;
  /** Recall result budget. Defaults to `"mid"`. */
  budget?: RecallBudget;
  /** Recall token budget. Defaults to `1024`. */
  maxTokens?: number;
  /** `context` tag written on retained items. Defaults to `"eve"`. */
  context?: string;
  /**
   * Also store the assistant's reply, not just the user's message. On by
   * default — the assistant's reply is usually where the answer lives (the
   * decision, the solution, the code it wrote), so both halves of the turn are
   * worth remembering. Set to `false` to retain only the user's message.
   */
  includeAssistantReply?: boolean;
  /** HTTP timeout in ms. Defaults to `15000`. */
  timeoutMs?: number;
  /** Called when a recall/retain HTTP call fails. Defaults to `console.warn`. */
  onError?: (error: unknown, phase: "recall" | "retain") => void;
}

export interface ResolvedAutoMemory {
  apiUrl: string;
  apiKey: string | null;
  bankId: string;
  recallQuery: string;
  budget: RecallBudget;
  maxTokens: number;
  context: string;
  includeAssistantReply: boolean;
  timeoutMs: number;
  onError: (error: unknown, phase: "recall" | "retain") => void;
}

/** First non-empty string among the candidates, or `null`. */
function firstNonEmpty(...values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.length > 0) return value;
  }
  return null;
}

/**
 * Whether a URL points at Hindsight Cloud, matched on host (so a trailing slash
 * or regional subdomain still triggers the missing-key guard). The dot boundary
 * avoids matching look-alikes like `nothindsight.vectorize.io`.
 */
export function isHindsightCloudUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return host === "hindsight.vectorize.io" || host.endsWith(".hindsight.vectorize.io");
  } catch {
    return false;
  }
}

/** Resolve options against env defaults. Pure; throws on Cloud + no key. */
export function resolveAutoMemory(
  options: AutoMemoryOptions = {},
  env: NodeJS.ProcessEnv = process.env
): ResolvedAutoMemory {
  const apiUrl = options.apiUrl ?? firstNonEmpty(env.HINDSIGHT_API_URL) ?? HINDSIGHT_CLOUD_API_URL;

  // `apiKey: null` is an explicit no-auth opt-out; `undefined` falls back to the env var.
  const apiKey =
    options.apiKey === undefined ? firstNonEmpty(env.HINDSIGHT_API_KEY) : options.apiKey;

  if (isHindsightCloudUrl(apiUrl) && !apiKey) {
    throw new Error(
      "Hindsight Cloud requires an API key. Set HINDSIGHT_API_KEY, pass `apiKey`, or point " +
        "`apiUrl`/HINDSIGHT_API_URL at a self-hosted server (use `apiKey: null` for a no-auth server)."
    );
  }

  return {
    apiUrl,
    apiKey,
    bankId: options.bankId ?? firstNonEmpty(env.HINDSIGHT_BANK_ID) ?? DEFAULT_BANK_ID,
    recallQuery: options.recallQuery ?? DEFAULT_RECALL_QUERY,
    budget: options.budget ?? "mid",
    maxTokens: options.maxTokens ?? 1024,
    context: options.context ?? "eve",
    includeAssistantReply: options.includeAssistantReply ?? true,
    timeoutMs: options.timeoutMs ?? 15_000,
    // eslint-disable-next-line no-console
    onError: options.onError ?? ((error) => console.warn("[hindsight-eve] memory error:", error)),
  };
}

// ---------------------------------------------------------------------------
// Turn pairing buffer — collects the user message and assistant answer for a
// turn so they can be retained together once the turn completes.
// ---------------------------------------------------------------------------

export interface TurnPair {
  user?: string;
  assistant?: string;
}
export type TurnBuffer = Map<string, TurnPair>;

/** Hard cap so a long-lived worker never leaks turns whose flush was missed. */
const MAX_BUFFERED_TURNS = 256;

function upsert(buffer: TurnBuffer, turnId: string, patch: TurnPair): void {
  const existing = buffer.get(turnId) ?? {};
  buffer.set(turnId, { ...existing, ...patch });
  if (buffer.size > MAX_BUFFERED_TURNS) {
    const oldest = buffer.keys().next().value;
    if (oldest !== undefined) buffer.delete(oldest);
  }
}

export function recordUserMessage(buffer: TurnBuffer, turnId: string, text: string): void {
  upsert(buffer, turnId, { user: text });
}

export function recordAssistantMessage(buffer: TurnBuffer, turnId: string, text: string): void {
  upsert(buffer, turnId, { assistant: text });
}

/** Remove and return a turn's buffered pair (used at flush time). */
export function takeTurn(buffer: TurnBuffer, turnId: string): TurnPair | undefined {
  const pair = buffer.get(turnId);
  buffer.delete(turnId);
  return pair;
}

/**
 * Build the retain `content` for a turn, or `null` to skip. Skips turns with no
 * user text. By default appends the assistant reply (that is usually where the
 * answer lives) with any injected recalled-context block stripped out so
 * recalled facts are never re-retained; pass `includeAssistant: false` to store
 * only the user's message.
 */
export function buildRetainContent(
  pair: TurnPair | undefined,
  includeAssistant = true
): string | null {
  const user = (pair?.user ?? "").trim();
  if (!user) return null;
  if (!includeAssistant) return `User: ${user}`;
  const assistant = stripSentinelBlocks(pair?.assistant ?? "").trim();
  return assistant ? `User: ${user}\n\nAssistant: ${assistant}` : `User: ${user}`;
}
