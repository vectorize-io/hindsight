export type ScopedToken = {
  token: string;
  prefix: string;
  label?: string;
};

export type ResolvedToken = {
  prefix: string;
  label?: string;
};

let cachedRaw: string | undefined;
let cachedTokens: ScopedToken[] = [];
let warnedInvalid = false;

/**
 * Parse `HINDSIGHT_CP_TOKENS` (JSON array of `{token, prefix, label?}`). Any
 * malformed configuration (unset, empty, invalid JSON, not an array) resolves
 * to no scoped tokens, logging a single warning on invalid JSON/shape so a
 * broken env doesn't spam logs on every request. Entries missing `token` or
 * `prefix` are ignored.
 */
function loadScopedTokens(): ScopedToken[] {
  const raw = process.env.HINDSIGHT_CP_TOKENS;
  if (raw === cachedRaw) return cachedTokens;

  cachedRaw = raw;
  cachedTokens = [];

  if (!raw || raw.trim().length === 0) return cachedTokens;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    if (!warnedInvalid) {
      console.warn("HINDSIGHT_CP_TOKENS is not valid JSON; ignoring scoped tokens.");
      warnedInvalid = true;
    }
    return cachedTokens;
  }

  if (!Array.isArray(parsed)) {
    if (!warnedInvalid) {
      console.warn("HINDSIGHT_CP_TOKENS must be a JSON array; ignoring scoped tokens.");
      warnedInvalid = true;
    }
    return cachedTokens;
  }

  cachedTokens = parsed.flatMap((entry): ScopedToken[] => {
    if (!entry || typeof entry !== "object") return [];
    const { token, prefix, label } = entry as Record<string, unknown>;
    if (typeof token !== "string" || token.length === 0) return [];
    if (typeof prefix !== "string") return [];
    return [{ token, prefix, label: typeof label === "string" ? label : undefined }];
  });

  return cachedTokens;
}

/**
 * Resolve a provided key to its bank prefix scope. The admin key
 * (`HINDSIGHT_CP_ACCESS_KEY`) maps to the empty prefix (all banks). Scoped
 * tokens map to their configured prefix. Returns null when nothing matches.
 *
 * Comparison is constant-time and every candidate is checked (no early return
 * on the first match) so response timing does not leak which token matched.
 */
export function resolveToken(provided: string | undefined): ResolvedToken | null {
  if (!provided) return null;

  let match: ResolvedToken | null = null;

  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
  if (accessKey && constantTimeEqual(provided, accessKey)) {
    match = { prefix: "", label: "admin" };
  }

  for (const entry of loadScopedTokens()) {
    if (constantTimeEqual(provided, entry.token)) {
      match = match ?? { prefix: entry.prefix, label: entry.label };
    }
  }

  return match;
}

/**
 * Prefix-scope predicate reused by the list filter, middleware, and body guard.
 * Empty prefix (admin) always passes. Otherwise the bank must equal the prefix
 * exactly or be a namespaced child (`<prefix>--...`), so "u2" does not match
 * "u20".
 */
export function bankAllowed(prefix: string, bankId: string): boolean {
  if (prefix === "") return true;
  return bankId === prefix || bankId.startsWith(`${prefix}--`);
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
