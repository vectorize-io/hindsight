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
 * broken env doesn't spam logs on every request. Entries missing `token` are
 * ignored. An entry with an empty or non-string `prefix` is rejected (not
 * silently admitted): the empty prefix is the admin scope, so accepting a
 * config typo like `{"token":"x","prefix":""}` would hand out full cross-bank
 * access. Such an entry is dropped with a warning.
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
    if (typeof prefix !== "string" || prefix.length === 0) {
      if (!warnedInvalid) {
        console.warn(
          "HINDSIGHT_CP_TOKENS entry has an empty or non-string prefix; dropping it " +
            "(an empty prefix is the admin scope and would grant full cross-bank access)."
        );
        warnedInvalid = true;
      }
      return [];
    }
    return [{ token, prefix, label: typeof label === "string" ? label : undefined }];
  });

  return cachedTokens;
}

/**
 * Resolve a provided key to its bank prefix scope. The admin key
 * (`HINDSIGHT_CP_ACCESS_KEY`) maps to the empty prefix (all banks). Scoped
 * tokens map to their configured prefix. Returns null when nothing matches.
 *
 * Every candidate is checked (no early return on the first match) so response
 * timing does not leak which same-length token matched. `constantTimeEqual`
 * short-circuits on a length mismatch, so token length is not hidden — that's
 * the standard tradeoff and not sensitive here (tokens are high-entropy
 * secrets, not guessable from their length).
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
 * Human-readable label for a resolved prefix, for UI display. Empty prefix is
 * the admin scope; otherwise the first configured token entry with that prefix
 * supplies the label (may be undefined).
 */
export function labelForPrefix(prefix: string): string | undefined {
  if (prefix === "") return "admin";
  for (const entry of loadScopedTokens()) {
    if (entry.prefix === prefix) return entry.label;
  }
  return undefined;
}

/**
 * Prefix-scope predicate reused by the list filter, middleware, and body guard.
 * Empty prefix (admin) always passes. Otherwise the bank must equal the prefix
 * exactly or be a namespaced child (`<prefix>--...`), so "u2" does not match
 * "u20".
 *
 * A bank id containing a path separator or a `..` segment is rejected outright,
 * even under the admin scope. Bank ids are interpolated into dataplane URL paths
 * downstream; without this guard a value like "u2--x/../victim" would pass the
 * `startsWith("u2--")` check here and then, once WHATWG URL normalization
 * collapses the dot segment, resolve to a foreign bank. Making the predicate
 * safe by construction means a route that builds its URL without
 * `dataplaneBankUrl()` can't silently reintroduce the traversal.
 */
export function bankAllowed(prefix: string, bankId: string): boolean {
  if (bankIdHasTraversal(bankId)) return false;
  if (prefix === "") return true;
  return bankId === prefix || bankId.startsWith(`${prefix}--`);
}

/**
 * True when a bank id contains a path separator (`/` or `\`) or a `..` path
 * segment — the shapes that let an id escape its bank path during URL
 * normalization. Percent-encoded forms are covered too, since a bank id is
 * decoded before it reaches the predicate (middleware decodes path ids, query
 * and body ids arrive already decoded).
 */
export function bankIdHasTraversal(bankId: string): boolean {
  if (bankId.includes("/") || bankId.includes("\\")) return true;
  // A ".." segment, bounded by start/end or a separator on either side.
  return /(^|[/\\])\.\.([/\\]|$)/.test(bankId);
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
