/**
 * A tiny YAML frontmatter serializer/parser for flat metadata maps.
 *
 * We only ever emit scalars (string | number | boolean | null) and arrays of
 * scalars, so a full YAML dependency is overkill. Double-quoted scalars use
 * JSON string syntax, which is a valid subset of YAML, giving us correct
 * escaping of colons, quotes, and newlines for free.
 */

export type FrontmatterValue = string | number | boolean | null | string[];
export type Frontmatter = Record<string, FrontmatterValue>;

/** True when a string is safe to emit unquoted as a YAML plain scalar. */
function isPlainSafe(s: string): boolean {
  if (s.length === 0) return false;
  if (s !== s.trim()) return false;
  // Avoid anything that could be interpreted as structure or a non-string type.
  if (/[:#\[\]{}&*!|>'"%@`,]/.test(s)) return false;
  if (/^[-?]/.test(s)) return false;
  if (/^(true|false|null|yes|no|on|off|~)$/i.test(s)) return false;
  if (/^[-+]?[0-9]/.test(s)) return false; // could parse as number/date
  if (/[\n\r\t]/.test(s)) return false;
  return true;
}

function emitScalar(value: string | number | boolean | null): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
  return isPlainSafe(value) ? value : JSON.stringify(value);
}

/** Serialize a frontmatter map to a `---`-delimited YAML block (no trailing newline). */
export function stringifyFrontmatter(data: Frontmatter): string {
  const lines: string[] = ["---"];
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined) continue;
    if (Array.isArray(value)) {
      if (value.length === 0) {
        lines.push(`${key}: []`);
      } else {
        lines.push(`${key}: [${value.map((v) => emitScalar(v)).join(", ")}]`);
      }
    } else {
      lines.push(`${key}: ${emitScalar(value)}`);
    }
  }
  lines.push("---");
  return lines.join("\n");
}

export interface ParsedDocument {
  frontmatter: Frontmatter;
  body: string;
}

function parseScalar(raw: string): FrontmatterValue {
  const t = raw.trim();
  if (t === "null" || t === "~" || t === "") return null;
  if (t === "true") return true;
  if (t === "false") return false;
  if (/^".*"$/.test(t)) {
    try {
      return JSON.parse(t) as string;
    } catch {
      return t.slice(1, -1);
    }
  }
  if (/^[-+]?\d+(\.\d+)?$/.test(t)) return Number(t);
  return t;
}

/**
 * Parse a document that may begin with a frontmatter block. Lossy by design —
 * sufficient for round-tripping what `stringifyFrontmatter` writes and for tests.
 */
export function parseDocument(text: string): ParsedDocument {
  if (!text.startsWith("---\n") && !text.startsWith("---\r\n")) {
    return { frontmatter: {}, body: text };
  }
  const normalized = text.replace(/\r\n/g, "\n");
  const end = normalized.indexOf("\n---", 3);
  if (end === -1) return { frontmatter: {}, body: text };

  const block = normalized.slice(4, end);
  let rest = normalized.slice(end + 4);
  if (rest.startsWith("\n")) rest = rest.slice(1);

  const frontmatter: Frontmatter = {};
  for (const line of block.split("\n")) {
    if (!line.trim()) continue;
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    const valueRaw = line.slice(idx + 1).trim();
    if (/^\[.*\]$/.test(valueRaw)) {
      const inner = valueRaw.slice(1, -1).trim();
      frontmatter[key] = inner ? splitFlowItems(inner).map((v) => String(parseScalar(v))) : [];
    } else {
      frontmatter[key] = parseScalar(valueRaw);
    }
  }
  return { frontmatter, body: rest };
}

/** Split `a, "b, c", d` respecting JSON-quoted items. */
function splitFlowItems(inner: string): string[] {
  const items: string[] = [];
  let current = "";
  let inQuote = false;
  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i];
    if (ch === '"' && inner[i - 1] !== "\\") inQuote = !inQuote;
    if (ch === "," && !inQuote) {
      items.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  if (current.trim()) items.push(current.trim());
  return items;
}
