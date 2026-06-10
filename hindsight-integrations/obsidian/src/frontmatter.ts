/**
 * Lightweight note normalization: split YAML frontmatter from the body and lift
 * a few well-known keys (tags, aliases, created/date) into Hindsight
 * tags/metadata. Intentionally dependency-free and conservative — we do not try
 * to fully parse YAML, only the simple key/value and list forms Obsidian writes.
 */

export interface NormalizedNote {
  /** The note body, frontmatter stripped. */
  body: string;
  /** Tags lifted from frontmatter `tags`/`aliases`. */
  tags: string[];
  /** ISO 8601 timestamp lifted from `created`/`date`, if present. */
  timestamp?: string;
  /** String metadata (folder, plus any lifted scalars). */
  metadata: Record<string, string>;
}

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

/** Parse a frontmatter scalar/list value into a list of strings. */
function parseList(value: string): string[] {
  const trimmed = value.trim();
  if (!trimmed) return [];
  // Inline flow list: [a, b, c]
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .map((s) => s.trim().replace(/^["']|["']$/g, ""))
      .filter(Boolean);
  }
  // Comma- or whitespace-separated scalar
  return trimmed
    .split(/[,\s]+/)
    .map((s) => s.replace(/^["']|["']$/g, ""))
    .filter(Boolean);
}

function stripQuotes(value: string): string {
  return value.trim().replace(/^["']|["']$/g, "");
}

export function normalizeNote(raw: string, folder: string): NormalizedNote {
  const tags: string[] = [];
  const metadata: Record<string, string> = {};
  if (folder) metadata.folder = folder;
  let timestamp: string | undefined;
  let body = raw;

  const match = raw.match(FRONTMATTER_RE);
  if (match) {
    body = raw.slice(match[0].length);
    const lines = match[1].split(/\r?\n/);
    let currentKey: string | null = null;

    for (const line of lines) {
      // Block list item:  "  - value"
      const listItem = line.match(/^\s*-\s+(.*)$/);
      if (listItem && currentKey) {
        const v = stripQuotes(listItem[1]);
        if (currentKey === "tags" || currentKey === "aliases") {
          if (v) tags.push(v);
        }
        continue;
      }

      const kv = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
      if (!kv) continue;
      const key = kv[1].toLowerCase();
      const value = kv[2];
      currentKey = key;

      if (key === "tags" || key === "aliases") {
        tags.push(...parseList(value));
      } else if ((key === "created" || key === "date") && !timestamp) {
        const v = stripQuotes(value);
        if (v) timestamp = v;
      } else {
        const v = stripQuotes(value);
        if (v) metadata[key] = v;
      }
    }
  }

  // Dedupe tags while preserving order.
  const uniqueTags = [...new Set(tags)];
  return { body: body.trim(), tags: uniqueTags, timestamp, metadata };
}
