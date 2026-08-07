/**
 * Display grouping for the integration sidebars.
 *
 * `category` in src/data/integrations.json is the taxonomy field; this maps those values onto the
 * few groups users actually navigate by. Previously every integration sat in one flat list — at 59
 * entries that was a wall of names with no way to tell a coding agent from an SDK.
 *
 * Deliberately free of the `@site/` alias and of any JSON import so it can be pulled in both from
 * the theme (webpack, where the alias exists) and from sidebars-integrations.ts (evaluated at
 * config load, where it does not).
 */
export interface IntegrationGroup {
  label: string;
  categories: string[];
}

// Order here is display order in both sidebars. Coding agents lead: they're the most common entry
// point into the docs.
export const INTEGRATION_GROUPS: IntegrationGroup[] = [
  {label: 'Coding agents', categories: ['coding-agent']},
  {label: 'Frameworks & SDKs', categories: ['framework']},
  // Chat apps, note-taking, voice platforms, MCP gateways — and the catch-all: an entry whose
  // category isn't listed above lands here rather than silently vanishing from the sidebar.
  {label: 'Apps & tools', categories: ['tool', 'mcp']},
];

export interface GroupedIntegrations<T> {
  label: string;
  entries: T[];
}

/**
 * How many entries a group shows before the rest move behind a "Show all" sub-category.
 *
 * All 59 expanded is a wall nobody reads; all collapsed hides the fact that the list is worth
 * opening. Showing a handful per group keeps every group's existence visible while the sidebar
 * stays scannable, and the overflow is one click away.
 */
export const GROUP_PREVIEW_COUNT = 6;

/** Split a group's entries into the ones shown inline and the ones behind "Show all". */
export function splitPreview<T>(entries: readonly T[]): {preview: T[]; overflow: T[]} {
  // One past the limit would put a single entry behind a click — never worth it.
  if (entries.length <= GROUP_PREVIEW_COUNT + 1) return {preview: [...entries], overflow: []};
  return {
    preview: entries.slice(0, GROUP_PREVIEW_COUNT),
    overflow: entries.slice(GROUP_PREVIEW_COUNT),
  };
}

/** Bucket entries into INTEGRATION_GROUPS order, preserving the order they arrive in. */
export function groupIntegrations<T extends {category: string}>(
  entries: readonly T[],
): GroupedIntegrations<T>[] {
  const fallback = INTEGRATION_GROUPS.length - 1;
  const buckets: GroupedIntegrations<T>[] = INTEGRATION_GROUPS.map((group) => ({
    label: group.label,
    entries: [],
  }));
  for (const entry of entries) {
    const index = INTEGRATION_GROUPS.findIndex((group) => group.categories.includes(entry.category));
    buckets[index === -1 ? fallback : index].entries.push(entry);
  }
  // The umbrella package leads the coding-agent group rather than sorting under "C": it is the
  // entry point for every other agent in that list, so it must be one of the entries on show.
  for (const bucket of buckets) {
    const i = bucket.entries.findIndex((e) => (e as {id?: string}).id === 'coding-agents');
    if (i > 0) bucket.entries.unshift(...bucket.entries.splice(i, 1));
  }
  return buckets.filter((bucket) => bucket.entries.length > 0);
}
