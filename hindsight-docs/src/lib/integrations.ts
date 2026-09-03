import integrationsData from '@site/src/data/integrations.json';

export interface IntegrationEntry {
  id: string;
  name: string;
  description: string;
  type: string;
  by: string;
  category: string;
  link: string;
  icon: string;
}

const byName = (a: IntegrationEntry, b: IntegrationEntry): number =>
  a.name.toLowerCase().localeCompare(b.name.toLowerCase());

// src/data/integrations.json is the single source of truth. Display order is
// alphabetical by name, shared by the Integrations Hub gallery and the docs
// sidebars (the JSON array order is no longer significant for display).
export const integrationsSorted: IntegrationEntry[] = [
  ...(integrationsData.integrations as IntegrationEntry[]),
].sort(byName);

// Only entries with an on-site doc page (external/http entries are gallery-only).
export const internalIntegrationsSorted: IntegrationEntry[] = integrationsSorted.filter((entry) =>
  entry.link.startsWith('/sdks/integrations/'),
);

// Display labels and running order for `category`, shared by the Integrations Hub
// gallery and the changelog listing so the two read as the same taxonomy. A
// category missing here falls to the end under its raw value rather than
// disappearing from the page.
export const CATEGORY_LABELS: Record<string, string> = {
  'coding-agent': 'Coding agents',
  framework: 'Frameworks & SDKs',
  mcp: 'MCP',
  tool: 'Tools & apps',
  legacy: 'Superseded',
};

const CATEGORY_ORDER = ['coding-agent', 'framework', 'mcp', 'tool', 'legacy'];

function categoryRank(category: string): number {
  const i = CATEGORY_ORDER.indexOf(category);
  return i === -1 ? CATEGORY_ORDER.length : i;
}

/** Group entries by category, in display order, dropping empty groups. */
export function groupByCategory<T extends {category: string}>(entries: T[]): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const entry of entries) {
    const bucket = groups.get(entry.category);
    if (bucket) {
      bucket.push(entry);
    } else {
      groups.set(entry.category, [entry]);
    }
  }
  return [...groups.entries()].sort(
    ([a], [b]) => categoryRank(a) - categoryRank(b) || a.localeCompare(b),
  );
}
