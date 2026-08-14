/**
 * Display grouping for the integration sidebars.
 *
 * `category` in src/data/integrations.json is the taxonomy field; this maps those values onto the
 * few groups users actually navigate by. Previously every integration sat in one flat list — at 59
 * entries that was a wall of names with no way to tell a coding agent from an SDK.
 *
 * Each group shows a short CURATED preview and hides the tail behind "Show all". The preview is
 * hand-picked rather than alphabetical: the first names in a sorted list are an accident of
 * spelling, and the point of a preview is to show what the group is for.
 *
 * Deliberately free of the `@site/` alias and of any JSON import so it can be pulled in both from
 * the theme (webpack, where the alias exists) and from sidebars-integrations.ts (evaluated at
 * config load, where it does not).
 */
export interface IntegrationGroup {
  label: string;
  categories: string[];
  /** Entries shown before "Show all", by id, in this order. */
  previewIds?: string[];
  /** Preview the harness logos instead of any entry — the coding-agent group only. */
  harnessPreview?: boolean;
  /** Superseded pages: kept and reachable, but out of the gallery and last in the sidebar. */
  legacy?: boolean;
}

/** The page every harness logo in the coding-agent preview points at. */
export const CODING_AGENTS_LINK = '/sdks/integrations/coding-agents';

/**
 * The coding-agent group previews HARNESSES, not integration pages.
 *
 * All of these are covered by the single Coding Agents plugin, so listing ten doc pages would
 * present it as ten separate integrations; every logo links to the one page instead. The logos also
 * make the group recognisable at a glance in a way a column of names is not. Files live in
 * static/img/harness/, named by the harness id the plugin itself uses.
 */
export const CODING_AGENT_HARNESSES: {label: string; icon: string}[] = [
  {label: 'Claude Code', icon: '/img/harness/claude-code.png'},
  {label: 'Codex CLI', icon: '/img/harness/codex.svg'},
  {label: 'opencode', icon: '/img/harness/opencode.png'},
  {label: 'Kilo CLI', icon: '/img/harness/kilo.svg'},
  {label: 'Cursor CLI', icon: '/img/harness/cursor-cli.svg'},
  {label: 'GitHub Copilot CLI', icon: '/img/harness/copilot-cli.svg'},
  {label: 'Grok Build', icon: '/img/harness/grok-build.svg'},
  {label: 'Antigravity CLI', icon: '/img/harness/antigravity-cli.png'},
  {label: 'Devin CLI', icon: '/img/harness/devin-cli.svg'},
  {label: 'Cline CLI', icon: '/img/harness/cline-cli.svg'},
  {label: 'DeepSeek Harness', icon: '/img/harness/dsh.svg'},
];

// Order here is display order in both sidebars. Coding agents lead: they're the most common entry
// point into the docs.
export const INTEGRATION_GROUPS: IntegrationGroup[] = [
  {label: 'Coding agents', categories: ['coding-agent'], harnessPreview: true},
  {
    label: 'Frameworks & SDKs',
    categories: ['framework'],
    previewIds: ['langgraph', 'vercel-ai-sdk', 'vercel-chat', 'eve', 'crewai'],
  },
  // Chat apps, note-taking, voice platforms, MCP gateways — and the catch-all: an entry whose
  // category isn't listed above lands here rather than silently vanishing from the sidebar.
  {
    label: 'Apps & tools',
    categories: ['tool', 'mcp'],
    previewIds: ['chatgpt', 'hermes', 'openclaw', 'obsidian'],
  },
  // Last, and every entry shown: these pages are superseded by the Coding Agents plugin and each
  // carries a banner saying so. They stay reachable — people still run these plugins and arrive
  // from old links — but they are out of the gallery and out of the way.
  {label: 'Legacy', categories: ['legacy'], legacy: true},
];

export interface HarnessLink {
  label: string;
  icon: string;
  href: string;
}

export interface GroupSidebar<T> {
  label: string;
  /** Fixed logo links shown instead of a preview — coding agents only. */
  harnessLinks: HarnessLink[];
  /** Curated entries shown inline. */
  preview: T[];
  /** Everything else, behind "Show all". */
  overflow: T[];
  /** Total entries in the group, for the "Show all N" label. */
  total: number;
}

/**
 * Bucket entries into INTEGRATION_GROUPS order and split each into preview + overflow.
 *
 * A `previewIds` entry that matches nothing is skipped rather than throwing: that list is
 * hand-edited, and a typo should cost one preview slot, not the whole sidebar.
 */
export function groupIntegrations<T extends {id: string; category: string}>(
  entries: readonly T[],
): GroupSidebar<T>[] {
  const fallback = INTEGRATION_GROUPS.length - 1;
  const buckets: T[][] = INTEGRATION_GROUPS.map(() => []);
  for (const entry of entries) {
    const index = INTEGRATION_GROUPS.findIndex((group) => group.categories.includes(entry.category));
    buckets[index === -1 ? fallback : index].push(entry);
  }

  return INTEGRATION_GROUPS.map((group, i) => {
    const all = buckets[i];
    const previewsHarnesses = group.harnessPreview === true;
    // The coding-agent group keeps every page behind "Show all": its inline slots are harness
    // logos pointing at the umbrella page, not at any individual integration.
    const preview = previewsHarnesses
      ? []
      : group.legacy
        ? all
        : (group.previewIds ?? [])
          .map((id) => all.find((entry) => entry.id === id))
          .filter((entry): entry is T => Boolean(entry));
    const shown = new Set(preview.map((entry) => entry.id));
    return {
      label: group.label,
      harnessLinks: previewsHarnesses
        ? CODING_AGENT_HARNESSES.map((harness) => ({...harness, href: CODING_AGENTS_LINK}))
        : [],
      preview,
      overflow: all.filter((entry) => !shown.has(entry.id)),
      total: all.length,
    };
  }).filter((group) => group.total > 0);
}
