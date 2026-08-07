import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import * as fs from 'fs';
import * as path from 'path';
import {groupIntegrations} from './src/lib/integration-groups';

interface IntegrationEntry {
  id: string;
  name: string;
  link: string;
  icon: string;
  category: string;
}

// Sidebar for the (unversioned) integration doc pages. Generated directly from
// src/data/integrations.json — the single source of truth — so adding an entry
// is all it takes. These pages aren't versioned, so unlike the main docs we can
// build the sidebar here at config-load time (no swizzle needed). Using `doc`
// items both associates each page with this sidebar (so it renders) and throws
// at build if a listed page is missing. Grouped by category (see
// src/lib/integration-groups.ts), alphabetical by name within each group,
// matching the Integrations Hub and the main docs sidebar.
const {integrations} = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'src', 'data', 'integrations.json'), 'utf-8'),
) as {integrations: IntegrationEntry[]};

const internal = integrations
  .filter((entry) => entry.link.startsWith('/sdks/integrations/'))
  .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));

const docItem = (entry: IntegrationEntry) => ({
  type: 'doc' as const,
  id: entry.link.replace('/sdks/integrations/', ''),
  label: entry.name,
  customProps: {icon: entry.icon},
});

const sidebars: SidebarsConfig = {
  integrationsSidebar: groupIntegrations(internal).map((group) => ({
    type: 'category' as const,
    label: group.label,
    // Open by default so each group's character is visible at a glance; the tail sits behind a
    // nested "Show all", which keeps 59 entries from becoming a wall you scroll past.
    collapsible: true,
    collapsed: false,
    items: [
      // Harness logos (coding agents only) all point at the umbrella page.
      ...group.harnessLinks.map((harness) => ({
        type: 'link' as const,
        href: harness.href,
        label: harness.label,
        customProps: {icon: harness.icon},
      })),
      ...group.preview.map(docItem),
      ...(group.overflow.length
        ? [
            {
              type: 'category' as const,
              // Accurate either way: the coding-agent group previews harnesses rather than pages, so
              // its overflow really is every entry; elsewhere some are already shown above.
              label: group.preview.length
                ? `Show ${group.overflow.length} more`
                : `Show all ${group.total}`,
              collapsible: true,
              collapsed: true,
              // Every page lives here, which is also what associates it with this sidebar.
              items: group.overflow.map(docItem),
            },
          ]
        : []),
    ],
  })),
};

export default sidebars;
