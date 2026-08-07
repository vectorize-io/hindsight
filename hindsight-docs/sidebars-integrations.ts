import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import * as fs from 'fs';
import * as path from 'path';
import {groupIntegrations, splitPreview} from './src/lib/integration-groups';

interface IntegrationEntry {
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
  integrationsSidebar: groupIntegrations(internal).map((group) => {
    const {preview, overflow} = splitPreview(group.entries);
    return {
      type: 'category' as const,
      label: group.label,
      // Open by default so every group's contents are visible at a glance; the tail sits behind a
      // nested "Show all", which keeps 59 entries from becoming a wall you scroll past.
      collapsible: true,
      collapsed: false,
      items: [
        ...preview.map(docItem),
        ...(overflow.length
          ? [
              {
                type: 'category' as const,
                label: `Show all ${group.entries.length}`,
                collapsible: true,
                collapsed: true,
                items: overflow.map(docItem),
              },
            ]
          : []),
      ],
    };
  }),
};

export default sidebars;
