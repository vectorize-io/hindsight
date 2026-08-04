import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import * as fs from 'fs';
import * as path from 'path';
import {groupIntegrations} from './src/lib/integration-groups';

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

const sidebars: SidebarsConfig = {
  integrationsSidebar: groupIntegrations(internal).map((group) => ({
    type: 'category' as const,
    label: group.label,
    // Collapsible — 59 entries in three groups is a lot to scroll past — but
    // expanded by default so nothing is hidden from someone browsing.
    collapsible: true,
    collapsed: false,
    items: group.entries.map((entry) => ({
      type: 'doc' as const,
      id: entry.link.replace('/sdks/integrations/', ''),
      label: entry.name,
      customProps: {icon: entry.icon},
    })),
  })),
};

export default sidebars;
