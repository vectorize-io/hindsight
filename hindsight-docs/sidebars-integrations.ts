import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import * as fs from 'fs';
import * as path from 'path';

interface IntegrationEntry {
  name: string;
  link: string;
  icon: string;
}

// Sidebar for the (unversioned) integration doc pages. Generated directly from
// src/data/integrations.json — the single source of truth — so adding an entry
// is all it takes. These pages aren't versioned, so unlike the main docs we can
// build the sidebar here at config-load time (no swizzle needed). Using `doc`
// items both associates each page with this sidebar (so it renders) and throws
// at build if a listed page is missing. Alphabetical by name, matching the
// Integrations Hub and the main docs sidebar.
const {integrations} = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'src', 'data', 'integrations.json'), 'utf-8'),
) as {integrations: IntegrationEntry[]};

const items = integrations
  .filter((entry) => entry.link.startsWith('/sdks/integrations/'))
  .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()))
  .map((entry) => ({
    type: 'doc' as const,
    id: entry.link.replace('/sdks/integrations/', ''),
    label: entry.name,
    customProps: {icon: entry.icon},
  }));

const sidebars: SidebarsConfig = {
  integrationsSidebar: [{type: 'category', label: 'Integrations', collapsible: false, items}],
};

export default sidebars;
