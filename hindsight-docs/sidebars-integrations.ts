import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import * as fs from 'fs';
import * as path from 'path';

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
// at build if a listed page is missing.
//
// Flat and alphabetical, deliberately: once you are ON an integration page you
// are comparing and hopping between them, so the useful view is every option at
// once. The curated, grouped preview lives in the MAIN docs sidebar, where the
// job is different — introducing the category to someone who is elsewhere.
const {integrations} = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'src', 'data', 'integrations.json'), 'utf-8'),
) as {integrations: IntegrationEntry[]};

const internal = integrations
  .filter((entry) => entry.link.startsWith('/sdks/integrations/'))
  .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));

const sidebars: SidebarsConfig = {
  integrationsSidebar: [
    {
      type: 'category',
      label: 'Integrations',
      collapsible: false,
      items: [
        ...internal.map((entry) => ({
          type: 'doc' as const,
          id: entry.link.replace('/sdks/integrations/', ''),
          label: entry.name,
          customProps: {icon: entry.icon},
        })),
        // The gallery adds search and filters a sidebar cannot.
        {
          type: 'link' as const,
          href: '/integrations',
          label: 'All integrations',
          customProps: {iconAfter: 'lu-arrow-up-right'},
        },
      ],
    },
  ],
};

export default sidebars;
