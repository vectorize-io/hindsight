import React, {type ReactNode} from 'react';
import Sidebar from '@theme-original/DocRoot/Layout/Sidebar';
import type SidebarType from '@theme/DocRoot/Layout/Sidebar';
import type {WrapperProps} from '@docusaurus/types';
import integrationsData from '@site/src/data/integrations.json';

type Props = WrapperProps<typeof SidebarType>;

// Single source of truth: src/data/integrations.json drives the "Integrations"
// sidebar category for every docs version. The sidebar files only carry a
// positional placeholder category; we fill its items here at render time, so
// adding an entry to integrations.json (with an internal /sdks/integrations/
// link) is all it takes — no per-version sidebar edits. Array order in the JSON
// is the manual display order. External entries (http links) are gallery-only.
const integrationItems = integrationsData.integrations
  .filter((entry) => entry.link.startsWith('/sdks/integrations/'))
  .map((entry) => ({
    type: 'link' as const,
    href: entry.link,
    label: entry.name,
    customProps: {icon: entry.icon},
  }));

function withIntegrations(sidebar: Props['sidebar']): Props['sidebar'] {
  if (!sidebar) {
    return sidebar;
  }
  return sidebar.map((item) =>
    item.type === 'category' && item.label === 'Integrations'
      ? {...item, items: integrationItems}
      : item,
  );
}

export default function SidebarWrapper(props: Props): ReactNode {
  return <Sidebar {...props} sidebar={withIntegrations(props.sidebar)} />;
}
