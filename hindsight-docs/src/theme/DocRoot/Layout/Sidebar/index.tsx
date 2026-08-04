import React, {type ReactNode} from 'react';
import Sidebar from '@theme-original/DocRoot/Layout/Sidebar';
import type SidebarType from '@theme/DocRoot/Layout/Sidebar';
import type {WrapperProps} from '@docusaurus/types';
import {internalIntegrationsSorted} from '@site/src/lib/integrations';
import {groupIntegrations} from '@site/src/lib/integration-groups';

type Props = WrapperProps<typeof SidebarType>;

// Single source of truth: src/data/integrations.json drives the "Integrations"
// sidebar category on the versioned main docs. Those sidebar files (current +
// frozen versioned_sidebars/*) carry only a placeholder category with one link
// to the gallery; we replace that placeholder at render time with the full,
// alphabetically-sorted list, so adding a JSON entry is all it takes — no
// per-version sidebar edits. The unversioned integration pages have their own
// generated sidebar (sidebars-integrations.ts), which is left untouched here.
//
// Grouped into coding agents / frameworks / apps rather than one flat run of 59
// links. Collapsed by default here (unlike the standalone integrations sidebar):
// this category is already nested inside the main docs tree, so expanding all
// three groups would bury the rest of the navigation.
const integrationItems = groupIntegrations(internalIntegrationsSorted).map((group) => ({
  type: 'category' as const,
  label: group.label,
  collapsible: true,
  collapsed: true,
  items: group.entries.map((entry) => ({
    type: 'link' as const,
    href: entry.link,
    label: entry.name,
    customProps: {icon: entry.icon},
  })),
}));

function isIntegrationsPlaceholder(item: NonNullable<Props['sidebar']>[number]): boolean {
  return (
    item.type === 'category' &&
    item.label === 'Integrations' &&
    item.items.length === 1 &&
    item.items[0]?.type === 'link' &&
    item.items[0].href === '/integrations'
  );
}

function withIntegrations(sidebar: Props['sidebar']): Props['sidebar'] {
  if (!sidebar) {
    return sidebar;
  }
  return sidebar.map((item) =>
    isIntegrationsPlaceholder(item) ? {...item, items: integrationItems} : item,
  );
}

export default function SidebarWrapper(props: Props): ReactNode {
  return <Sidebar {...props} sidebar={withIntegrations(props.sidebar)} />;
}
