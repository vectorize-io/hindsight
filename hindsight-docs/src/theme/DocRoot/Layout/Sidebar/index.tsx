import React, {type ReactNode} from 'react';
import Sidebar from '@theme-original/DocRoot/Layout/Sidebar';
import type SidebarType from '@theme/DocRoot/Layout/Sidebar';
import type {WrapperProps} from '@docusaurus/types';
import {internalIntegrationsSorted} from '@site/src/lib/integrations';
import {groupIntegrations} from '@site/src/lib/integration-groups';
import featured from '@site/src/data/featured-posts.json';

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
const linkItem = (entry: {link: string; name: string; icon: string}) => ({
  type: 'link' as const,
  href: entry.link,
  label: entry.name,
  customProps: {icon: entry.icon},
});

// Groups render INLINE and non-collapsible: this is a preview, not an index. Nothing hides behind a
// disclosure, so the shape of the integration surface is readable at a glance, and anyone wanting
// the complete list follows the gallery link at the end. The full alphabetical list lives in the
// integration pages' own sidebar (sidebars-integrations.ts), where you are already browsing them.
const integrationItems = [
  ...groupIntegrations(internalIntegrationsSorted).map((group) => ({
    type: 'category' as const,
    label: group.label,
    collapsible: false,
    collapsed: false,
    items: [
      // Harness logos (coding agents only) all point at the umbrella page.
      ...group.harnessLinks.map((harness) => ({
        type: 'link' as const,
        href: harness.href,
        label: harness.label,
        customProps: {icon: harness.icon},
      })),
      ...group.preview.map(linkItem),
    ],
  })),
  {
    type: 'link' as const,
    href: '/integrations',
    label: 'All integrations',
    customProps: {iconAfter: 'lu-arrow-up-right'},
  },
];

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
  // The placeholder is REPLACED BY its contents rather than filled: an "Integrations" wrapper around
  // three group headings meant two levels of nesting to say one thing, and pushed every entry a
  // further indent in. The groups take its place, so they sit at the same level as the rest of the
  // navigation and in the same position the wrapper occupied.
  return sidebar.flatMap((item) => (isIntegrationsPlaceholder(item) ? integrationItems : [item]));
}

// "/blog/2026/10/01/slug" -> "Oct 1, 2026"; undefined if the link carries no date.
function postDate(href: string): string | undefined {
  const m = href.match(/\/blog\/(\d{4})\/(\d{2})\/(\d{2})\//);
  if (!m) {
    return undefined;
  }
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

// Pinned blog posts (src/data/featured-posts.json) go first, on every sidebar and version, so they
// stay in view. The blog index shows the same pins; scripts/check-featured-posts.mjs guards the file.
const featuredCategory = {
  type: 'category' as const,
  label: 'Blog',
  collapsible: false,
  collapsed: false,
  items: [
    ...featured.posts.map((post) => ({
      type: 'link' as const,
      href: post.href,
      label: post.label,
      customProps: {cover: post.cover, date: postDate(post.href)},
    })),
    {
      type: 'link' as const,
      href: '/blog',
      label: 'All posts',
      customProps: {iconAfter: 'lu-arrow-up-right'},
    },
  ],
};

function withFeatured(sidebar: Props['sidebar']): Props['sidebar'] {
  if (!sidebar || featured.posts.length === 0) {
    return sidebar;
  }
  return [featuredCategory, ...sidebar];
}

export default function SidebarWrapper(props: Props): ReactNode {
  return <Sidebar {...props} sidebar={withFeatured(withIntegrations(props.sidebar))} />;
}
