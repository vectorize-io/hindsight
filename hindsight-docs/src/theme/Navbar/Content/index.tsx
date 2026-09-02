import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {
  useThemeConfig,
  ErrorCauseBoundary,
  ThemeClassNames,
} from '@docusaurus/theme-common';
import {splitNavbarItems} from '@docusaurus/theme-common/internal';
import NavbarItem, {type Props as NavbarItemConfig} from '@theme/NavbarItem';
import NavbarColorModeToggle from '@theme/Navbar/ColorModeToggle';
import SearchBar from '@theme/SearchBar';
import NavbarMobileSidebarToggle from '@theme/Navbar/MobileSidebar/Toggle';
import NavbarLogo from '@theme/Navbar/Logo';
import NavbarSearch from '@theme/Navbar/Search';

import styles from './styles.module.css';

function useNavbarItems() {
  // TODO temporary casting until ThemeConfig type is improved
  return useThemeConfig().navbar.items as NavbarItemConfig[];
}

// The sign-up button is the one call to action in the navbar, so it renders
// last on the right — after the icon buttons and past the divider — instead of
// in the middle of them where its config position would otherwise put it.
function isCallToAction(item: NavbarItemConfig): boolean {
  return hasClassName(item, 'navbar-item-signup');
}

// "Docs" and the version dropdown read as one control — the version qualifies
// the docs section rather than pointing somewhere of its own. They share a
// wrapper so custom.css can draw a single surface behind both; splitting the
// background across two siblings would seam down the middle.
const DOCS_GROUP_CLASSNAMES = ['navbar-item-developer', 'navbar-item-version'];

function isDocsGroupItem(item: NavbarItemConfig): boolean {
  return DOCS_GROUP_CLASSNAMES.some((name) => hasClassName(item, name));
}

function hasClassName(item: NavbarItemConfig, name: string): boolean {
  return typeof item.className === 'string' && item.className.includes(name);
}

function NavbarItems({items}: {items: NavbarItemConfig[]}): ReactNode {
  return (
    <>
      {items.map((item, i) => (
        <ErrorCauseBoundary
          key={i}
          onError={(error) =>
            new Error(
              `A theme navbar item failed to render.
Please double-check the following navbar item (themeConfig.navbar.items) of your Docusaurus config:
${JSON.stringify(item, null, 2)}`,
              {cause: error},
            )
          }>
          <NavbarItem {...item} />
        </ErrorCauseBoundary>
      ))}
    </>
  );
}

function NavbarContentLayout({
  left,
  right,
}: {
  left: ReactNode;
  right: ReactNode;
}) {
  return (
    <div className="navbar__inner">
      <div
        className={clsx(
          ThemeClassNames.layout.navbar.containerLeft,
          'navbar__items',
        )}>
        {left}
      </div>
      <div
        className={clsx(
          ThemeClassNames.layout.navbar.containerRight,
          'navbar__items navbar__items--right',
        )}>
        {right}
      </div>
    </div>
  );
}

export default function NavbarContent(): ReactNode {
  const items = useNavbarItems();
  const [leftItems, rightItems] = splitNavbarItems(items);
  const ctaItems = rightItems.filter(isCallToAction);
  const secondaryRightItems = rightItems.filter((item) => !isCallToAction(item));
  const docsGroupItems = leftItems.filter(isDocsGroupItem);
  const otherLeftItems = leftItems.filter((item) => !isDocsGroupItem(item));

  const searchBarItem = items.find((item) => item.type === 'search');

  return (
    <NavbarContentLayout
      left={
        // TODO stop hardcoding items?
        // Always render toggle, CSS controls visibility at 1400px breakpoint
        <>
          {/* Brand block: spans the sidebar's column, so the divider closing
              it continues the sidebar's own edge (see custom.css). */}
          <div className="navbar__brand-block">
            <NavbarMobileSidebarToggle />
            <NavbarLogo />
          </div>
          {docsGroupItems.length > 0 && (
            <div className="navbar__docs-group">
              <NavbarItems items={docsGroupItems} />
            </div>
          )}
          <NavbarItems items={otherLeftItems} />
        </>
      }
      right={
        // TODO stop hardcoding items?
        // Ask the user to add the respective navbar items => more flexible
        <>
          <NavbarItems items={secondaryRightItems} />
          <NavbarColorModeToggle className={styles.colorModeToggle} />
          {!searchBarItem && (
            <NavbarSearch>
              <SearchBar />
            </NavbarSearch>
          )}
          <NavbarItems items={ctaItems} />
        </>
      }
    />
  );
}
