import React, {useEffect, useRef} from 'react';
import Content from '@theme-original/DocSidebar/Desktop/Content';
import type ContentType from '@theme/DocSidebar/Desktop/Content';
import type {WrapperProps} from '@docusaurus/types';
import SearchBar from '@theme/SearchBar';

import styles from './styles.module.css';

type Props = WrapperProps<typeof ContentType>;

/**
 * The search plugin hardcodes its placeholder to the "Search" translation and
 * exposes no option for it. Overriding the string through i18n/en/code.json is
 * not available to us either: writing any translation file turns on the
 * translation pipeline for the docs plugin, which then rejects the sidebars
 * over duplicate entry labels across versions.
 *
 * React never rewrites the attribute after mount (the prop it renders is a
 * constant, so it never diffs), which leaves setting it here as the one place
 * that holds.
 */
function useSearchPlaceholder(text: string) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const input = containerRef.current?.querySelector('input');
    if (input) {
      input.placeholder = text;
    }
  }, [text]);

  return containerRef;
}

/**
 * Puts search at the top of the docs sidebar instead of in the navbar: it
 * belongs with the navigation it searches, and it buys back the navbar space
 * the search field used to occupy.
 *
 * It stays mounted in the navbar for pages with no sidebar (blog, galleries) —
 * custom.css hides that copy wherever this one renders, so exactly one search
 * field is on screen at a time.
 */
export default function ContentWrapper(props: Props): JSX.Element {
  const searchRef = useSearchPlaceholder('Search docs');

  return (
    <>
      <div className={styles.sidebarSearch} ref={searchRef}>
        <SearchBar />
      </div>
      <Content {...props} />
    </>
  );
}
