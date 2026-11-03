import React, {type ReactNode} from 'react';
import styles from './ProviderSetup.module.css';

/**
 * A provider's setup on the Models page, folded under a small "Setup" toggle so the page reads
 * as a list of providers. The heading and the one-line description above it stay visible, so
 * anchors and the table of contents still land on the provider.
 */
// The section the reader just jumped to: its heading, its paragraph and its setup, opened, with a
// bar down their left so it's clear where to read next. The bar lives on <body>, outside React's
// tree, and follows the section when a setup opens or the window resizes.
let focused: Element[] = [];
let bar: HTMLDivElement | null = null;

function placeBar() {
  if (!bar || !focused.length) return;
  const first = focused[0].getBoundingClientRect();
  const last = focused[focused.length - 1].getBoundingClientRect();
  bar.style.top = `${first.top + window.scrollY - 6}px`;
  bar.style.height = `${last.bottom - first.top + 12}px`;
  bar.style.left = `${first.left + window.scrollX - 12}px`;
}

/** Opens the setup under the heading with this id and marks its section. */
export function revealProvider(id: string) {
  const heading = document.getElementById(id);
  if (!heading || !/^H[23]$/.test(heading.tagName)) return;
  const section: Element[] = [heading];
  for (let el = heading.nextElementSibling; el && !/^(H2|H3|HR)$/.test(el.tagName); el = el.nextElementSibling) {
    section.push(el);
  }
  const setup = section.find((el) => el.tagName === 'DETAILS') as HTMLDetailsElement | undefined;
  if (!setup) return;
  setup.open = true;
  focused = section;
  if (!bar) {
    bar = document.createElement('div');
    bar.className = styles.focusBar;
    document.body.appendChild(bar);
    window.addEventListener('resize', placeBar);
    document.addEventListener('toggle', placeBar, true);
  }
  bar.classList.remove(styles.focusIn);
  void bar.offsetWidth; // restart the fade-in
  bar.classList.add(styles.focusIn);
  placeBar();
}

export function ProviderSetup({children, label = 'Setup'}: {children: ReactNode; label?: string}) {
  return (
    <details className={styles.setup}>
      <summary className={styles.summary}>{label}</summary>
      <div className={styles.body}>{children}</div>
    </details>
  );
}
