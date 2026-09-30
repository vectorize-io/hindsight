import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import styles from './PageHero.module.css';

interface PageHeroProps {
  title: string;
  subtitle?: string;
}

/**
 * The band at the top of the standalone pages — FAQ, Best Practices, Cookbook,
 * Changelog, the blog and every integration changelog.
 *
 * The band itself is the shared `hs-hero-band` recipe in custom.css, the same
 * one the homepage hero and the two hub pages use, so this file carries only
 * its sizing and its own two pieces of content.
 */
export default function PageHero({title, subtitle}: PageHeroProps): ReactNode {
  return (
    <div className={clsx('hs-hero-band', styles.hero)}>
      <h1 className={styles.title}>{title}</h1>
      {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
    </div>
  );
}
