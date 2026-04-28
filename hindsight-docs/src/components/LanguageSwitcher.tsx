import React from 'react';
import {useLocation} from '@docusaurus/router';
import Link from '@docusaurus/Link';
import styles from './LanguageSwitcher.module.css';

// List of guide slugs that have Chinese translations (2026-04-28 batch)
const TRANSLATED_GUIDES = new Set([
  'guide-connect-chatgpt-and-perplexity-to-hindsight',
  'guide-fix-openclaw-retention-and-recall-on-default-main-sessions',
  'guide-reduce-hindsight-consolidation-memory-fan-out',
  'guide-run-hindsight-cli-on-linux-arm64',
  'guide-share-hindsight-memory-across-chatgpt-and-perplexity',
  'guide-size-hindsight-memory-footprint-for-deployments',
  'guide-use-mental-model-tags-in-hindsight-list-view',
]);

export default function LanguageSwitcher(): React.ReactNode {
  const {pathname} = useLocation();

  // Detect current locale from pathname
  // /zh/guides/... = Chinese, /guides/... = English
  const isChineseUrl = pathname.startsWith('/zh/');
  const currentLocale = isChineseUrl ? 'zh-CN' : 'en';

  // Check if we're on a guides page
  const isGuidesPage = pathname.includes('/guides/');
  if (!isGuidesPage) {
    return null;
  }

  // Extract the slug from the path
  // Path format: /guides/2026/04/28/slug or /zh/guides/2026/04/28/slug
  const guideMatch = pathname.match(/guides\/(\d{4})\/(\d{2})\/(\d{2})\/([^/]+)/);
  if (!guideMatch) {
    return null;
  }

  const slug = guideMatch[4];

  // Check if this guide has a translation
  if (!TRANSLATED_GUIDES.has(slug)) {
    return null;
  }

  // Generate the alternate language URL
  let alternateUrl: string;
  if (currentLocale === 'en') {
    // Switch to Chinese
    alternateUrl = `/zh${pathname}`;
  } else {
    // Switch to English - remove /zh prefix
    alternateUrl = pathname.replace(/^\/zh/, '');
  }

  const alternateLabel = currentLocale === 'en' ? '中文' : 'English';
  const toggleTitle = currentLocale === 'en' ? 'Read in Chinese' : 'Read in English';

  return (
    <div className={styles.languageSwitcher}>
      <Link to={alternateUrl} className={styles.languageButton} title={toggleTitle}>
        <svg className={styles.icon} width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 1C4.13 1 1 4.13 1 8s3.13 7 7 7 7-3.13 7-7-3.13-7-7-7zm0 12.5c-3.04 0-5.5-2.46-5.5-5.5S4.96 2.5 8 2.5s5.5 2.46 5.5 5.5-2.46 5.5-5.5 5.5zm2.5-5.5c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5.67 1.5 1.5 1.5 1.5-.67 1.5-1.5z" fill="currentColor"/>
          <path d="M8 4c-2.2 0-4 1.8-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 1-1 2-2 3v1h2c1.1 0 2-.9 2-2 0-2.2-1.8-4-4-4z" fill="currentColor"/>
        </svg>
        {alternateLabel}
      </Link>
    </div>
  );
}
