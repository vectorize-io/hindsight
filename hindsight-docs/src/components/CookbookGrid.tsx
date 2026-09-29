import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './CookbookGrid.module.css';

export interface CookbookCard {
  title: string;
  href: string;
  description?: string;
  tags?: {
    sdk?: string;
    topic?: string;
  };
}

interface CookbookGridProps {
  items: CookbookCard[];
}

function sdkIcon(sdk: string): string | null {
  if (sdk.startsWith('@') || sdk.includes('node') || sdk.includes('chat') || sdk.includes('ai-sdk')) {
    return '/img/icons/nodejs.png';
  }
  if (sdk.includes('-go') || sdk === 'go') {
    return '/img/icons/golang.png';
  }
  if (sdk.includes('agno')) {
    return '/img/icons/agno.png';
  }
  if (sdk.includes('smolagents')) {
    return '/img/icons/smolagents.png';
  }
  if (sdk.includes('strands')) {
    return '/img/icons/strands.png';
  }
  if (sdk.includes('hindsight-client') || sdk.includes('hindsight-api') || sdk.includes('litellm') || sdk.includes('pydantic') || sdk.includes('crewai')) {
    return '/img/icons/python.svg';
  }
  return null;
}

function SdkIcon({sdk, className}: {sdk: string; className?: string}) {
  const icon = sdkIcon(sdk);
  const src = useBaseUrl(icon ?? '');
  if (!icon) return null;
  return <img src={src} alt="" className={className} aria-hidden />;
}

function Card({title, href, description, tags}: CookbookCard) {
  return (
    <Link to={href} className={styles.card}>
      <div className={styles.cardBody}>
        <h3 className={styles.cardTitle}>{title}</h3>
        {description && <p className={styles.cardDescription}>{description}</p>}
        {(tags?.topic || tags?.sdk) && (
          <div className={styles.cardFooter}>
            {tags.topic && <span className={styles.cardTopic}>{tags.topic}</span>}
            {tags.sdk && (
              <span className={styles.cardSdk}>
                <SdkIcon sdk={tags.sdk} className={styles.sdkIcon} />
                {tags.sdk}
              </span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}

export default function CookbookGrid({items}: CookbookGridProps) {
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedSdk, setSelectedSdk] = useState<string | null>(null);

  const topics = [...new Set(items.map((i) => i.tags?.topic).filter(Boolean))] as string[];
  const sdks = [...new Set(items.map((i) => i.tags?.sdk).filter(Boolean))] as string[];

  const filtered = items.filter((item) => {
    if (selectedTopic && item.tags?.topic !== selectedTopic) return false;
    if (selectedSdk && item.tags?.sdk !== selectedSdk) return false;
    return true;
  });

  const hasFilters = topics.length > 1 || sdks.length > 1;

  return (
    <div>
      {hasFilters && (
        <div className={styles.filters}>
          {topics.length > 1 && (
            <div className={styles.filterGroup}>
              <span className={styles.filterLabel}>Topic</span>
              <button
                className={`${styles.filterPill} ${selectedTopic === null ? styles.filterPillActive : ''}`}
                onClick={() => setSelectedTopic(null)}>
                All
              </button>
              {topics.map((topic) => (
                <button
                  key={topic}
                  className={`${styles.filterPill} ${selectedTopic === topic ? styles.filterPillActive : ''}`}
                  onClick={() => setSelectedTopic(selectedTopic === topic ? null : topic)}>
                  {topic}
                </button>
              ))}
            </div>
          )}
          {sdks.length > 1 && (
            <div className={styles.filterGroup}>
              <span className={styles.filterLabel}>SDK</span>
              <button
                className={`${styles.filterPill} ${selectedSdk === null ? styles.filterPillActive : ''}`}
                onClick={() => setSelectedSdk(null)}>
                All
              </button>
              {sdks.map((sdk) => (
                <button
                  key={sdk}
                  className={`${styles.filterPill} ${selectedSdk === sdk ? styles.filterPillActive : ''}`}
                  onClick={() => setSelectedSdk(selectedSdk === sdk ? null : sdk)}>
                  <SdkIcon sdk={sdk} className={styles.filterPillIcon} />
                  {sdk}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <div className={styles.grid}>
        {filtered.map((item) => (
          <Card key={item.href} {...item} />
        ))}
      </div>
    </div>
  );
}
