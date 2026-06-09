import React, {useMemo, useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import IntegrationsBanner from '@site/src/components/IntegrationsBanner';
import {integrationsSorted} from '@site/src/lib/integrations';
import styles from './index.module.css';

const INTEGRATIONS_JSON_URL =
  'https://github.com/vectorize-io/hindsight/edit/main/hindsight-docs/src/data/integrations.json';

type IntegrationType = 'official' | 'community';

interface Integration {
  id: string;
  name: string;
  description: string;
  type: IntegrationType;
  by: string;
  category: string;
  link: string;
  icon?: string;
}

function IntegrationCard({integration}: {integration: Integration}) {
  const iconSrc = useBaseUrl(integration.icon ?? '');
  const faviconSrc = useBaseUrl('/img/favicon.png');
  const isExternal = integration.link.startsWith('http');

  return (
    <Link to={integration.link} className={styles.card} {...(isExternal ? {target: '_blank', rel: 'noopener noreferrer'} : {})}>
      <div className={styles.cardHeader}>
        {integration.icon && <img src={iconSrc} alt="" className={styles.cardIcon} aria-hidden />}
        <span className={`${styles.typeBadge} ${integration.type === 'official' ? styles.typeBadgeOfficial : styles.typeBadgeCommunity}`}>
          {integration.type === 'official' ? 'Official' : 'Community'}
        </span>
      </div>
      <div className={styles.cardBody}>
        <h3 className={styles.cardTitle}>{integration.name}</h3>
        <p className={styles.cardDescription}>{integration.description}</p>
      </div>
      <div className={styles.cardFooter}>
        {integration.type === 'official' ? (
          <span className={styles.byLine}>
            <img src={faviconSrc} alt="" className={styles.authorIcon} aria-hidden />
            <span className={styles.authorName}>Hindsight Team</span>
          </span>
        ) : (
          <span className={styles.byLine}>
            <img src={`https://github.com/${integration.by}.png?size=40`} alt="" className={styles.authorIcon} aria-hidden />
            <span className={styles.authorName}>@{integration.by}</span>
          </span>
        )}
      </div>
    </Link>
  );
}

export default function IntegrationsHub(): React.ReactElement {
  const [search, setSearch] = useState('');
  const [selectedType, setSelectedType] = useState<IntegrationType | 'all'>('all');

  const integrations = integrationsSorted as unknown as Integration[];

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    return integrations.filter((i) => {
      if (selectedType !== 'all' && i.type !== selectedType) return false;
      if (q && !i.name.toLowerCase().includes(q) && !i.description.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [integrations, search, selectedType]);

  const officialCount = integrations.filter((i) => i.type === 'official').length;
  const communityCount = integrations.filter((i) => i.type === 'community').length;

  return (
    <Layout title="Integrations Hub" description="Browse official and community integrations for Hindsight agent memory">

      {/* Full-width hero with its own background */}
      <div className={styles.heroSection}>
        <h1 className={styles.heroTitle}>Integrations Hub</h1>
        <p className={styles.heroSubtitle}>
          Connect Hindsight to your stack. Browse official integrations and community-built connectors.
        </p>

        <div className={styles.searchWrapper}>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Search integrations…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search integrations"
            autoComplete="off"
          />
          {search && (
            <button className={styles.searchClear} onClick={() => setSearch('')} aria-label="Clear search">
              ×
            </button>
          )}
        </div>

        <div className={styles.heroStats}>
          <span className={styles.stat}><strong>{officialCount}</strong> official</span>
          <span className={styles.statDivider}>·</span>
          <span className={styles.stat}><strong>{communityCount}</strong> community</span>
        </div>
      </div>

      {/* Scrolling banner */}
      <IntegrationsBanner />

      {/* Main content */}
      <div className={styles.page}>
        <div className={styles.toolbar}>
          <div className={styles.filterGroup}>
            {(['all', 'official', 'community'] as const).map((t) => (
              <button
                key={t}
                className={`${styles.filterPill} ${selectedType === t ? styles.filterPillActive : ''}`}
                onClick={() => setSelectedType(t)}>
                {t === 'all' ? 'All' : t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
          <span className={styles.resultCount}>{filtered.length} integration{filtered.length !== 1 ? 's' : ''}</span>
        </div>

        {filtered.length === 0 ? (
          <div className={styles.empty}>
            <p>No integrations match your search.</p>
            <button className={styles.resetButton} onClick={() => { setSearch(''); setSelectedType('all'); }}>
              Reset filters
            </button>
          </div>
        ) : (
          <div className={styles.grid}>
            {filtered.map((integration) => (
              <IntegrationCard key={integration.id} integration={integration} />
            ))}
          </div>
        )}

        <div className={styles.submitBanner}>
          <div className={styles.submitBannerContent}>
            <h2 className={styles.submitBannerTitle}>Built something with Hindsight?</h2>
            <p className={styles.submitBannerText}>
              Share your integration with the community. Open a pull request and add your entry to the integrations list.
            </p>
            <Link
              href={INTEGRATIONS_JSON_URL}
              className={styles.submitButton}
              target="_blank"
              rel="noopener noreferrer">
              Submit an integration →
            </Link>
          </div>
        </div>
      </div>
    </Layout>
  );
}
