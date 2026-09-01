import React, {useMemo, useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import IntegrationsBanner from '@site/src/components/IntegrationsBanner';
import {integrationsSorted} from '@site/src/lib/integrations';
import styles from './index.module.css';

/**
 * Pinned above the grid. These three are the ones we want a first-time visitor to see: the umbrella
 * coding-agent plugin, the SDK most TypeScript apps reach for, and the agent harness with the
 * deepest native integration.
 */
const FEATURED_IDS = ['coding-agents', 'vercel-ai-sdk', 'openclaw'];

/**
 * Agents covered by the Coding Agents plugin, drawn on its card. The whole pitch of that package is
 * "one install, every agent", which a single icon cannot convey — the row of logos is the pitch.
 * Files live in static/img/harness/, named by the harness id the plugin itself uses.
 */
const CODING_AGENT_LOGOS: {id: string; name: string; file: string}[] = [
  {id: 'claude-code', name: 'Claude Code', file: 'claude-code.png'},
  {id: 'codex', name: 'Codex CLI', file: 'codex.svg'},
  {id: 'opencode', name: 'opencode', file: 'opencode.png'},
  {id: 'kilo', name: 'Kilo CLI', file: 'kilo.svg'},
  {id: 'cursor-cli', name: 'Cursor CLI', file: 'cursor-cli.svg'},
  {id: 'copilot-cli', name: 'GitHub Copilot CLI', file: 'copilot-cli.svg'},
  {id: 'grok-build', name: 'Grok Build', file: 'grok-build.svg'},
  {id: 'qwen-code', name: 'Qwen Code', file: 'qwen-code.svg'},
  {id: 'antigravity-cli', name: 'Antigravity CLI', file: 'antigravity-cli.png'},
  {id: 'devin-cli', name: 'Devin CLI', file: 'devin-cli.svg'},
  {id: 'cline-cli', name: 'Cline CLI', file: 'cline-cli.svg'},
  {id: 'dsh', name: 'DeepSeek Harness', file: 'dsh.svg'},
];

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
  const harnessBase = useBaseUrl('/img/harness/');
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
        {integration.id === 'coding-agents' && (
          <div className={styles.harnessStrip} aria-label="Supported coding agents">
            {CODING_AGENT_LOGOS.map((h) => (
              <img
                key={h.id}
                src={`${harnessBase}${h.file}`}
                alt={h.name}
                title={h.name}
                className={styles.harnessLogo}
                loading="lazy"
              />
            ))}
          </div>
        )}
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

  // Superseded pages stay published and linked from the docs sidebar, but the gallery is where
  // people come to CHOOSE an integration — offering something we are actively migrating them off
  // would be pointing them at a dead end.
  const integrations = (integrationsSorted as unknown as Integration[]).filter(
    (i) => i.category !== 'legacy',
  );

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    return integrations.filter((i) => {
      if (selectedType !== 'all' && i.type !== selectedType) return false;
      if (q && !i.name.toLowerCase().includes(q) && !i.description.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [integrations, search, selectedType]);

  // Featured only makes sense on the unfiltered view: once someone searches or filters, pinned
  // cards would sit above results that don't match them and read as noise.
  const showFeatured = !search.trim() && selectedType === 'all';
  const featured = useMemo(
    () =>
      FEATURED_IDS.map((id) => integrations.find((i) => i.id === id)).filter(
        (i): i is Integration => Boolean(i),
      ),
    [integrations],
  );
  const rest = useMemo(
    () => (showFeatured ? filtered.filter((i) => !FEATURED_IDS.includes(i.id)) : filtered),
    [filtered, showFeatured],
  );

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

        {showFeatured && featured.length > 0 && (
          <section className={styles.featuredSection}>
            <h2 className={styles.featuredTitle}>Featured</h2>
            <div className={styles.featuredGrid}>
              {featured.map((integration) => (
                <IntegrationCard key={integration.id} integration={integration} />
              ))}
            </div>
          </section>
        )}

        {showFeatured && (
          <>
            <hr className={styles.sectionDivider} />
            <h2 className={styles.sectionTitle}>All integrations</h2>
          </>
        )}

        {filtered.length === 0 ? (
          <div className={styles.empty}>
            <p>No integrations match your search.</p>
            <button className={styles.resetButton} onClick={() => { setSearch(''); setSelectedType('all'); }}>
              Reset filters
            </button>
          </div>
        ) : (
          <div className={styles.grid}>
            {rest.map((integration) => (
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
