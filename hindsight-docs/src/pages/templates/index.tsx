import React, {useMemo, useState, useCallback} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import catalog from '@site/src/data/templates.json';
import integrationsData from '@site/src/data/integrations.json';
import styles from './index.module.css';

const TEMPLATES_JSON_URL =
  'https://github.com/vectorize-io/hindsight/edit/main/hindsight-docs/src/data/templates.json';

// Webpack's require.context eagerly bundles every .json file under
// src/data/templates/, so adding a template only requires creating
// the manifest file and adding a catalog entry — no code change here.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const manifestContext = (require as any).context('@site/src/data/templates', false, /\.json$/);

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  integrations?: string[];
  manifest: Record<string, unknown>;
}

const templatesData: Template[] = catalog.templates.map((entry) => {
  // catalog's manifest_file is 'templates/<id>.json'; require.context keys are './<id>.json'
  const key = './' + entry.manifest_file.replace(/^templates\//, '');
  const manifest = manifestContext(key);
  if (!manifest) {
    throw new Error(`No manifest found for ${entry.manifest_file}`);
  }
  return {
    id: entry.id,
    name: entry.name,
    description: entry.description,
    category: entry.category,
    integrations: entry.integrations,
    manifest,
  };
});

const CATEGORIES = ['all', 'chat', 'coding', 'assistant'] as const;
type Category = (typeof CATEGORIES)[number];

const CATEGORY_LABELS: Record<Category, string> = {
  all: 'All',
  chat: 'Chat',
  coding: 'Coding',
  assistant: 'Assistant',
};

// Build a lookup from integration ID to icon path and name
const INTEGRATION_MAP = Object.fromEntries(
  integrationsData.integrations.map((i) => [i.id, {icon: i.icon, name: i.name}]),
);

function IntegrationIcons({ids}: {ids: string[]}) {
  return (
    <div className={styles.integrationIcons}>
      {ids.map((id) => {
        const info = INTEGRATION_MAP[id];
        if (!info?.icon) return null;
        const src = useBaseUrl(info.icon);
        return <img key={id} src={src} alt={info.name} title={info.name} className={styles.integrationIcon} />;
      })}
    </div>
  );
}

function TemplateCard({template, onSelect}: {template: Template; onSelect: () => void}) {
  return (
    <button className={styles.card} onClick={onSelect}>
      <div className={styles.cardHeader}>
        <span className={styles.categoryBadge}>{template.category}</span>
        {template.integrations && template.integrations.length > 0 && (
          <IntegrationIcons ids={template.integrations} />
        )}
      </div>
      <div className={styles.cardBody}>
        <h3 className={styles.cardTitle}>{template.name}</h3>
        <p className={styles.cardDescription}>{template.description}</p>
      </div>
      <div className={styles.cardFooter}>
        <span className={styles.viewLabel}>View manifest &rarr;</span>
      </div>
    </button>
  );
}

function ManifestModal({
  template,
  onClose,
}: {
  template: Template;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(template.manifest, null, 2);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(json);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [json]);

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <div>
            <h2 className={styles.modalTitle}>{template.name}</h2>
            <p className={styles.modalDescription}>{template.description}</p>
          </div>
          <button className={styles.modalClose} onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>
        <div className={styles.modalBody}>
          <div className={styles.manifestHeader}>
            <span className={styles.manifestLabel}>Template Manifest</span>
            <button className={styles.copyButton} onClick={handleCopy}>
              {copied ? 'Copied!' : 'Copy JSON'}
            </button>
          </div>
          <pre className={styles.manifestCode}>
            <code>{json}</code>
          </pre>
        </div>
        <div className={styles.modalFooter}>
          <p className={styles.usageHint}>
            Use this manifest with <code>POST /v1/default/banks/&#123;bank_id&#125;/import</code> or
            paste it in the bank creation dialog in the control plane.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function TemplateGallery(): React.ReactElement {
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<Category>('all');
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);

  const templates = templatesData;

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    return templates.filter((t) => {
      if (selectedCategory !== 'all' && t.category !== selectedCategory) return false;
      if (q && !t.name.toLowerCase().includes(q) && !t.description.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [templates, search, selectedCategory]);

  return (
    <Layout title="Bank Templates Hub" description="Pre-built bank templates for common use cases">
      <div className={styles.heroSection}>
        <h1 className={styles.heroTitle}>Bank Templates Hub</h1>
        <p className={styles.heroSubtitle}>
          Pre-built bank templates to get started fast. Browse, preview, and import into your Hindsight banks.
          {' '}<a href="/developer/api/bank-templates" className={styles.heroLink}>Learn how templates work &rarr;</a>
        </p>

        <div className={styles.searchWrapper}>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Search templates..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search templates"
            autoComplete="off"
            autoFocus
          />
          {search && (
            <button className={styles.searchClear} onClick={() => setSearch('')} aria-label="Clear search">
              &times;
            </button>
          )}
        </div>

        <div className={styles.heroStats}>
          <span className={styles.stat}>
            <strong>{templates.length}</strong> templates
          </span>
          <span className={styles.statDivider}>&middot;</span>
          <span className={styles.stat}>
            <strong>{new Set(templates.map((t) => t.category)).size}</strong> categories
          </span>
        </div>
      </div>

      <div className={styles.page}>
        <div className={styles.toolbar}>
          <div className={styles.filterGroup}>
            {CATEGORIES.map((c) => (
              <button
                key={c}
                className={`${styles.filterPill} ${selectedCategory === c ? styles.filterPillActive : ''}`}
                onClick={() => setSelectedCategory(c)}>
                {CATEGORY_LABELS[c]}
              </button>
            ))}
          </div>
          <span className={styles.resultCount}>
            {filtered.length} template{filtered.length !== 1 ? 's' : ''}
          </span>
        </div>

        {filtered.length === 0 ? (
          <div className={styles.empty}>
            <p>No templates match your search.</p>
            <button
              className={styles.resetButton}
              onClick={() => {
                setSearch('');
                setSelectedCategory('all');
              }}>
              Reset filters
            </button>
          </div>
        ) : (
          <div className={styles.grid}>
            {filtered.map((t) => (
              <TemplateCard key={t.id} template={t} onSelect={() => setSelectedTemplate(t)} />
            ))}
          </div>
        )}

        <div className={styles.submitBanner}>
          <div className={styles.submitBannerContent}>
            <h3 className={styles.submitBannerTitle}>Have a template to share?</h3>
            <p className={styles.submitBannerText}>
              Contribute it to the community. Open a pull request and add your entry to the bank templates.
            </p>
            <Link
              href={TEMPLATES_JSON_URL}
              className={styles.submitButton}
              target="_blank"
              rel="noopener noreferrer">
              Submit a template →
            </Link>
          </div>
        </div>
      </div>

      {selectedTemplate && <ManifestModal template={selectedTemplate} onClose={() => setSelectedTemplate(null)} />}
    </Layout>
  );
}
