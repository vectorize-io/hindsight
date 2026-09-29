import React, {useEffect, useMemo, useRef, useState} from 'react';
import {flushSync} from 'react-dom';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {useLocation, useHistory} from '@docusaurus/router';
import type {Props} from '@theme/BlogListPage';
import type {PropBlogPostContent} from '@docusaurus/plugin-content-blog';
import PageHero from '@site/src/components/PageHero';
import styles from './styles.module.css';

type Category = {slug: string; label: string; tag: string | null};

const CATEGORIES: Category[] = [
  {slug: 'all', label: 'All', tag: null},
  {slug: 'cloud', label: 'Hindsight Cloud', tag: 'hindsight-cloud'},
  {slug: 'deep-dives', label: 'Deep Dives', tag: 'deep-dive'},
  {slug: 'releases', label: 'Announcements & Releases', tag: 'release'},
  {slug: 'tutorials', label: 'Tutorials & Integrations', tag: 'tutorial'},
];

const PAGE_SIZE = 9;

type DocumentWithViewTransition = Document & {
  startViewTransition?: (updateCallback: () => void) => unknown;
};

// Wrap filter-state updates in a View Transition so filtered-out cards fade
// away and surviving cards glide to their new grid position instead of
// snapping. flushSync is required: the DOM must be updated inside the
// callback, before the browser captures the "new" snapshot. Browsers without
// the API just get the instant update.
function withViewTransition(update: () => void): void {
  const doc = document as DocumentWithViewTransition;
  if (doc.startViewTransition) {
    doc.startViewTransition(() => flushSync(update));
  } else {
    update();
  }
}

// Each card needs a unique view-transition-name for the browser to track it
// across filter changes; derive it from the (unique) permalink.
function cardTransitionName(permalink: string): string {
  return `post-${permalink.replace(/[^a-zA-Z0-9-]/g, '-')}`;
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
}

// First category (skipping "All") whose tag the post carries, for the card badge.
function categoryLabelFor(content: PropBlogPostContent): string | null {
  const match = CATEGORIES.find((cat) => cat.tag && postHasTag(content, cat.tag));
  return match ? match.label : null;
}

function BlogCard({content}: {content: PropBlogPostContent}) {
  const {metadata, assets} = content;
  const {title, description, date, readingTime, permalink, frontMatter} = metadata;
  const image = assets.image ?? frontMatter.image ?? '/img/blog-default.jpg';
  const categoryLabel = categoryLabelFor(content);

  return (
    <Link
      to={permalink}
      className={styles.card}
      style={{viewTransitionName: cardTransitionName(permalink)}}
    >
      <div className={styles.cardImageWrapper}>
        {image ? (
          <img src={image} alt={title} className={styles.cardImage} />
        ) : (
          <div className={styles.cardImagePlaceholder} />
        )}
      </div>
      <div className={styles.cardBody}>
        {categoryLabel && <span className={styles.cardBadge}>{categoryLabel}</span>}
        <h2 className={styles.cardTitle}>{title}</h2>
        {description && <p className={styles.cardDescription}>{description}</p>}
        <div className={styles.cardFooter}>
          <span className={styles.cardDate}>{formatDate(date)}</span>
          {readingTime !== undefined && (
            <span className={styles.cardReadTime}>{Math.ceil(readingTime)} min read</span>
          )}
        </div>
      </div>
    </Link>
  );
}

function postHasTag(content: PropBlogPostContent, tag: string): boolean {
  return (content.metadata.tags ?? []).some((t) => t.label === tag);
}

export default function BlogListPage({items, metadata}: Props): React.ReactElement {
  const {blogTitle, blogDescription} = metadata;
  const location = useLocation();
  const history = useHistory();

  const searchParams = new URLSearchParams(location.search);
  const requestedCat = searchParams.get('cat') ?? 'all';
  const activeCategory = CATEGORIES.find((c) => c.slug === requestedCat) ?? CATEGORIES[0];

  const [query, setQuery] = useState('');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  const filteredItems = useMemo(() => {
    const byCategory = activeCategory.tag
      ? items.filter(({content}) => postHasTag(content, activeCategory.tag!))
      : items;
    const q = query.trim().toLowerCase();
    if (!q) {
      return byCategory;
    }
    return byCategory.filter(
      ({content}) =>
        content.metadata.title.toLowerCase().includes(q) ||
        (content.metadata.description ?? '').toLowerCase().includes(q),
    );
  }, [items, activeCategory.tag, query]);

  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
  }, [activeCategory.slug, query]);

  const hasMore = visibleCount < filteredItems.length;

  // Recreate the observer whenever visibleCount changes: IntersectionObserver
  // only fires on intersection *changes*, so if the sentinel is still in view
  // after a batch renders (short pages), re-observing triggers the initial
  // callback again and keeps loading until the sentinel scrolls out of range.
  useEffect(() => {
    if (!hasMore || !sentinelRef.current) {
      return undefined;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisibleCount((count) => Math.min(count + PAGE_SIZE, filteredItems.length));
        }
      },
      {rootMargin: '400px 0px'},
    );
    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [hasMore, visibleCount, filteredItems.length]);

  const visiblePosts = filteredItems.slice(0, visibleCount);

  const selectCategory = (slug: string) => {
    const params = new URLSearchParams();
    if (slug !== 'all') {
      params.set('cat', slug);
    }
    const search = params.toString();
    withViewTransition(() => {
      history.push({pathname: location.pathname, search: search ? `?${search}` : ''});
    });
  };

  return (
    <Layout title={blogTitle} description={blogDescription}>
      <main className={styles.blogPage}>
        <PageHero title={blogTitle} subtitle={blogDescription} />

        <div className={styles.controls}>
          <nav className={styles.categoryStrip} aria-label="Blog categories">
            {CATEGORIES.map((cat) => (
              <button
                key={cat.slug}
                type="button"
                onClick={() => selectCategory(cat.slug)}
                className={clsx(
                  styles.categoryPill,
                  cat.slug === activeCategory.slug && styles.categoryPillActive,
                )}
                aria-pressed={cat.slug === activeCategory.slug}
              >
                {cat.label}
              </button>
            ))}
          </nav>
          <input
            type="search"
            value={query}
            onChange={(e) => {
              const value = e.target.value;
              withViewTransition(() => setQuery(value));
            }}
            placeholder="Search posts…"
            aria-label="Search posts by title"
            className={styles.postFilterInput}
          />
        </div>

        {visiblePosts.length > 0 ? (
          <section className={styles.section}>
            <div className={styles.grid}>
              {visiblePosts.map(({content: BlogPostContent}) => (
                <BlogCard key={BlogPostContent.metadata.permalink} content={BlogPostContent} />
              ))}
            </div>
          </section>
        ) : (
          <p className={styles.emptyState}>
            {query.trim()
              ? 'No posts match your search.'
              : 'No posts in this category yet.'}
          </p>
        )}

        {hasMore && <div ref={sentinelRef} className={styles.scrollSentinel} aria-hidden="true" />}
      </main>
    </Layout>
  );
}
