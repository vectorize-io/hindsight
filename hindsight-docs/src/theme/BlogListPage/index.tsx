import React from 'react';
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

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
}

function BlogCard({content}: {content: PropBlogPostContent}) {
  const {metadata, assets} = content;
  const {title, description, date, readingTime, permalink, frontMatter} = metadata;
  const image = assets.image ?? frontMatter.image ?? '/img/blog-default.jpg';

  return (
    <Link to={permalink} className={styles.card}>
      <div className={styles.cardImageWrapper}>
        {image ? (
          <img src={image} alt={title} className={styles.cardImage} />
        ) : (
          <div className={styles.cardImagePlaceholder} />
        )}
      </div>
      <div className={styles.cardBody}>
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
  const currentPage = Math.max(1, parseInt(searchParams.get('page') ?? '1', 10));

  const filteredItems = activeCategory.tag
    ? items.filter(({content}) => postHasTag(content, activeCategory.tag!))
    : items;

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const safePage = Math.min(currentPage, totalPages);
  const pagePosts = filteredItems.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const selectCategory = (slug: string) => {
    const params = new URLSearchParams();
    if (slug !== 'all') {
      params.set('cat', slug);
    }
    const search = params.toString();
    history.push({pathname: location.pathname, search: search ? `?${search}` : ''});
  };

  const goToPage = (page: number) => {
    const params = new URLSearchParams(location.search);
    if (page === 1) {
      params.delete('page');
    } else {
      params.set('page', String(page));
    }
    const search = params.toString();
    history.push({pathname: location.pathname, search: search ? `?${search}` : ''});
  };

  return (
    <Layout title={blogTitle} description={blogDescription}>
      <main className={styles.blogPage}>
        <PageHero title={blogTitle} subtitle={blogDescription} />

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

        {pagePosts.length > 0 ? (
          <section className={styles.section}>
            <div className={styles.grid}>
              {pagePosts.map(({content: BlogPostContent}) => (
                <BlogCard key={BlogPostContent.metadata.permalink} content={BlogPostContent} />
              ))}
            </div>
          </section>
        ) : (
          <p className={styles.emptyState}>No posts in this category yet.</p>
        )}

        {totalPages > 1 && (
          <nav className={styles.pagination}>
            {safePage > 1 && (
              <button onClick={() => goToPage(safePage - 1)} className={styles.paginationButton}>
                ← Previous
              </button>
            )}
            <span className={styles.paginationInfo}>
              Page {safePage} of {totalPages}
            </span>
            {safePage < totalPages && (
              <button onClick={() => goToPage(safePage + 1)} className={styles.paginationButton}>
                Next →
              </button>
            )}
          </nav>
        )}
      </main>
    </Layout>
  );
}
