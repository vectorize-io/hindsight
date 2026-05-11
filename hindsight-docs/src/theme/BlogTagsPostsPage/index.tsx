import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import type {Props} from '@theme/BlogTagsPostsPage';
import type {PropBlogPostContent} from '@docusaurus/plugin-content-blog';
import styles from '../BlogListPage/styles.module.css';

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

const TAG_LABELS: Record<string, {title: string; subtitle: string}> = {
  'hindsight-cloud': {
    title: 'Hindsight Cloud',
    subtitle: 'News and updates from Hindsight Cloud',
  },
  'deep-dive': {
    title: 'Deep Dives',
    subtitle: 'Architecture, engineering, and benchmarks',
  },
  'release': {
    title: 'Announcements & Releases',
    subtitle: 'New versions, milestones, and product news',
  },
  'tutorial': {
    title: 'Tutorials & Integrations',
    subtitle: 'How to use Hindsight with your stack',
  },
};

export default function BlogTagsPostsPage({tag, items, listMetadata}: Props): React.ReactElement {
  const {totalPages, page, nextPage, previousPage} = listMetadata;
  const tagInfo = TAG_LABELS[tag.label] ?? {title: tag.label, subtitle: undefined};

  return (
    <Layout title={tagInfo.title} description={tagInfo.subtitle}>
      <main className={styles.blogPage}>
        <header className={styles.header}>
          <h1 className={styles.headerTitle}>{tagInfo.title}</h1>
          {tagInfo.subtitle && <p className={styles.headerSubtitle}>{tagInfo.subtitle}</p>}
        </header>

        <div className={styles.grid}>
          {items.map(({content: BlogPostContent}) => (
            <BlogCard key={BlogPostContent.metadata.permalink} content={BlogPostContent} />
          ))}
        </div>

        {totalPages > 1 && (
          <nav className={styles.pagination}>
            {previousPage && (
              <Link to={previousPage} className={styles.paginationButton}>
                ← Previous
              </Link>
            )}
            <span className={styles.paginationInfo}>
              Page {page} of {totalPages}
            </span>
            {nextPage && (
              <Link to={nextPage} className={styles.paginationButton}>
                Next →
              </Link>
            )}
          </nav>
        )}
      </main>
    </Layout>
  );
}
