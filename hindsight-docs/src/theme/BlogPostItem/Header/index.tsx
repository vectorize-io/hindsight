import React, {type ReactNode} from 'react';
import {useBlogPost} from '@docusaurus/plugin-content-blog/client';
import BlogPostItemHeaderTitle from '@theme/BlogPostItem/Header/Title';
import BlogPostItemHeaderInfo from '@theme/BlogPostItem/Header/Info';
import BlogPostItemHeaderAuthors from '@theme/BlogPostItem/Header/Authors';

// On a post page: title, the frontmatter description as a subtitle, then one
// "<photo> By <authors>" / date row above a hairline. Lists keep the upstream header.
export default function BlogPostItemHeader(): ReactNode {
  const {metadata, assets, isBlogPostPage} = useBlogPost();
  if (!isBlogPostPage) {
    return (
      <header>
        <BlogPostItemHeaderTitle />
        <BlogPostItemHeaderInfo />
        <BlogPostItemHeaderAuthors />
      </header>
    );
  }
  const {authors, frontMatter} = metadata;
  const names = authors.map((a) => a.name).filter(Boolean);
  const images = authors.map((a, i) => assets.authorsImageUrls[i] ?? a.imageURL).filter(Boolean);
  return (
    <header className="hs-blog-header">
      <BlogPostItemHeaderTitle />
      {frontMatter.description && <p className="hs-blog-header__description">{frontMatter.description}</p>}
      <div className="hs-blog-header__meta">
        {names.length > 0 && (
          <span className="hs-blog-header__authors">
            {images.map((src) => (
              <img key={src} src={src} alt="" className="hs-blog-header__avatar" />
            ))}
            By {names.join(', ')}
          </span>
        )}
        <div className="hs-blog-header__info">
          <BlogPostItemHeaderInfo />
        </div>
      </div>
    </header>
  );
}
