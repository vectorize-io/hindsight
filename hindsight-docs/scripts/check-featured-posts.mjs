#!/usr/bin/env node
/**
 * Validates src/data/featured-posts.json, the blog posts pinned in the docs
 * sidebar and at the top of the blog index: at most `max` pins, every `href`
 * is a real post, every `cover` is a real file under static/.
 *
 * Run: node scripts/check-featured-posts.mjs
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const { max, posts } = JSON.parse(readFileSync(join(docsRoot, 'src/data/featured-posts.json'), 'utf-8'));

// A post's URL is /blog/<slug> from frontmatter, else Docusaurus' default
// /blog/YYYY/MM/DD/<name> from a YYYY-MM-DD-<name>.md(x) filename.
const postHrefs = new Set();
for (const file of readdirSync(join(docsRoot, 'blog'))) {
  const named = file.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)\.mdx?$/);
  if (!named) continue;
  const slug = readFileSync(join(docsRoot, 'blog', file), 'utf-8').match(/^slug:\s*"?([^"\n]+?)"?\s*$/m);
  postHrefs.add(slug ? `/blog/${slug[1].replace(/^\//, '')}` : `/blog/${named[1]}/${named[2]}/${named[3]}/${named[4]}`);
}

const errors = [];
if (posts.length > max) {
  errors.push(`${posts.length} pinned posts, max is ${max}. Unpin one first.`);
}
for (const post of posts) {
  if (!postHrefs.has(post.href)) errors.push(`"${post.href}" is not a blog post.`);
  if (!existsSync(join(docsRoot, 'static', post.cover))) errors.push(`cover "${post.cover}" not found under static/.`);
}

if (errors.length > 0) {
  for (const e of errors) console.error(`\x1b[31m✗\x1b[0m featured-posts.json: ${e}`);
  process.exit(1);
}
console.log(`\x1b[32m✓\x1b[0m ${posts.length}/${max} featured posts are valid.`);
