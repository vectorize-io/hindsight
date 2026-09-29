#!/usr/bin/env node
/**
 * Validates that every integration docs page has `title` and `description`
 * in its frontmatter for SEO purposes.
 *
 * Only checks docs/sdks/integrations/ (the current/unreleased version).
 * Versioned docs are frozen snapshots and checked separately on release.
 *
 * Run: node scripts/check-integration-seo.mjs
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const integrationsDir = join(__dirname, '..', 'docs-integrations');

const IGNORED_FILES = ['_template.md', '_category_.json'];

function parseFrontmatter(content) {
  if (!content.startsWith('---')) return {};
  const end = content.indexOf('\n---', 3);
  if (end === -1) return {};
  const fm = content.slice(4, end);
  const fields = {};
  for (const line of fm.split('\n')) {
    const match = line.match(/^(\w[\w-]*):\s*(.+)$/);
    if (match) fields[match[1]] = match[2].trim();
  }
  return fields;
}

// ─── Main ────────────────────────────────────────────────────────────────────

const files = readdirSync(integrationsDir).filter(
  f => (f.endsWith('.md') || f.endsWith('.mdx')) && !IGNORED_FILES.includes(f)
);

const violations = [];

for (const filename of files) {
  const filepath = join(integrationsDir, filename);
  const content = readFileSync(filepath, 'utf8');
  const fm = parseFrontmatter(content);
  const missing = [];
  if (!fm.title) missing.push('title');
  if (!fm.description) missing.push('description');
  if (missing.length > 0) {
    violations.push({ filename, missing });
  }
}

if (violations.length > 0) {
  console.error('[integration-seo] ❌ The following integration pages are missing required frontmatter:\n');
  for (const { filename, missing } of violations) {
    console.error(`  docs-integrations/${filename} — missing: ${missing.join(', ')}`);
  }
  console.error('\nAll integration pages must have both `title` and `description` in their frontmatter.');
  console.error('Example:\n');
  console.error('  ---');
  console.error('  sidebar_position: 1');
  console.error('  title: "MyFramework Persistent Memory with Hindsight | Integration"');
  console.error('  description: "Add long-term memory to MyFramework agents with Hindsight. ..."');
  console.error('  ---');
  process.exit(1);
} else {
  console.log(`[integration-seo] ✅ All ${files.length} integration pages have title and description.`);
}
