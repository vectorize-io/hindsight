#!/usr/bin/env node
/**
 * Generate docs-integrations/coding-agents.md from the integration's README.
 *
 * The two pages said the same things in different words and drifted — the docs page kept install
 * instructions that no longer worked and a harness table that had fallen behind. The README is the
 * single source now; this rewrites the doc page from it so "keep them in sync" is mechanical
 * instead of a habit.
 *
 * Two adjustments are applied, both because the audience differs:
 *   - Docusaurus frontmatter replaces the README's H1 (the title comes from frontmatter).
 *   - Repo-relative links (`src/…`, `../other-integration`) are dropped to plain text: they resolve
 *     on GitHub but 404 on the docs site.
 *   - Absolute asset URLs on our own domain become site-relative. The README must use absolute URLs
 *     so images render on npm and GitHub, but on the docs site those same URLs pin every image to
 *     PRODUCTION — so a new asset shows as broken locally and in previews until it is deployed,
 *     which is exactly when you are trying to look at it.
 *
 * Run: node hindsight-docs/scripts/sync-coding-agents-doc.mjs [--check]
 * `--check` fails when the doc page is out of date instead of writing it (for CI).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const readme = join(here, '..', '..', 'hindsight-integrations', 'coding-agents', 'README.md');
const page = join(here, '..', 'docs-integrations', 'coding-agents.md');

// Kept in the doc page's frontmatter rather than derived, so title/description stay tuned for
// search without touching the README's own heading.
const FRONTMATTER = `---
sidebar_position: 6
title: "Coding Agents"
description: "One Hindsight memory plugin for coding agents — per-repo memory banks built automatically from git history and past sessions, injected into the agent as it works."
---

{/* GENERATED from hindsight-integrations/coding-agents/README.md — edit that file, then run
    node hindsight-docs/scripts/sync-coding-agents-doc.mjs */}
`;

/** Sections that only make sense inside the repo (contributor-facing), dropped from the doc page. */
const DROP_SECTIONS = ['Layout', 'Ingestion internals (no CLI)'];

function build() {
  const src = readFileSync(readme, 'utf8');
  const lines = src.split('\n');
  const out = [];
  let dropping = false;
  for (const line of lines) {
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      const [, hashes, text] = heading;
      if (hashes.length === 1) continue; // the H1 becomes frontmatter `title`
      dropping = hashes.length === 2 && DROP_SECTIONS.includes(text.trim());
      if (dropping) continue;
    }
    if (!dropping) out.push(line);
  }
  const body = out
    .join('\n')
    // Repo-relative links 404 on the docs site; keep the label, drop the link.
    .replace(/\[([^\]]+)\]\((?!https?:|\/)[^)]+\)/g, '$1')
    // Our own absolute URLs -> site-relative. Assets so the page uses THIS build's static
    // files rather than whatever is live in production; doc links so they resolve within the
    // site (and so the docs-skill generator can turn them into file-relative paths, which it
    // cannot do with an absolute URL). The README keeps them absolute because it also renders
    // on GitHub, where site-relative would 404.
    .replace(/https:\/\/hindsight\.vectorize\.io\/([^\s"')]+)/g, '/$1')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return `${FRONTMATTER}\n${body}\n`;
}

const generated = build();
if (process.argv.includes('--check')) {
  const current = readFileSync(page, 'utf8');
  if (current !== generated) {
    console.error(
      '[coding-agents] ❌ docs page is out of date with the README.\n' +
        '  Run: node hindsight-docs/scripts/sync-coding-agents-doc.mjs',
    );
    process.exit(1);
  }
  console.log('[coding-agents] ✅ docs page matches the README.');
} else {
  writeFileSync(page, generated);
  console.log(`[coding-agents] wrote ${page} from the README.`);
}
