#!/usr/bin/env node
/**
 * Generate docs-integrations/hermes.md from the integration's README.
 *
 * The two pages were maintained by hand and drifted badly: by the time Hindsight left the Hermes
 * tree the doc page documented 15 config keys the plugin never read (`apiPort`, `retainRoles`,
 * camelCase spellings of real ones) and omitted 24 that it did, while still describing the bundled
 * provider as the copy that wins. The README is the single source now; this rewrites the doc page
 * from it so "keep them in sync" is mechanical instead of a habit.
 *
 * Same shape as sync-coding-agents-doc.mjs, and the adjustments are for the same reasons:
 *   - Docusaurus frontmatter replaces the README's H1 (the title comes from frontmatter), and
 *     carries the title/description that check-integration-seo.mjs requires.
 *   - Repo-relative links are dropped to plain text: they resolve on GitHub but 404 on the site.
 *   - Contributor-only sections are dropped — a reader of the docs site is not editing this tree.
 *
 * Run: node hindsight-docs/scripts/sync-hermes-doc.mjs [--check]
 * `--check` fails when the doc page is out of date instead of writing it (for CI).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const readme = join(here, '..', '..', 'hindsight-integrations', 'hermes', 'README.md');
const page = join(here, '..', 'docs-integrations', 'hermes.md');

// Kept here rather than derived, so title/description stay tuned for search without touching the
// README's own heading. sidebar_position preserves the page's existing slot in the sidebar.
const FRONTMATTER = `---
sidebar_position: 10
title: "Hermes Agent Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Hermes Agent with Hindsight. Install from the Hermes plugin catalog, then recall context automatically before every turn and retain conversations for future sessions."
---

{/* GENERATED from hindsight-integrations/hermes/README.md — edit that file, then run
    node hindsight-docs/scripts/sync-hermes-doc.mjs */}
`;

/** Sections that only make sense inside the repo (contributor-facing), dropped from the doc page. */
const DROP_SECTIONS = ['Development'];

/**
 * Prepended after the frontmatter: site-only context that has no place in a README read from the
 * plugin directory (the Desktop app is a different surface, and the old pip plugin is a docs-era
 * migration note pointing at a guide that only exists on the site).
 */
const SITE_ONLY_INTRO = `:::tip
Using the **Hermes desktop app**? You can select and configure Hindsight entirely in Settings — no
terminal required. See [Hermes Desktop](/sdks/integrations/hermes-desktop).
:::

:::warning Deprecated: the standalone \`hindsight-hermes\` plugin
The old **\`hindsight-hermes\`** pip plugin (installed into the Hermes virtual environment and
registered through the \`hermes_agent.plugins\` entry point) is **deprecated** — on current Hermes
builds its tools fail with \`{"error": "Timeout context manager should be used inside a task"}\`.
Follow [Migrate hindsight-hermes to Native Hermes Memory](/guides/2026/04/14/guide-migrate-hindsight-hermes-to-native-hermes-memory)
to switch over while keeping the same memory bank.
:::
`;

function build() {
  const src = readFileSync(readme, 'utf8');
  const out = [];
  let dropping = false;
  for (const line of src.split('\n')) {
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      const [, hashes, text] = heading;
      if (hashes.length === 1) continue; // the H1 becomes frontmatter `title`
      // Only an H2 opens or closes a dropped section. Recomputing on every heading would let an
      // H3 *inside* a dropped section switch dropping back off and leak the rest of it.
      if (hashes.length === 2) {
        dropping = DROP_SECTIONS.includes(text.trim());
        if (dropping) continue;
      } else if (dropping) {
        continue;
      }
    }
    if (!dropping) out.push(line);
  }
  const body = out
    .join('\n')
    // Repo-relative links 404 on the docs site; keep the label, drop the link.
    .replace(/\[([^\]]+)\]\((?!https?:|\/)[^)]+\)/g, '$1')
    // Our own absolute URLs -> site-relative, so the page stays within this build.
    .replace(/https:\/\/hindsight\.vectorize\.io\/([^\s"')]+)/g, '/$1')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return `${FRONTMATTER}\n${SITE_ONLY_INTRO}\n${body}\n`;
}

const generated = build();
if (process.argv.includes('--check')) {
  if (readFileSync(page, 'utf8') !== generated) {
    console.error(
      '[hermes] ❌ docs page is out of date with the README.\n' +
        '  Run: node hindsight-docs/scripts/sync-hermes-doc.mjs',
    );
    process.exit(1);
  }
  console.log('[hermes] ✅ docs page matches the README.');
} else {
  writeFileSync(page, generated);
  console.log(`[hermes] wrote ${page} from the README.`);
}
