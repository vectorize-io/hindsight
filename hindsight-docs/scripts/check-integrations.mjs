#!/usr/bin/env node
/**
 * Integrations single-source-of-truth guardrails.
 *
 * src/data/integrations.json is the single source: it drives the /integrations
 * gallery and (via the swizzled DocPage/Layout/Sidebar component) the
 * Integrations sidebar category on every docs version. This script enforces the
 * two invariants that keep it honest:
 *
 *   1. Forward — every entry with an internal `/sdks/integrations/<slug>` link
 *      has a doc page at docs-integrations/<slug>.{md,mdx}. (The sidebar is
 *      injected at render time, so it isn't covered by Docusaurus' build-time
 *      broken-link check — this is what catches a missing page.)
 *
 *   2. Reverse — every *released* integration (a published git tag
 *      `integrations/<name>/vX.Y.Z`) appears in integrations.json, so a release
 *      can't silently skip the gallery/sidebar. Degrades to a skip when tags
 *      aren't available (shallow checkout); CI fetches tags (fetch-depth: 0).
 *
 * Run: node scripts/check-integrations.mjs
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsDir = join(__dirname, '..');
const integrationsJson = join(docsDir, 'src', 'data', 'integrations.json');
const integrationsDocsDir = join(docsDir, 'docs-integrations');

// Released integrations that intentionally have no gallery/doc page.
// cloudflare-oauth-proxy is internal infrastructure (an OAuth proxy Worker,
// `"private": true`), not a user-facing framework integration.
const EXCLUDED = new Set(['cloudflare-oauth-proxy']);

const { integrations } = JSON.parse(readFileSync(integrationsJson, 'utf8'));
const internal = integrations.filter((entry) => entry.link.startsWith('/sdks/integrations/'));

let failed = false;

// ─── 1. Forward: every internal entry has a doc page ──────────────────────────
const missingPages = [];
for (const entry of internal) {
  const slug = entry.link.replace('/sdks/integrations/', '');
  const hasDoc = ['md', 'mdx'].some((ext) => existsSync(join(integrationsDocsDir, `${slug}.${ext}`)));
  if (!hasDoc) {
    missingPages.push({ id: entry.id, slug });
  }
}
if (missingPages.length > 0) {
  failed = true;
  console.error('[integrations] ❌ integrations.json entries with no doc page:\n');
  for (const { id, slug } of missingPages) {
    console.error(`  ${id} — expected docs-integrations/${slug}.{md,mdx}`);
  }
  console.error('\nAdd the doc page, or remove or externalize the entry in integrations.json.\n');
} else {
  console.log(`[integrations] ✅ All ${internal.length} integration entries have a doc page.`);
}

// ─── 2. Reverse: every released integration is in integrations.json ───────────
const documented = new Set(internal.map((entry) => entry.link.replace('/sdks/integrations/', '')));

function releasedIntegrations() {
  let raw;
  try {
    raw = execFileSync('git', ['tag', '-l', 'integrations/*'], { encoding: 'utf8' });
  } catch {
    return null; // git unavailable
  }
  const names = new Set();
  for (const tag of raw.split('\n')) {
    const m = tag.match(/^integrations\/(.+)\/v\d/);
    if (m) names.add(m[1]);
  }
  return names;
}

const released = releasedIntegrations();
if (released === null || released.size === 0) {
  console.warn(
    '[integrations] ⚠️  No integration tags found (shallow checkout?). ' +
      'Skipping reverse check — fetch tags (fetch-depth: 0) to enforce in CI.',
  );
} else {
  const missingEntries = [...released]
    .filter((name) => !EXCLUDED.has(name) && !documented.has(name))
    .sort();
  if (missingEntries.length > 0) {
    failed = true;
    console.error('[integrations] ❌ Released integrations missing from integrations.json:\n');
    for (const name of missingEntries) {
      console.error(`  ${name} — released as integrations/${name}/vX.Y.Z but no entry in integrations.json`);
    }
    console.error(
      '\nAdd each to integrations.json (with an internal `link`), or — if not user-facing — ' +
        'add it to the EXCLUDED set in this script.\n',
    );
  } else {
    console.log(`[integrations] ✅ All ${released.size - EXCLUDED.size} released integrations are present in integrations.json.`);
  }
}

process.exit(failed ? 1 : 0);
