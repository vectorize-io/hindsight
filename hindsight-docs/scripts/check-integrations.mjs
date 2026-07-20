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
 *      When a released integration is missing from the PR's integrations.json,
 *      the script also checks the base branch (GITHUB_BASE_REF or origin/main).
 *      If the entry is missing from the base branch too, the tag was created by
 *      a *different* PR or a recent merge — downgrading to a warning avoids
 *      breaking every open PR whenever a new integration is released.  Only a
 *      true regression (entry present in base, removed by this PR) is an error.
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

// ── helpers ────────────────────────────────────────────────────────────────────

function loadIntegrations(path) {
  const raw = readFileSync(path, 'utf8');
  return JSON.parse(raw).integrations;
}

/** Build the set of documented slugs from an integrations array. */
function documentedSlugs(integrations) {
  return new Set(
    integrations
      .filter((e) => e.link && e.link.startsWith('/sdks/integrations/'))
      .map((e) => e.link.replace('/sdks/integrations/', '')),
  );
}

/**
 * Return the base branch's integrations array (or null if unavailable).
 * Uses GITHUB_BASE_REF when running in CI, falling back to origin/main.
 */
function baseBranchIntegrations() {
  const baseRef =
    process.env.GITHUB_BASE_REF && process.env.GITHUB_BASE_REF !== ''
      ? `origin/${process.env.GITHUB_BASE_REF}`
      : 'origin/main';
  try {
    const raw = execFileSync(
      'git',
      ['show', `${baseRef}:hindsight-docs/src/data/integrations.json`],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
    );
    return JSON.parse(raw).integrations;
  } catch {
    return null;
  }
}

/** Return the set of released integration names from git tags. */
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

// ── main ───────────────────────────────────────────────────────────────────────

const integrations = loadIntegrations(integrationsJson);
const documented = documentedSlugs(integrations);

let failed = false;

// ─── 1. Forward: every internal entry has a doc page ──────────────────────────
const missingPages = [];
for (const entry of integrations.filter((e) => e.link && e.link.startsWith('/sdks/integrations/'))) {
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
  console.log(
    `[integrations] ✅ All ${integrations.filter((e) => e.link && e.link.startsWith('/sdks/integrations/')).length} integration entries have a doc page.`,
  );
}

// ─── 2. Reverse: every released integration is in integrations.json ───────────
const released = releasedIntegrations();
if (released === null || released.size === 0) {
  console.warn(
    '[integrations] ⚠️  No integration tags found (shallow checkout?). ' +
      'Skipping reverse check — fetch tags (fetch-depth: 0) to enforce in CI.',
  );
} else {
  const missingFromPr = [...released]
    .filter((name) => !EXCLUDED.has(name) && !documented.has(name))
    .sort();

  if (missingFromPr.length === 0) {
    console.log(
      `[integrations] ✅ All ${released.size - EXCLUDED.size} released integrations are present in integrations.json.`,
    );
  } else {
    // Differentiate: is the entry missing from the base branch too?
    // If it's missing from base, the tag was created by a different PR or
    // a recent merge — downgrade to a warning so it doesn't break every
    // open PR whenever a new integration is released.
    const baseIntegrations = baseBranchIntegrations();
    const baseDocumented = baseIntegrations ? documentedSlugs(baseIntegrations) : null;
    const regressions = []; // in base but removed by this PR → error
    const newReleases = []; // missing from base too → warning (not this PR's fault)

    if (baseDocumented) {
      for (const name of missingFromPr) {
        (baseDocumented.has(name) ? regressions : newReleases).push(name);
      }
    } else {
      // Can't determine base state — assume upstream drift, warn only.
      newReleases.push(...missingFromPr);
    }

    if (regressions.length > 0) {
      failed = true;
      console.error(
        '[integrations] ❌ Integrations present in base but removed by this PR:\n',
      );
      for (const name of regressions) {
        console.error(
          `  ${name} — in base integrations.json but missing from this branch`,
        );
      }
      console.error(
        '\nRestore the entry in integrations.json to fix this regression.\n',
      );
    }

    if (newReleases.length > 0) {
      const ref =
        process.env.GITHUB_BASE_REF && process.env.GITHUB_BASE_REF !== ''
          ? process.env.GITHUB_BASE_REF
          : 'main';
      console.warn(
        `[integrations] ⚠️  New releases not yet in integrations.json (likely merged to ${ref} without updating integrations.json — not caused by this PR):\n`,
      );
      for (const name of newReleases) {
        console.warn(
          `  ${name} — released as integrations/${name}/vX.Y.Z but no entry in integrations.json (base branch also missing it)`,
        );
      }
      console.warn(
        '\nThis is a warning (not an error) because the release was not made by this PR.\n' +
          'The maintainer who released the integration should add it to integrations.json.\n',
      );
    }
  }
}

process.exit(failed ? 1 : 0);
