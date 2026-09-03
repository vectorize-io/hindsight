const fs = require('node:fs');
const path = require('node:path');

/**
 * Build-time index of the per-integration changelog pages.
 *
 * `src/pages/changelog/integrations/<slug>.md` is written by the release script
 * (hindsight-dev/hindsight_dev/generate_changelog.py) — nothing in the docs site
 * linked those pages, so for a long time they were reachable only by guessing
 * the URL. This plugin reads that directory at build time and exposes one entry
 * per page as global data, so the Integrations Hub can put a Changelog button on
 * exactly the integrations that have one — a new integration grows the button on
 * its first release, with nothing to update by hand.
 *
 * Display metadata (name, icon, category) is joined from integrations.json on
 * the *link slug*, not the entry id — `vercel-ai-sdk` links to
 * `/sdks/integrations/ai-sdk` and its changelog is `ai-sdk.md`. A changelog with
 * no gallery entry is skipped: that means it isn't user-facing
 * (cloudflare-oauth-proxy, an internal OAuth Worker), and check-integrations.mjs
 * already guarantees every *released* integration has an entry.
 */
module.exports = function integrationChangelogsPlugin(context) {
  const siteDir = context.siteDir;
  const changelogDir = path.join(siteDir, 'src', 'pages', 'changelog', 'integrations');
  const integrationsJson = path.join(siteDir, 'src', 'data', 'integrations.json');

  return {
    name: 'integration-changelogs',

    getPathsToWatch() {
      return [path.join(changelogDir, '*.md'), integrationsJson];
    },

    async loadContent() {
      const {integrations} = JSON.parse(fs.readFileSync(integrationsJson, 'utf8'));

      const bySlug = new Map();
      for (const entry of integrations) {
        if (entry.link.startsWith('/sdks/integrations/')) {
          bySlug.set(entry.link.replace('/sdks/integrations/', ''), entry);
        }
      }

      const files = fs
        .readdirSync(changelogDir)
        .filter((f) => f.endsWith('.md') && f !== 'index.md');

      const entries = [];
      for (const file of files) {
        const slug = file.slice(0, -'.md'.length);
        const meta = bySlug.get(slug);
        if (!meta) continue;

        const body = fs.readFileSync(path.join(changelogDir, file), 'utf8');
        // The release script prepends `## [x.y.z](tag url)`, newest first.
        const latest = body.match(/^## \[([^\]]+)\]/m);
        const releaseCount = (body.match(/^## \[/gm) || []).length;

        entries.push({
          slug,
          id: meta.id,
          name: meta.name,
          description: meta.description,
          icon: meta.icon,
          category: meta.category,
          docLink: meta.link,
          latestVersion: latest ? latest[1] : null,
          releaseCount,
        });
      }

      entries.sort((a, b) => a.name.localeCompare(b.name));
      return entries;
    },

    async contentLoaded({content, actions}) {
      actions.setGlobalData({entries: content});
    },
  };
};
