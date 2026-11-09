/**
 * Static redirects for docs pages that were renamed.
 *
 * Old links (search results, bookmarks, other sites) must keep working after a
 * page moves. For every built route ending in a new path, this writes a small
 * HTML page at the old path that forwards to it, under the same version prefix
 * (/developer/pages, /next/developer/pages, ...). A redirect is only written
 * where the old path is not still a real page, so older doc versions that
 * still have the page under its old name are left alone.
 */
const fs = require('fs');
const path = require('path');

// new path suffix -> old path suffix
const RENAMED = {
  '/developer/pages': '/developer/knowledge-pages',
  '/developer/api/pages': '/developer/api/knowledge-pages',
};

module.exports = function renamedDocsPlugin() {
  return {
    name: 'renamed-docs',

    async postBuild({outDir, routesPaths}) {
      const routes = new Set(routesPaths);
      for (const route of routesPaths) {
        for (const [to, from] of Object.entries(RENAMED)) {
          if (!route.endsWith(to)) continue;
          const oldRoute = route.slice(0, -to.length) + from;
          if (routes.has(oldRoute)) continue;
          const html = `<!doctype html><html><head><meta charset="utf-8"><title>Redirecting…</title><link rel="canonical" href="${route}"><meta http-equiv="refresh" content="0; url=${route}"><script>location.replace(${JSON.stringify(route)} + location.hash)</script></head><body><a href="${route}">${route}</a></body></html>`;
          // trailingSlash: false -> pages are written as <route>.html
          const file = path.join(outDir, `${oldRoute}.html`);
          fs.mkdirSync(path.dirname(file), {recursive: true});
          fs.writeFileSync(file, html);
        }
      }
    },
  };
};
