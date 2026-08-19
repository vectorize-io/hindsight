import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const umamiUrl = process.env.UMAMI_URL;
const umamiWebsiteId = process.env.UMAMI_WEBSITE_ID;

// Announcement bar - supports HTML for links
// Set to empty string '' to hide the bar
const ANNOUNCEMENT_BAR = 'Hindsight is State-of-the-Art on Memory for AI Agents <a href="https://arxiv.org/abs/2512.12818" target="_blank">Read the paper →</a>';

const config: Config = {
  title: 'Hindsight',
  tagline: 'Hindsight: Agent Memory That Works Like Human Memory',
  favicon: 'img/favicon.png',

  future: {
    v4: true,
    // Docusaurus Faster, opted in one flag at a time. The full `true` preset
    // does not work here: both the SWC JS minifier and the SSG worker threads
    // crash while rendering /api-reference (Redoc needs a `Prism` global that
    // neither setup provides). Everything else is safe and roughly halves the
    // cold build.
    experimental_faster: {
      swcJsLoader: true,
      swcJsMinimizer: false, // breaks Redoc SSG: "ReferenceError: Prism is not defined"
      swcHtmlMinimizer: false,
      lightningCssMinimizer: true,
      mdxCrossCompilerCache: true,
      rspackBundler: true,
      rspackPersistentCache: true,
      ssgWorkerThreads: false, // same Redoc SSG failure on /api-reference
    },
  },

  markdown: {
    mermaid: true,
  },

  url: 'https://hindsight.vectorize.io',
  baseUrl: '/',

  organizationName: 'vectorize-io',
  projectName: 'hindsight',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://fonts.googleapis.com',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://fonts.gstatic.com',
        crossorigin: 'anonymous',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'stylesheet',
        href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap',
        media: 'print',
        onload: "this.media='all'",
      },
    },
  ],

  scripts: [
    ...(umamiUrl && umamiWebsiteId
      ? [
          {
            src: `${umamiUrl}/script.js`,
            async: true,
            defer: true,
            'data-website-id': umamiWebsiteId,
          },
        ]
      : []),
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          // The sidebar already shows where a page sits, and every doc opens
          // with its own H1 — the trail only repeated both.
          // src/theme/DocBreadcrumbs still renders: the original component
          // returns null on this flag, and the wrapper carries the skill banner.
          breadcrumbs: false,
          // Whether to include the "current" (Next / unreleased) docs version.
          //
          // Controlled by the single explicit env var INCLUDE_CURRENT_VERSION.
          // `start-docs.sh` sets it to "true" so local dev always sees Next;
          // production builds leave it unset so only released versions ship.
          //
          // We deliberately do NOT sniff NODE_ENV here — it's unreliable
          // across Docusaurus hot-reload paths and used to cause the Next
          // version to disappear intermittently when editing files.
          onlyIncludeVersions: (() => {
            const includeCurrent = process.env.INCLUDE_CURRENT_VERSION === 'true';
            let released: string[] = [];
            try {
              released = require('./versions.json') as string[];
            } catch {
              // No versions.json yet — nothing has been released.
              return undefined;
            }
            return includeCurrent ? ['current', ...released] : released;
          })(),
          // Disable version badges on all versions
          versions: (() => {
            const config: Record<string, {badge: boolean}> = {
              current: {badge: false},
            };
            try {
              const versions = require('./versions.json') as string[];
              versions.forEach((v: string) => {
                config[v] = {badge: false};
              });
            } catch {
              // No versions yet
            }
            return config;
          })(),
        },
        blog: {
          showReadingTime: true,
          blogTitle: 'Hindsight Blog',
          blogDescription: 'Updates, insights, and deep dives into agent memory',
          postsPerPage: 'ALL',
          blogSidebarCount: 0,
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
    [
      'redocusaurus',
      {
        specs: [
          {
            id: 'hindsight-api',
            spec: 'static/openapi.json',
            route: '/api-reference',
            url: '/openapi.json',
          },
        ],
        theme: {
          primaryColor: '#0074d9',
          sidebar: {
            backgroundColor: '#09090b',
          },
          rightPanel: {
            backgroundColor: '#18181b',
          },
          typography: {
            fontSize: '15px',
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            headings: {
              fontFamily: "'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            },
            code: {
              fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', Monaco, Consolas, monospace",
              fontSize: '13px',
            },
          },
        },
        config: {
          scrollYOffset: 60,
          nativeScrollbars: true,
          expandSingleSchemaField: true,
          expandResponses: '200,201',
        },
      },
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'integrations',
        path: './docs-integrations',
        routeBasePath: 'sdks/integrations',
        breadcrumbs: false, // as above — its own docs plugin, its own flag
        // Unversioned plugin: gives the integration pages a sidebar generated
        // from src/data/integrations.json without versioning them.
        sidebarPath: './sidebars-integrations.ts',
      },
    ],
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'guides',
        routeBasePath: 'guides',
        path: './guides',
        showReadingTime: true,
        postsPerPage: 'ALL',
        blogSidebarCount: 0,
        blogTitle: 'Guides',
        blogDescription: 'In-depth guides for AI memory and agent development',
        feedOptions: {type: []},
        onUntruncatedBlogPosts: 'ignore',
      },
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        docsRouteBasePath: '/',
        indexBlog: true,
        blogRouteBasePath: '/blog',
        highlightSearchTermsOnTargetPage: false,
        // The search field lives in the docs sidebar (see
        // src/theme/DocSidebar/Desktop/Content), so its results panel opens to
        // the right of the field. Autodetection would infer "right" from the
        // navbar and push the panel off the left of the window.
        searchBarPosition: 'left',
      },
    ],
  ],

  themeConfig: {
    ...(ANNOUNCEMENT_BAR && {
      announcementBar: {
        id: 'announcement',
        content: ANNOUNCEMENT_BAR,
        backgroundColor: '#0074d9',
        textColor: '#ffffff',
        isCloseable: false,
      },
    }),
    image: 'img/logo.png',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      logo: {
        alt: 'Hindsight Logo',
        src: 'img/logo.png',
        style: { height: '32px' },
      },
      items: [
        {
          type: 'doc',
          docId: 'developer/index',
          position: 'left',
          label: 'Docs',
          className: 'navbar-item-developer',
        },
        // Sits directly after "Docs": the version only ever qualifies the docs,
        // so it reads as part of that item rather than as a sixth destination.
        {
          type: 'docsVersionDropdown',
          position: 'left',
          className: 'navbar-item-version',
        },
        {
          to: '/integrations',
          position: 'left',
          label: 'Integrations',
          className: 'navbar-item-integrations',
        },
        {
          to: '/changelog',
          position: 'left',
          label: 'Changelog',
          className: 'navbar-item-changelog',
        },
        {
          href: 'https://learn.hindsight.vectorize.io',
          position: 'left',
          label: 'Academy',
          className: 'navbar-item-academy',
        },
        {
          type: 'dropdown',
          label: 'Resources',
          position: 'left',
          className: 'navbar-item-resources',
          items: [
            {
              to: '/templates',
              label: 'Bank Templates Hub',
              customProps: { icon: 'lu-layout-template' },
            },
            {
              to: '/best-practices',
              label: 'Best Practices',
              customProps: { icon: 'lu-star' },
            },
            {
              to: '/faq',
              label: 'FAQ',
              customProps: { icon: 'lu-circle-help' },
            },
            {
              to: '/cookbook',
              label: 'Cookbook',
              customProps: { icon: 'lu-book' },
            },
            {
              to: '/blog',
              label: 'Blog',
              customProps: { icon: 'lu-rss' },
            },
            {
              to: '/api-reference',
              label: 'API Reference',
              customProps: { icon: 'lu-book-open' },
            },
            {
              href: 'https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg',
              label: 'Community',
              customProps: { icon: 'si-slack' },
            },
            {
              href: 'https://benchmarks.hindsight.vectorize.io/',
              label: 'Benchmarks',
              customProps: { icon: 'lu-chart-bar' },
            },
            {
              href: 'https://benchmarks.hindsight.vectorize.io/',
              label: 'Which Model Should I Use?',
              customProps: { icon: 'lu-cpu' },
            },
            {
              href: 'https://arxiv.org/abs/2512.12818',
              label: 'Paper',
              customProps: { icon: 'lu-file-text' },
            },
          ],
        },
        {
          href: 'https://github.com/vectorize-io/hindsight',
          position: 'right',
          label: 'GitHub',
          className: 'header-github-link',
        },
        // Rendered last on the right (after the color-mode toggle) by
        // src/theme/Navbar/Content — the only call to action up there.
        {
          href: 'https://ui.hindsight.vectorize.io/signup',
          position: 'right',
          label: 'Sign up',
          className: 'navbar-item-signup',
        },
      ],
    },
    footer: {
      // Not 'dark': that paints a slab in the theme's dark palette regardless
      // of the active theme, so on a light page the footer arrived as a black
      // block. It now continues the page and is separated by a rule alone
      // (custom.css).
      style: 'light',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Introduction',
              to: '/',
            },
            {
              label: 'Developer Guide',
              to: '/developer/installation',
            },
            {
              label: 'Clients & Integrations',
              to: '/sdks/python',
            },
            {
              label: 'API Reference',
              to: '/api-reference/',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Cookbook',
              to: '/cookbook',
            },
            {
              label: 'Changelog',
              to: '/changelog',
            },
            {
              label: 'Guides',
              to: '/guides',
            },
            {
              label: 'Academy',
              href: 'https://learn.hindsight.vectorize.io',
            },
            {
              label: 'Hindsight Cloud',
              href: 'https://ui.hindsight.vectorize.io/signup',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/vectorize-io/hindsight',
            },
            {
              label: 'Slack',
              href: 'https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg',
            },
            {
              label: 'Vectorize',
              href: 'https://vectorize.io',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} <a href="https://vectorize.io">Vectorize, Inc.</a>`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'python', 'rust'],
    },
    mermaid: {
      theme: {
        light: 'base',
        dark: 'base',
      },
      options: {
        themeVariables: {
          // Gradient start (#0074d9 blue) for nodes
          primaryColor: '#0074d9',
          primaryTextColor: '#ffffff',
          primaryBorderColor: '#005db0',
          // Gradient end (#009296 teal) for edges/clusters
          secondaryColor: '#009296',
          secondaryTextColor: '#ffffff',
          secondaryBorderColor: '#007a7d',
          // Tertiary
          tertiaryColor: '#e6f7f8',
          tertiaryTextColor: '#1e293b',
          // Lines and edges - gradient end color
          lineColor: '#009296',
          // Text
          textColor: '#1e293b',
          // Node specific - gradient start
          nodeBkg: '#0074d9',
          nodeTextColor: '#ffffff',
          nodeBorder: '#005db0',
          // Main background
          mainBkg: '#0074d9',
          // Clusters/subgraphs - gradient end
          clusterBkg: 'rgba(0, 146, 150, 0.08)',
          clusterBorder: '#009296',
          // Labels
          edgeLabelBackground: 'transparent',
          labelBackground: 'transparent',
          // Font - Inter to match body text
          fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        },
      },
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
