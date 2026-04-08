import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const umamiUrl = process.env.UMAMI_URL;
const umamiWebsiteId = process.env.UMAMI_WEBSITE_ID;

// Announcement bar - supports HTML for links
// Set to empty string '' to hide the bar
const ANNOUNCEMENT_BAR = 'Hindsight is State-of-the-Art on Memory for AI Agents | <a href="https://arxiv.org/abs/2512.12818" target="_blank">Read the paper →</a>';

const config: Config = {
  title: 'Hindsight',
  tagline: 'Hindsight: Agent Memory That Works Like Human Memory',
  favicon: 'img/favicon.png',

  future: {
    v4: true,
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
          // Only show "next" version in development or when INCLUDE_CURRENT_VERSION=true
          // In production, only show released versions from versions.json
          onlyIncludeVersions: (() => {
            const isDev = process.env.NODE_ENV === 'development' || process.env.INCLUDE_CURRENT_VERSION === 'true';
            try {
              const versions = require('./versions.json') as string[];
              // In dev mode, explicitly include 'current' (Next) + all released versions
              // In production, only show released versions
              return isDev ? ['current', ...versions] : versions;
            } catch {
              return undefined; // No versions yet, show current
            }
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
        sidebarPath: false,
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
          label: 'Developer',
          className: 'navbar-item-developer',
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
          type: 'docsVersionDropdown',
          position: 'right',
          className: 'navbar-item-version',
        },
        {
          href: 'https://ui.hindsight.vectorize.io/signup',
          position: 'right',
          label: 'Cloud',
          className: 'navbar-item-cloud',
        },
        {
          href: 'https://github.com/vectorize-io/hindsight',
          position: 'right',
          label: 'GitHub',
          className: 'header-github-link',
        },
      ],
    },
    footer: {
      style: 'dark',
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
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Vectorize, Inc.`,
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
