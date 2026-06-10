import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  developerSidebar: [
    {
      type: 'category',
      label: 'Architecture',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'developer/index',
          label: 'Overview',
          customProps: { icon: 'lu-book' },
        },
        {
          type: 'doc',
          id: 'developer/retain',
          label: 'Retain',
          customProps: { icon: 'lu-brain' },
        },
        {
          type: 'doc',
          id: 'developer/retrieval',
          label: 'Recall',
          customProps: { icon: 'lu-search' },
        },
        {
          type: 'doc',
          id: 'developer/reflect',
          label: 'Reflect',
          customProps: { icon: 'lu-message' },
        },
        {
          type: 'doc',
          id: 'developer/observations',
          label: 'Observations',
          customProps: { icon: 'lu-activity' },
        },
        {
          type: 'doc',
          id: 'developer/multilingual',
          label: 'Multilingual',
          customProps: { icon: 'lu-languages' },
        },
        {
          type: 'doc',
          id: 'developer/performance',
          label: 'Performance',
          customProps: { icon: 'lu-zap' },
        },
        {
          type: 'doc',
          id: 'developer/storage',
          label: 'Storage',
          customProps: { icon: 'lu-database' },
        },
        {
          type: 'doc',
          id: 'developer/rag-vs-hindsight',
          label: 'RAG vs Memory',
          customProps: { icon: 'lu-compare' },
        },
      ],
    },
    {
      type: 'category',
      label: 'API',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'developer/api/quickstart',
          label: 'Quick Start',
          customProps: { icon: 'lu-rocket' },
        },
        {
          type: 'doc',
          id: 'developer/api/retain',
          label: 'Retain',
          customProps: { icon: 'lu-brain' },
        },
        {
          type: 'doc',
          id: 'developer/api/recall',
          label: 'Recall',
          customProps: { icon: 'lu-search' },
        },
        {
          type: 'doc',
          id: 'developer/api/reflect',
          label: 'Reflect',
          customProps: { icon: 'lu-message' },
        },
        {
          type: 'doc',
          id: 'developer/api/mental-models',
          label: 'Mental Models',
          customProps: { icon: 'lu-layers' },
        },
        {
          type: 'doc',
          id: 'developer/api/memory-banks',
          label: 'Memory Banks',
          customProps: { icon: 'lu-memory' },
        },
        {
          type: 'doc',
          id: 'developer/api/documents',
          label: 'Documents',
          customProps: { icon: 'lu-file' },
        },
        {
          type: 'doc',
          id: 'developer/api/operations',
          label: 'Operations',
          customProps: { icon: 'lu-cpu' },
        },
        {
          type: 'doc',
          id: 'developer/api/webhooks',
          label: 'Webhooks',
          customProps: { icon: 'lu-webhook' },
        },
        {
          type: 'doc',
          id: 'developer/api/bank-templates',
          label: 'Bank Templates',
          customProps: { icon: 'lu-file-json' },
        },
        {
          type: 'link',
          href: '/api-reference',
          label: 'API Reference',
          customProps: { icon: 'lu-book-open', iconAfter: 'lu-arrow-up-right' },
        },
      ],
    },
    {
      type: 'category',
      label: 'Security',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'developer/memory-defense/index',
          label: 'Memory Defense',
          customProps: { icon: 'lu-shield' },
        },
      ],
    },
    {
      type: 'category',
      label: 'Clients',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'sdks/python',
          label: 'Python',
          customProps: { icon: 'si-python' },
        },
        {
          type: 'doc',
          id: 'sdks/nodejs',
          label: 'TypeScript',
          customProps: { icon: '/img/icons/typescript.png' },
        },
        {
          type: 'doc',
          id: 'sdks/go',
          label: 'Go',
          customProps: { icon: 'si-go' },
        },
        {
          type: 'doc',
          id: 'sdks/cli',
          label: 'CLI',
          customProps: { icon: 'lu-terminal' },
        },
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      collapsible: false,
      // Positional placeholder. The real items are injected at render time from
      // src/data/integrations.json by the swizzled DocPage/Layout/Sidebar
      // component — single source of truth, shared across all docs versions.
      // (Docusaurus rejects an empty category, so we seed one valid link to the
      // gallery; the component replaces these items.)
      items: [{type: 'link', href: '/integrations', label: 'Browse all integrations'}],
    },
    {
      type: 'category',
      label: 'Hosting',
      collapsible: false,
      items: [
        {
          type: 'link',
          href: 'https://ui.hindsight.vectorize.io/signup',
          label: 'Cloud',
          customProps: { icon: 'lu-cloud', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'doc',
          id: 'developer/installation',
          label: 'Installation',
          customProps: { icon: 'lu-package' },
        },
        {
          type: 'doc',
          id: 'developer/services',
          label: 'Services',
          customProps: { icon: 'lu-server' },
        },
        {
          type: 'doc',
          id: 'developer/configuration',
          label: 'Configuration',
          customProps: { icon: 'lu-settings' },
        },
        {
          type: 'doc',
          id: 'developer/admin-cli',
          label: 'Admin CLI',
          customProps: { icon: 'lu-terminal' },
        },
        {
          type: 'doc',
          id: 'developer/extensions',
          label: 'Extensions',
          customProps: { icon: 'lu-plug' },
        },
        {
          type: 'doc',
          id: 'developer/models',
          label: 'Models',
          customProps: { icon: 'lu-cpu' },
        },
        {
          type: 'doc',
          id: 'developer/monitoring',
          label: 'Monitoring',
          customProps: { icon: 'lu-activity' },
        },
        {
          type: 'doc',
          id: 'developer/mcp-server',
          label: 'MCP Server',
          customProps: { icon: 'lu-network' },
        },
      ],
    },
    {
      type: 'category',
      label: 'Installation',
      collapsible: false,
      items: [
        {
          type: 'link',
          href: '/developer/installation#docker',
          label: 'Docker',
          customProps: { icon: 'si-docker', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: '/developer/installation#helm--kubernetes',
          label: 'Kubernetes',
          customProps: { icon: 'si-kubernetes', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: '/developer/installation#bare-metal-pip',
          label: 'Bare Metal',
          customProps: { icon: 'lu-hard-drive', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'doc',
          id: 'sdks/hindsight-all',
          label: 'Programmatic API (Python)',
          customProps: { icon: 'si-python' },
        },
        {
          type: 'doc',
          id: 'sdks/hindsight-all-npm',
          label: 'Programmatic API (Node.js)',
          customProps: { icon: 'si-nodedotjs' },
        },
        {
          type: 'doc',
          id: 'sdks/embed',
          label: 'Daemon CLI',
          customProps: { icon: 'lu-terminal' },
        },
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      collapsible: false,
      items: [
        {
          type: 'link',
          href: '/templates',
          label: 'Bank Templates Hub',
          customProps: { icon: 'lu-layout-template', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: '/best-practices',
          label: 'Best Practices',
          customProps: { icon: 'lu-star', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: '/faq',
          label: 'FAQ',
          customProps: { icon: 'lu-circle-help', iconAfter: 'lu-arrow-up-right' },
        },
      ],
    },
    {
      type: 'category',
      label: 'More',
      collapsible: false,
      items: [
        {
          type: 'link',
          href: '/cookbook',
          label: 'Cookbook',
          customProps: { icon: 'lu-book', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: '/blog',
          label: 'Blog',
          customProps: { icon: 'lu-rss', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: 'https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg',
          label: 'Community',
          customProps: { icon: 'si-slack', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: 'https://github.com/vectorize-io/hindsight',
          label: 'GitHub',
          customProps: { icon: 'si-github', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: 'https://benchmarks.hindsight.vectorize.io/',
          label: 'Benchmarks',
          customProps: { icon: 'lu-chart-bar', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: 'https://benchmarks.hindsight.vectorize.io/',
          label: 'Which Model Should I Use?',
          customProps: { icon: 'lu-cpu', iconAfter: 'lu-arrow-up-right' },
        },
        {
          type: 'link',
          href: 'https://arxiv.org/abs/2512.12818',
          label: 'Paper',
          customProps: { icon: 'lu-file-text', iconAfter: 'lu-arrow-up-right' },
        },
      ],
    },
  ],
};

export default sidebars;
