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
        },
        {
          type: 'doc',
          id: 'developer/retain',
          label: 'Retain',
        },
        {
          type: 'doc',
          id: 'developer/retrieval',
          label: 'Recall',
        },
        {
          type: 'doc',
          id: 'developer/reflect',
          label: 'Reflect',
        },
        {
          type: 'doc',
          id: 'developer/observations',
          label: 'Observations',
        },
        {
          type: 'doc',
          id: 'developer/multilingual',
          label: 'Multilingual',
        },
        {
          type: 'doc',
          id: 'developer/performance',
          label: 'Performance',
        },
        {
          type: 'doc',
          id: 'developer/storage',
          label: 'Storage',
        },
        {
          type: 'doc',
          id: 'developer/rag-vs-hindsight',
          label: 'RAG vs Memory',
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
        },
        {
          type: 'doc',
          id: 'developer/api/retain',
          label: 'Retain',
        },
        {
          type: 'doc',
          id: 'developer/api/recall',
          label: 'Recall',
        },
        {
          type: 'doc',
          id: 'developer/api/reflect',
          label: 'Reflect',
        },
        {
          type: 'doc',
          id: 'developer/api/mental-models',
          label: 'Mental Models',
        },
        {
          type: 'doc',
          id: 'developer/api/memory-banks',
          label: 'Memory Banks',
        },
        {
          type: 'doc',
          id: 'developer/api/documents',
          label: 'Documents',
        },
        {
          type: 'doc',
          id: 'developer/api/operations',
          label: 'Operations',
        },
      ],
    },
    {
      type: 'category',
      label: 'Hosting',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'developer/installation',
          label: 'Installation',
        },
        {
          type: 'doc',
          id: 'developer/services',
          label: 'Services',
        },
        {
          type: 'doc',
          id: 'developer/configuration',
          label: 'Configuration',
        },
        {
          type: 'doc',
          id: 'developer/admin-cli',
          label: 'Admin CLI',
        },
        {
          type: 'doc',
          id: 'developer/extensions',
          label: 'Extensions',
        },
        {
          type: 'doc',
          id: 'developer/models',
          label: 'Models',
        },
        {
          type: 'doc',
          id: 'developer/monitoring',
          label: 'Monitoring',
        },
        {
          type: 'doc',
          id: 'developer/mcp-server',
          label: 'MCP Server',
        },
      ],
    },
  ],
  sdksSidebar: [
    {
      type: 'category',
      label: 'Clients',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'sdks/python',
          label: 'Python',
        },
        {
          type: 'doc',
          id: 'sdks/nodejs',
          label: 'Node.js',
        },
        {
          type: 'doc',
          id: 'sdks/cli',
          label: 'CLI',
        },
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'sdks/integrations/local-mcp',
          label: 'Local MCP Server',
        },
        {
          type: 'doc',
          id: 'sdks/integrations/litellm',
          label: 'LiteLLM',
        },
        {
          type: 'doc',
          id: 'sdks/integrations/skills',
          label: 'Skills',
        },
      ],
    },
  ],
  cookbookSidebar: [
    {
      type: 'doc',
      id: 'cookbook/index',
      label: 'Overview',
    },
    {
      type: 'category',
      label: 'Recipes',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'cookbook/recipes/quickstart',
          label: 'Hindsight Quickstart',
        },
        {
          type: 'doc',
          id: 'cookbook/recipes/per-user-memory',
          label: 'Per-User Memory',
        },
        {
          type: 'doc',
          id: 'cookbook/recipes/support-agent-shared-knowledge',
          label: 'Support Agent with Shared Knowledge',
        },
        {
          type: 'doc',
          id: 'cookbook/recipes/litellm-memory-demo',
          label: 'Memory with LiteLLM',
        },
        {
          type: 'doc',
          id: 'cookbook/recipes/tool-learning-demo',
          label: 'Routing Tool Learning',
        }
      ],
    },
    {
      type: 'category',
      label: 'Applications',
      collapsible: false,
      items: [
        {
          type: 'doc',
          id: 'cookbook/applications/openai-fitness-coach',
          label: 'OpenAI Agent + Hindsight Memory Integration',
        }
      ],
    },
  ],
};

export default sidebars;
