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
        {
          type: 'doc',
          id: 'sdks/embed',
          label: 'Embedded Python',
          customProps: { icon: '/img/icons/package.svg' },
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
          customProps: { icon: '/img/icons/mcp.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/litellm',
          label: 'LiteLLM',
          customProps: { icon: '/img/icons/litellm.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/openclaw',
          label: 'OpenClaw',
          customProps: { icon: '/img/icons/openclaw.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/ai-sdk',
          label: 'Vercel AI SDK',
          customProps: { icon: '/img/icons/vercel.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/chat',
          label: 'Vercel Chat SDK',
          customProps: { icon: '/img/icons/vercel.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/crewai',
          label: 'CrewAI',
          customProps: { icon: '/img/icons/crewai.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/pydantic-ai',
          label: 'Pydantic AI',
          customProps: { icon: '/img/icons/pydanticai.png' },
        },
        {
          type: 'doc',
          id: 'sdks/integrations/skills',
          label: 'Skills',
          customProps: { icon: '/img/icons/skills.png' },
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
  ],
};

export default sidebars;
