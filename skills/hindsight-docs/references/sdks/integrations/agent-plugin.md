
# Agent Plugins

Portable long-term memory for any [Agent Plugins](https://agent-plugins.org) client, powered by Hindsight.

Agent Plugins is the vendor-neutral open standard for packaging Agent Skills + MCP servers into a single distributable plugin. Hindsight ships one plugin that every compatible client can load.

## Quick Start

1. Get your API key from the Hindsight Cloud dashboard.
2. Set `HINDSIGHT_API_KEY` and optionally `HINDSIGHT_BANK_ID`.
3. Install the plugin in your client through its plugin/MCP UI.

Once installed, agents can call `recall`, `retain`, and `reflect` automatically through the bundled skill.
