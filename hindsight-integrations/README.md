# Hindsight Integrations

First-party integrations that give popular AI agents, frameworks, and CLIs persistent long-term memory via [Hindsight](https://vectorize.io/hindsight).

Each integration lives in its own subdirectory with its own README, configuration, and tests. Pick the one that matches the agent or framework you're using.

## Coding agents & CLIs

| Integration | What it does | Install |
| --- | --- | --- |
| [**Claude Code**](./claude-code) | Hooks-based memory for Anthropic's Claude Code. Auto-retains every session, recalls context on each prompt. | `npx hindsight-cc` |
| [**OpenCode**](./opencode) | TypeScript plugin with retain/recall/reflect tools, auto-retain on idle, memory injection on session start, compaction preservation. | Add `@vectorize-io/opencode-hindsight` to `opencode.json` |
| [**Codex CLI**](./codex) | Python hook scripts for OpenAI's Codex CLI. Auto-recall on `UserPromptSubmit`, auto-retain on `Stop`. | `curl -fsSL https://hindsight.vectorize.io/get-codex \| bash` |
| [**Cursor CLI**](./cursor-cli) | Python hook scripts for Cursor CLI. Auto-recall on `beforeSubmitPrompt`, auto-retain on `stop`, final flush on `sessionEnd`. | `./scripts/install.sh` |
| [**Continue.dev**](./continue) | HTTP context provider for precise `@hindsight` recall in chat, plus optional MCP-server + rules for automatic recall/retain in agent mode. | `pip install hindsight-continue` |
| [**Roo Code**](./roo-code) | Persistent memory for Roo Code VS Code extension. | See README |
| [**Hermes (OpenAI Agents SDK)**](./hermes) | Memory layer for OpenAI Agents SDK. | See README |
| [**Grok Build**](./grok-build) | Hooks for Grok Build (xAI). | See README |
| [**Claude Code Skills**](./chat) | Skills integration for Claude Code agents. | See README |

## Agent frameworks

| Integration | What it does |
| --- | --- |
| [**LiteLLM**](./litellm) | Proxy callbacks — every model proxied through LiteLLM gets memory. Zero code changes. |
| [**CrewAI**](./crewai) | Long-term memory tools for CrewAI agents. |
| [**Pydantic AI**](./pydantic-ai) | Dependency-injected memory for Pydantic AI agents. |
| [**Vercel AI SDK**](./ai-sdk) | Persistent memory for Vercel AI SDK apps. |
| [**Vercel Chat**](./chat) | Drop-in memory for the Vercel AI Chatbot. |
| [**LangGraph / LangChain**](./langgraph) | Memory Tools, Graph Nodes, and BaseStore adapter patterns. |
| [**LlamaIndex**](./llamaindex) | Agent-driven (BaseToolSpec) and automatic (BaseMemory) memory. |
| [**Google ADK**](./google-adk) | `BaseMemoryService` implementation for ADK. |
| [**Strands Agents**](./strands) | Retain/recall/reflect tools for Strands. |
| [**AG2**](./ag2) | Cross-conversation memory tools. |
| [**AutoGen**](./autogen) | `FunctionTool` instances for retain/recall/reflect. |
| [**OpenAI Agents SDK**](./openai-agents) | `FunctionTool`-based memory. |
| [**OpenHands**](./openhands) | Native MCP server config + recall/retain rule for OpenHands (formerly OpenDevin). |
| [**NemoClaw**](./nemoclaw) | One-command setup for NemoClaw sandboxes. |
| [**Right Agent**](./right-agent) | Native memory provider for Right Agent sandboxes. |
| [**Pipecat**](./pipecat) | Memory nodes for Pipecat voice pipelines. |
| [**AgentCore**](./agentcore) | Memory tools for AgentCore agents. |
| [**Smolagents**](./smolagents) | Retain/recall/reflect tools for HuggingFace Smolagents. |
| [**OpenClaw**](./openclaw) | Memory for OpenClaw workflows. |

## Workflow & automation platforms

| Integration | What it does |
| --- | --- |
| [**n8n**](./n8n) | Community node — drop retain/recall/reflect into any n8n workflow. |
| [**Zapier**](./zapier) | Zapier app — retain/recall/reflect actions plus instant memory-event triggers. |
| [**Dify**](./dify) | Persistent memory for Dify apps. |
| [**Flowise**](./flowise) | Memory nodes for Flowise flows. |
| [**Vapi**](./vapi) | Persistent memory for Vapi voice agents. |
| [**Gemini Spark**](./gemini-spark) | Memory for Gemini Spark. |
| [**Roo Code**](./roo-code) | Memory for the Roo Code VS Code extension. |
| [**Google ADK**](./google-adk) | `BaseMemoryService` for Google ADK. |
| [**Cloudflare OAuth Proxy**](./cloudflare-oauth-proxy) | OAuth proxy so you can run Hindsight behind Cloudflare Access. |
| [**Paperclip**](./paperclip) | Memory layer for Paperclip workflows. |

## Adding a new integration

See [CLAUDE.md → "Adding New Integrations"](../CLAUDE.md) for the checklist (tests, CI job, release registration, code standards).

## Releases

Each integration is versioned and released independently:

```bash
./scripts/release-integration.sh <integration-name> <version>
# e.g.
./scripts/release-integration.sh cursor-cli patch
```

The release script reads the version from the integration's `settings.json` / `package.json` / `pyproject.toml`, generates a changelog entry, and tags `integrations/<name>/v<version>`.
