# Changelog

All notable changes to the Hindsight Flowise integration.

## 0.1.0

- Initial scaffold with three Tool nodes (Hindsight Retain / Recall / Reflect) and a shared `hindsightApi` credential.
- Each node returns a LangChain `DynamicStructuredTool` from `init()`, so it slots into Flowise tool sockets and any LangChain agent.
- Built against `@vectorize-io/hindsight-client` ^0.5.6.
- Source files use upstream-relative imports (`'../../../src/Interface'`, `'../src/Interface'`) and copy 1:1 into `FlowiseAI/Flowise/packages/components/`.
