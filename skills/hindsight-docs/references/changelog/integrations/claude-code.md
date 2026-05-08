---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="Claude Code Changelog" subtitle="hindsight-memory — Hindsight memory plugin for Claude Code." />

← Claude Code integration

## [0.6.4](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.6.4)

**Bug Fixes**

- Fixed Claude Code recall parameter naming to avoid failures when limiting results.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/offendingcommit" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@offendingcommit</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/909a4fd4" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>909a4fd4</a>
- Improved page retrieval reliability by returning full page content and handling oversized tool results.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/cdbartholomew" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@cdbartholomew</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b2a693ab" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b2a693ab</a>
- Corrected the agent knowledge page list tool to fetch metadata-only details, preventing incorrect or overly large responses.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/cdbartholomew" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@cdbartholomew</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/6c6ee73c" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>6c6ee73c</a>

## [0.6.3](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.6.3)

**Features**

- Adds explicit directory-to-bank mapping and better support for Git worktrees when using the claude-code integration.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/691c65ac" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>691c65ac</a>

## [0.6.2](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.6.2)

**Bug Fixes**

- Fixes Claude Code integration setup by bootstrapping required Python dependencies in an isolated virtual environment under the plugin data directory.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/012c100e" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>012c100e</a>

## [0.6.1](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.6.1)

**Features**

- Claude Code’s create-agent skill now supports and understands SDA project layouts when generating agents.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/0231094d" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>0231094d</a>

## [0.6.0](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.6.0)

**Features**

- Added Claude Code knowledge tools backed by a Python MCP server, enabling the integration to read/write and use Hindsight knowledge more directly.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/6c55dbde" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>6c55dbde</a>
- Introduced subagents and a new “create-agent” skill to help users create and manage agent configurations/workflows within Claude Code.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/1b9d6d91" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>1b9d6d91</a>

## [0.4.0](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.4.0)

**Features**

- Allows recalling memories from additional banks alongside the primary bank for broader context retrieval.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/starbit-biostar" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@starbit-biostar</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/cba2b0d8" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>cba2b0d8</a>

**Improvements**

- Adds a \{user_id\} template variable for retainTags to better organize memories by user.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/soichisumi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@soichisumi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/9181c9a2" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>9181c9a2</a>

**Bug Fixes**

- Fixes transcript parsing when tool results include list-style content, preventing recall/retain from failing on some sessions.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b67b6886" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b67b6886</a>
- Ensures dynamically generated memory bank IDs preserve raw UTF-8, avoiding incorrect bank selection for non-ASCII identifiers.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/Desko77" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@Desko77</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/08a75b5b" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>08a75b5b</a>
- Prevents memory compaction from overwriting memories that were meant to be retained.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/9c9a5a29" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>9c9a5a29</a>
- Ensures a session’s final memories are retained on SessionEnd even when retainEveryNTurns is greater than 1.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/starbit-biostar" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@starbit-biostar</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/aefc1ebc" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>aefc1ebc</a>
- Reads transcript files as UTF-8 to prevent errors or corrupted text when transcripts contain non-ASCII characters.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/tordf" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@tordf</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/eb9be903" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>eb9be903</a>

## [0.3.1](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.3.1)

**Bug Fixes**

- All Claude Code integration HTTP requests now include an identifying User-Agent for better compatibility and observability. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

## [0.3.0](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.3.0)

**Features**

- Claude Code integration now retains tool calls as structured JSON for more accurate memory and retrieval. ([`8cb8b912`](https://github.com/vectorize-io/hindsight/commit/8cb8b912))

## [0.2.0](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.2.0)

**Features**

- Added a Claude Code integration plugin for capturing and using Hindsight memory in Claude Code. ([`f4390bdc`](https://github.com/vectorize-io/hindsight/commit/f4390bdc))
- Claude Code integration can retain full sessions with document upsert and configurable tagging. ([`2d31b67d`](https://github.com/vectorize-io/hindsight/commit/2d31b67d))

**Improvements**

- Improved Claude Code plugin installation and configuration experience. ([`35b2cbb6`](https://github.com/vectorize-io/hindsight/commit/35b2cbb6))
- Integrations no longer rely on hardcoded default models, allowing model selection to be fully configured. ([`58e68f3e`](https://github.com/vectorize-io/hindsight/commit/58e68f3e))
- Claude Code now starts the Hindsight background daemon automatically at session start for smoother operation. ([`26944e25`](https://github.com/vectorize-io/hindsight/commit/26944e25))

**Bug Fixes**

- Added a supported setup command to register hooks reliably, fixing hook registration issues. ([`22ca6a8d`](https://github.com/vectorize-io/hindsight/commit/22ca6a8d))
- Fixed Claude Code integration compatibility on Windows. ([`a94a90ea`](https://github.com/vectorize-io/hindsight/commit/a94a90ea))
