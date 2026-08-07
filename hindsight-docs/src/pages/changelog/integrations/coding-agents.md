---
hide_table_of_contents: true
---

# Coding Agents Integration Changelog

Changelog for [`@vectorize-io/hindsight-coding-agents`](https://www.npmjs.com/package/@vectorize-io/hindsight-coding-agents).

For the source code, see [`hindsight-integrations/coding-agents`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/coding-agents).

← [Back to main changelog](/changelog)

## [0.0.5](https://github.com/vectorize-io/hindsight/tree/integrations/coding-agents/v0.0.5)

**Features**

- Added a local daemon mode for the coding-agents integration and allow choosing which Hindsight server to connect to during installation.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/e1b3d438e" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>e1b3d438e</a>

**Improvements**

- Improved Codex plugin migration by carrying over the existing endpoint when switching to the coding-agents integration.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/18a58ef3a" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>18a58ef3a</a>

## [0.0.4](https://github.com/vectorize-io/hindsight/tree/integrations/coding-agents/v0.0.4)

**Features**

- Added a new --import-conversations option to import past coding-agent conversations into Hindsight as a migration path from per-agent plugins.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/17b7f46ae" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>17b7f46ae</a>

**Bug Fixes**

- Fixed how imported sessions are attributed so they’re matched by recorded working directory (cwd) rather than by session name.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/d238d2f7d" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>d238d2f7d</a>
- Improved Devin transcript importing by using node:sqlite, and the installer now refuses to proceed when the required SQLite support isn’t available (with corrected guidance messaging).<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/333812c85" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>333812c85</a>

## [0.0.3](https://github.com/vectorize-io/hindsight/tree/integrations/coding-agents/v0.0.3)

**Breaking Changes**

- Installation now requires explicitly choosing a target (either “install all” or a specific harness).<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/05b55d20f" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>05b55d20f</a>

**Bug Fixes**

- Added support for additional harnesses and improved environment-based configuration fallback so installs work more reliably across setups.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/2b1c4989f" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>2b1c4989f</a>

## [0.0.2](https://github.com/vectorize-io/hindsight/tree/integrations/coding-agents/v0.0.2)

**Bug Fixes**

- Fixes re-install behavior so Antigravity and the Claude MCP server are correctly repointed instead of left misconfigured.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/47b679ec2" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>47b679ec2</a>

## [0.0.1](https://github.com/vectorize-io/hindsight/tree/integrations/coding-agents/v0.0.1)

**Features**

- Enabled coding agents to use Hindsight’s pluggable long-term memory for persistent context across tasks.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/nicoloboschi.png?size=40" alt="@nicoloboschi" width="18" height="18" style={{borderRadius: "50%"}} />@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b5d8439c8" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b5d8439c8</a>
