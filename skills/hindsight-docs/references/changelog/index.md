---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="Changelog" subtitle="User-facing changes only. Internal maintenance and infrastructure updates are omitted." />

## [0.5.3](https://github.com/vectorize-io/hindsight/releases/tag/v0.5.3)

**Features**

- Add a setting to limit how many memories can be consolidated per round, helping control consolidation workload and cost.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/ca561aca" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>ca561aca</a>
- Add integration with the OpenAI Agents SDK.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b8da88c8" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b8da88c8</a>
- Improve mental model refresh and updates with structured operations and cleaner observation handling on upsert.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/8b80959b" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>8b80959b</a>
- Allow extensions to requeue work by throwing a DeferOperation exception from worker operations.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/f8904797" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>f8904797</a>
- Make recall budget mapping configurable per memory bank.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/576c44d2" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>576c44d2</a>
- Control plane now shows failed consolidation counts with a drilldown to investigate issues.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/e1e5f36c" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>e1e5f36c</a>
- Add mental-model staleness signals and a refreshed UI/experience for reviewing model history and snapshots.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/654e4c0c" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>654e4c0c</a>
- Replace the embedded Paperclip library with the Paperclip plugin for more flexible integrations.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/benfrank241" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@benfrank241</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/c571fac7" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>c571fac7</a>
- JSON logs can now include tenant information and support a configurable allowlist for what gets logged.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/3bedc1ce" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>3bedc1ce</a>
- CLI now supports named connection profiles (via -p/--profile) for easier switching between environments.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/70d60e96" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>70d60e96</a>

**Improvements**

- Reduce the default number of retries for LLM calls to fail faster when providers are erroring or unavailable.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b52b483c" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b52b483c</a>
- Make reranker failures easier to diagnose and add a configurable timeout for the TEI reranker.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/octo-patch" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@octo-patch</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/69383af8" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>69383af8</a>

**Bug Fixes**

- Fix crashes when using Jina MLX on Metal GPUs by serializing GPU operations.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/lkttle" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@lkttle</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/2e74a324" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>2e74a324</a>
- Fix database migration path so upgrades from v0.4.22 to v0.5.x work correctly.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/5437cc02" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>5437cc02</a>
- Prevent orphaned observations if a source memory is deleted during consolidation.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/f9042e37" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>f9042e37</a>
- Fix Ollama requests by explicitly disabling "think" mode in the native call payload.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/karl-8888" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@karl-8888</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/7d4fd1aa" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>7d4fd1aa</a>
- Fix file retain uploads and prevent orphaned retained files.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/christerence" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@christerence</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/9e30ae25" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>9e30ae25</a>
- Fix file retain API to correctly accept and map a provided "timestamp" field.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/christerence" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@christerence</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/13f3052e" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>13f3052e</a>
- Improve fairness across tenants when workers claim tasks, reducing the chance of noisy tenants starving others.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/cdbartholomew" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@cdbartholomew</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/a5e53721" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>a5e53721</a>
- Ensure the mental model max_tokens setting is respected during refresh/reflect operations.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/568e3c30" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>568e3c30</a>
- Fix control-plane links by properly encoding bank IDs in URLs end-to-end.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/cbaec36f" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>cbaec36f</a>
- Make task submission idempotent when a payload is already set, preventing duplicate/failed submissions.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/088dfecb" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>088dfecb</a>

## [0.5.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.5.2)

**Features**

- Added a co-occurrence graph view for exploring entity relationships in the control plane.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/f64c5d20" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>f64c5d20</a>
- Added recall controls to the mental model trigger API/CLI so you can tune what gets recalled during runs.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/f2fc8f9f" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>f2fc8f9f</a>
- Async operations now expose task payload details and associated document IDs for better observability and debugging.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/870bf4a3" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>870bf4a3</a>

**Improvements**

- Revamped the control plane bank statistics view for clearer insights.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/34365c32" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>34365c32</a>
- Clients now send an identifying User-Agent header on all HTTP requests for easier server-side diagnostics.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/9372462e" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>9372462e</a>

**Bug Fixes**

- Fixed consolidation retry budget handling so retries are correctly applied at the LLM call site.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/r266-tech" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@r266-tech</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/dee58139" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>dee58139</a>
- Fixed a crash during retain when embeddings and extracted facts counts didn’t match.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/dbd1d1a7" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>dbd1d1a7</a>
- Improved embedded mode cleanup stability by adding a timeout when acquiring the cleanup lock (prevents hangs).<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/r266-tech" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@r266-tech</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/6b5aa3af" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>6b5aa3af</a>
- OpenClaw plugins now reliably register agent hooks on every entry invocation.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/1be5ff33" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>1be5ff33</a>
- TypeScript SDK now re-exports BankTemplate types from the package root for simpler imports.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/mrkhachaturov" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@mrkhachaturov</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/581bbf3f" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>581bbf3f</a>
- Bank template configuration validation was aligned with configurable fields to prevent invalid/ignored settings.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/mrkhachaturov" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@mrkhachaturov</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/099f4c92" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>099f4c92</a>

## [0.5.1](https://github.com/vectorize-io/hindsight/releases/tag/v0.5.1)

**Breaking Changes**

- OpenClaw now reads configuration from plugin config instead of environment variables. ([`e22ae05f`](https://github.com/vectorize-io/hindsight/commit/e22ae05f))

**Features**

- Added SiliconFlow as a supported reranker provider. ([`d0b2ab9a`](https://github.com/vectorize-io/hindsight/commit/d0b2ab9a))
- Added an interactive OpenClaw setup wizard with Cloud / API / Embedded modes. ([`87322396`](https://github.com/vectorize-io/hindsight/commit/87322396))
- Added a config-aware CLI to backfill OpenClaw history. ([`72fd3d59`](https://github.com/vectorize-io/hindsight/commit/72fd3d59))
- Added OpenClaw session pattern filtering to ignore or treat sessions as stateless. ([`5a61ac50`](https://github.com/vectorize-io/hindsight/commit/5a61ac50))
- Added a Cloudflare OAuth proxy integration option for self-hosted Hindsight. ([`aad07a14`](https://github.com/vectorize-io/hindsight/commit/aad07a14))
- Expanded the CLI to cover all OpenAPI endpoints and request-body parameters. ([`c05c491d`](https://github.com/vectorize-io/hindsight/commit/c05c491d))
- Added a default bank template environment variable (HINDSIGHT_API_DEFAULT_BANK_TEMPLATE). ([`fc941d5c`](https://github.com/vectorize-io/hindsight/commit/fc941d5c))
- Added a daemon lifecycle package (@vectorize-io/hindsight-all) to simplify running the all-in-one daemon. ([`576016f5`](https://github.com/vectorize-io/hindsight/commit/576016f5))
- Added recallTags and recallTagsMatch configuration options to control which tagged memories are recalled. ([`b57e337f`](https://github.com/vectorize-io/hindsight/commit/b57e337f))

**Improvements**

- Improved OpenClaw reliability with more resilient startup behavior and richer retain metadata. ([`1f1716bd`](https://github.com/vectorize-io/hindsight/commit/1f1716bd))

**Bug Fixes**

- OpenClaw setup wizard now prompts for the token value (not the env var name). ([`9679d813`](https://github.com/vectorize-io/hindsight/commit/9679d813))
- Fixed embedded mode daemon start/stop race that could terminate healthy daemons. ([`e5724fcb`](https://github.com/vectorize-io/hindsight/commit/e5724fcb))
- Fixed reranker initialization issues to show real import errors and avoid a Transformers 5.x race in jina-mlx. ([`f82f58fa`](https://github.com/vectorize-io/hindsight/commit/f82f58fa))
- Fixed worker consolidation slot accounting to respect the configured maximum concurrency. ([`2d74007d`](https://github.com/vectorize-io/hindsight/commit/2d74007d))
- Improved CLI API error output by including the HTTP response body. ([`93300b91`](https://github.com/vectorize-io/hindsight/commit/93300b91))
- Fixed CLI memory listing showing "[UNKNOWN]" for fact types. ([`2635bbb4`](https://github.com/vectorize-io/hindsight/commit/2635bbb4))
- Fixed recall ranking so RRF ordering is preserved when the reranker is configured as a passthrough. ([`4f9cf15c`](https://github.com/vectorize-io/hindsight/commit/4f9cf15c))
- Fixed retain chunk insertion to be idempotent and avoid repeated retries on integrity errors. ([`2d95f78b`](https://github.com/vectorize-io/hindsight/commit/2d95f78b))
- Fixed retain ANN seed temp table creation to run inside a transaction for better reliability. ([`3fc87e76`](https://github.com/vectorize-io/hindsight/commit/3fc87e76))
- Fixed LLM requests to use the correct max token parameter for reasoning models and Azure OpenAI. ([`7b2263ba`](https://github.com/vectorize-io/hindsight/commit/7b2263ba))

## [0.5.0](https://github.com/vectorize-io/hindsight/releases/tag/v0.5.0)

**Breaking Changes**

- Removed BFS and MPFP graph retrieval strategies. LinkExpansionRetriever is now the sole graph retrieval algorithm, offering simpler, faster, and more accurate results. ([`ea834bc7`](https://github.com/vectorize-io/hindsight/commit/ea834bc7))
- Dropped the `hindsight-hermes` integration package. ([`cf0537ba`](https://github.com/vectorize-io/hindsight/commit/cf0537ba))

**Features**

- Built-in llama.cpp LLM provider for fully local inference without external API calls. ([`f74b577e`](https://github.com/vectorize-io/hindsight/commit/f74b577e))
- Retain `update_mode='append'` for concatenating new content onto an existing document instead of replacing it. ([`3c633e5e`](https://github.com/vectorize-io/hindsight/commit/3c633e5e))
- OpenRouter support for LLM, embeddings, and reranking. ([`e5944b63`](https://github.com/vectorize-io/hindsight/commit/e5944b63))
- Bank template import/export with Template Hub — export a bank's configuration, mental models, and directives as a reusable manifest, then import into other banks. ([`30a319a6`](https://github.com/vectorize-io/hindsight/commit/30a319a6))
- Constellation view in the Control Plane — interactive, zoomable canvas visualization of entity relationship graphs with heat-gradient coloring and dark mode support. ([`36783df3`](https://github.com/vectorize-io/hindsight/commit/36783df3))
- Added `detail` parameter to list/get mental model endpoints for controlling response verbosity. ([`8d1bfbbd`](https://github.com/vectorize-io/hindsight/commit/8d1bfbbd))
- Added AutoGen integration (`hindsight-autogen`) for persistent long-term memory in AutoGen agents. ([`a757765a`](https://github.com/vectorize-io/hindsight/commit/a757765a))
- Added Paperclip integration (`@vectorize-io/hindsight-paperclip`) with Express middleware and process adapter modes for stateless agent memory. ([`81441ee9`](https://github.com/vectorize-io/hindsight/commit/81441ee9))
- Added OpenCode persistent memory plugin for the OpenCode editor. ([`e1c6220f`](https://github.com/vectorize-io/hindsight/commit/e1c6220f))
- OpenClaw JSONL-backed retain queue for external API resilience — buffers retain calls locally when the API is unreachable. ([`087545cc`](https://github.com/vectorize-io/hindsight/commit/087545cc))
- OpenClaw now supports `bankId` for static bank configurations. ([`0e81d1a2`](https://github.com/vectorize-io/hindsight/commit/0e81d1a2))
- Added Google embeddings and reranker provider support. ([`07de798c`](https://github.com/vectorize-io/hindsight/commit/07de798c))
- Added persistent volume support in Helm chart for local model cache. ([`cefa7554`](https://github.com/vectorize-io/hindsight/commit/cefa7554))
- MCP server now includes a `sync_retain` tool and validates UUID inputs. ([`48185a4b`](https://github.com/vectorize-io/hindsight/commit/48185a4b))
- Recall combined scoring now includes `proof_count` boost for better ranking. ([`26794aab`](https://github.com/vectorize-io/hindsight/commit/26794aab))

**Improvements**

- 3-phase retain pipeline restructures memory ingestion into pre-resolve, insert, and post-link phases, dramatically improving throughput under concurrent load by removing slow reads from write transactions. ([`914ba796`](https://github.com/vectorize-io/hindsight/commit/914ba796))
- Recall entity graph expansion now caps per-entity fanout and includes a timeout fallback, preventing slow queries on banks with high-fanout entities. ([`57f15445`](https://github.com/vectorize-io/hindsight/commit/57f15445))
- Fact serialization in think-prompt now includes `occurred_end` and `mentioned_at` for richer temporal context. ([`37348c85`](https://github.com/vectorize-io/hindsight/commit/37348c85))
- Consolidation observation quality improved with structured processing rules. ([`6f173b10`](https://github.com/vectorize-io/hindsight/commit/6f173b10))

**Bug Fixes**

- LiteLLM SDK embeddings `encoding_format` is now configurable instead of hardcoded. ([`cece2c90`](https://github.com/vectorize-io/hindsight/commit/cece2c90))
- Fixed out-of-range `content_index` crash in recall result mapping. ([`9790d904`](https://github.com/vectorize-io/hindsight/commit/9790d904))
- Experience fact types are now preserved correctly during normalization. ([`9cfdd464`](https://github.com/vectorize-io/hindsight/commit/9cfdd464))
- Clear memories endpoint no longer deletes the bank profile. ([`26a64cc0`](https://github.com/vectorize-io/hindsight/commit/26a64cc0))
- Embedding daemon clears stale processes on the port before starting. ([`7d6c570a`](https://github.com/vectorize-io/hindsight/commit/7d6c570a))
- Per-bank vector index migration now respects vector extension configuration. ([`4fd7c5d1`](https://github.com/vectorize-io/hindsight/commit/4fd7c5d1))
- Timeline group sort uses numeric date comparison instead of locale string comparison. ([`f3f2c6b0`](https://github.com/vectorize-io/hindsight/commit/f3f2c6b0))
- Resolved 25 test regressions from the streaming retain pipeline. ([`7415ebff`](https://github.com/vectorize-io/hindsight/commit/7415ebff))
- MCP server now auto-coerces string-encoded JSON in tool arguments. ([`443c94c8`](https://github.com/vectorize-io/hindsight/commit/443c94c8))
- Entity labels structure is now validated on PATCH to prevent invalid configurations. ([`7e23f8e1`](https://github.com/vectorize-io/hindsight/commit/7e23f8e1))
- Fixed `bank_id` metric label to be opt-in, preventing OTel memory leak. ([`cf4bd598`](https://github.com/vectorize-io/hindsight/commit/cf4bd598))
- Fixed `max_tokens` handling for OpenAI-compatible endpoints with custom base URLs. ([`cd99eef4`](https://github.com/vectorize-io/hindsight/commit/cd99eef4))
- Fixed `event_date` AttributeError when date is None in fact extraction. ([`6cb309f7`](https://github.com/vectorize-io/hindsight/commit/6cb309f7))
- Query analyzer now handles dateparser internal crashes gracefully. ([`e0e65c44`](https://github.com/vectorize-io/hindsight/commit/e0e65c44))
- Embedding profile `.env` overwrite skipped when config has no Hindsight keys. ([`9e2890ba`](https://github.com/vectorize-io/hindsight/commit/9e2890ba))
- Windows compatibility fix for hindsight-embed. ([`f9fe6953`](https://github.com/vectorize-io/hindsight/commit/f9fe6953))
- Addressed critical and high severity security vulnerabilities in dependencies. ([`ee4510a7`](https://github.com/vectorize-io/hindsight/commit/ee4510a7))

## [0.4.22](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.22)

**Features**

- API now supports passing custom LLM request parameters via the HINDSIGHT_API_LLM_EXTRA_BODY configuration. ([`ecaa1ad1`](https://github.com/vectorize-io/hindsight/commit/ecaa1ad1))
- Document metadata is now exposed through the API and control plane. ([`627ec5d5`](https://github.com/vectorize-io/hindsight/commit/627ec5d5))
- Added a /code-review skill for automated code quality checks against project standards. ([`bdb33c58`](https://github.com/vectorize-io/hindsight/commit/bdb33c58))
- ZeroEntropy reranker now supports a configurable base URL. ([`a915584e`](https://github.com/vectorize-io/hindsight/commit/a915584e))
- Codex can now retain structured tool calls from rollout files. ([`3461398b`](https://github.com/vectorize-io/hindsight/commit/3461398b))

**Improvements**

- Embeddings via the LiteLLM SDK can now optionally specify output dimensions. ([`f841bcb9`](https://github.com/vectorize-io/hindsight/commit/f841bcb9))
- API responses now include an X-Ignored-Params header to warn when unknown request parameters were ignored. ([`cef42d81`](https://github.com/vectorize-io/hindsight/commit/cef42d81))
- OpenClaw CLI startup is faster by deferring heavy initialization until the service starts. ([`41025c3b`](https://github.com/vectorize-io/hindsight/commit/41025c3b))

**Bug Fixes**

- Mental model triggers now support the full config schema, including tag matching and tag group filters. ([`2c32ffad`](https://github.com/vectorize-io/hindsight/commit/2c32ffad))
- Cohere reranking via Azure endpoints now works reliably (avoids 404 errors). ([`84985ee9`](https://github.com/vectorize-io/hindsight/commit/84985ee9))
- Claude Code provider no longer defers to built-in tools, preventing MCP tool handling issues. ([`fa82efc8`](https://github.com/vectorize-io/hindsight/commit/fa82efc8))
- Recall endpoint now returns metadata correctly instead of dropping it from the response. ([`4768bf39`](https://github.com/vectorize-io/hindsight/commit/4768bf39))
- Gemini 3.1+ tool calls now read thought signatures correctly. ([`1b5c262a`](https://github.com/vectorize-io/hindsight/commit/1b5c262a))
- First-person agent memories are now correctly classified as "experience" facts. ([`00961156`](https://github.com/vectorize-io/hindsight/commit/00961156))
- Codex upgrades now preserve and merge new settings instead of skipping them. ([`b104bad0`](https://github.com/vectorize-io/hindsight/commit/b104bad0))
- LlamaIndex integration fixes improve document ID handling, memory API behavior, and ReAct tracing. ([`d93dfea8`](https://github.com/vectorize-io/hindsight/commit/d93dfea8))

## [0.4.21](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.21)

**Features**

- Added audit logging for feature usage tracking, including request duration in audit entries. ([`083295dc`](https://github.com/vectorize-io/hindsight/commit/083295dc))
- Added Hindsight memory integration for the OpenAI Codex CLI. ([`0b17a67c`](https://github.com/vectorize-io/hindsight/commit/0b17a67c))
- Added an MCP hook to filter tool visibility per user. ([`f8285b7b`](https://github.com/vectorize-io/hindsight/commit/f8285b7b))
- Added a per-bank limit setting to cap the number of observations stored per scope. ([`b32767ca`](https://github.com/vectorize-io/hindsight/commit/b32767ca))
- Added native Windows support so Hindsight can run without Docker. ([`c5700ff5`](https://github.com/vectorize-io/hindsight/commit/c5700ff5))
- Added a 'none' LLM provider to support chunk-only storage without LLM calls. ([`9e5a066d`](https://github.com/vectorize-io/hindsight/commit/9e5a066d))
- Added a setup command/skill to register hooks more reliably. ([`22ca6a8d`](https://github.com/vectorize-io/hindsight/commit/22ca6a8d))
- Hermes now supports file-based configuration. ([`0ff36548`](https://github.com/vectorize-io/hindsight/commit/0ff36548))
- Added a LiteLLM-based provider to support Bedrock and many additional LLM providers. ([`db70fdbe`](https://github.com/vectorize-io/hindsight/commit/db70fdbe))
- Added support for Strands Agents SDK integration with Hindsight memory tools. ([`7fe773c0`](https://github.com/vectorize-io/hindsight/commit/7fe773c0))
- Added LlamaIndex integration. ([`2d787c4f`](https://github.com/vectorize-io/hindsight/commit/2d787c4f))
- Added AG2 framework integration. ([`73123870`](https://github.com/vectorize-io/hindsight/commit/73123870))
- Added support for Ark and Volcano LLM providers. ([`417fac61`](https://github.com/vectorize-io/hindsight/commit/417fac61))
- Retain now supports delta mode to skip LLM processing for unchanged chunks on upsert. ([`fd88c0ef`](https://github.com/vectorize-io/hindsight/commit/fd88c0ef))
- Claude Code integration can now retain full sessions with document upsert and configurable tags, and records tool calls as structured JSON. ([`2d31b67d`](https://github.com/vectorize-io/hindsight/commit/2d31b67d))
- MCP retain tool now supports selecting a retain strategy via a parameter. ([`4285e944`](https://github.com/vectorize-io/hindsight/commit/4285e944))

**Improvements**

- OpenClaw logging is now configurable and can emit structured output. ([`d441ab81`](https://github.com/vectorize-io/hindsight/commit/d441ab81))
- Made inclusion of source facts in search observations configurable. ([`5095d5e3`](https://github.com/vectorize-io/hindsight/commit/5095d5e3))
- Integrations no longer use hardcoded default models, relying on configured defaults instead. ([`58e68f3e`](https://github.com/vectorize-io/hindsight/commit/58e68f3e))

**Bug Fixes**

- Improved MCP server compatibility by handling Claude Code GET probes and allowing stateless HTTP mode to be configured. ([`d8050387`](https://github.com/vectorize-io/hindsight/commit/d8050387))
- Per-bank vector index creation now respects the configured vector extension setting. ([`6488c9bc`](https://github.com/vectorize-io/hindsight/commit/6488c9bc))
- Verbose retain extraction now correctly includes the retain mission context. ([`d2965e64`](https://github.com/vectorize-io/hindsight/commit/d2965e64))
- Codex integration no longer crashes on startup when the API quota is exhausted (HTTP 429). ([`111e8c70`](https://github.com/vectorize-io/hindsight/commit/111e8c70))
- OpenAI embeddings client now correctly parses query parameters included in base_url. ([`a209ef1a`](https://github.com/vectorize-io/hindsight/commit/a209ef1a))
- Fixed tool_choice handling for Codex/Claude Code when forcing specific tool calls. ([`585ac76f`](https://github.com/vectorize-io/hindsight/commit/585ac76f))
- OpenClaw auto-recall now supports a configurable timeout to prevent hangs. ([`cd4d449f`](https://github.com/vectorize-io/hindsight/commit/cd4d449f))
- Fixed control plane UI issues affecting recall and data viewing. ([`6bb83f46`](https://github.com/vectorize-io/hindsight/commit/6bb83f46))
- Recall responses now include associated metadata. ([`0bcbf849`](https://github.com/vectorize-io/hindsight/commit/0bcbf849))
- Python client update_bank_config() now exposes all configurable fields. ([`7c18723f`](https://github.com/vectorize-io/hindsight/commit/7c18723f))
- API OpenAPI schema now correctly includes Pydantic v2 ValidationError fields. ([`939cb40a`](https://github.com/vectorize-io/hindsight/commit/939cb40a))
- JSON-string tags are now coerced to lists for MemoryItem and MCP tools to prevent tagging errors. ([`c5273f5f`](https://github.com/vectorize-io/hindsight/commit/c5273f5f))

## [0.4.20](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.20)

**Features**

- Add a one-command setup CLI package for the NemoClaw integration. ([`d284de28`](https://github.com/vectorize-io/hindsight/commit/d284de28))
- Add a LangGraph integration for using Hindsight memory within LangGraph agents. ([`b4320254`](https://github.com/vectorize-io/hindsight/commit/b4320254))
- Add reflect filters to exclude specific fact types and mental model content during reflection. ([`ea662d06`](https://github.com/vectorize-io/hindsight/commit/ea662d06))
- Introduce independent versioning for integrations so they can be released separately from the core server. ([`31f1c53c`](https://github.com/vectorize-io/hindsight/commit/31f1c53c))
- Add a Claude Code integration plugin. ([`f4390bdc`](https://github.com/vectorize-io/hindsight/commit/f4390bdc))

**Improvements**

- Add a wall-clock timeout to reflect operations so they don’t run indefinitely. ([`8ce06e3e`](https://github.com/vectorize-io/hindsight/commit/8ce06e3e))
- Provide richer context when validating operations via the OperationValidator extension. ([`2eb1019d`](https://github.com/vectorize-io/hindsight/commit/2eb1019d))
- Make the hindsight-api package runnable directly via uvx by adding script entry points. ([`97f7a365`](https://github.com/vectorize-io/hindsight/commit/97f7a365))
- Support passing query parameters during OpenAI-compatible client initialization for broader provider compatibility. ([`20e17f28`](https://github.com/vectorize-io/hindsight/commit/20e17f28))
- Upgrade the default MiniMax model from M2.5 to M2.7. ([`1f1462a5`](https://github.com/vectorize-io/hindsight/commit/1f1462a5))

**Bug Fixes**

- Prevent context overflow during observation search by disabling source facts in results. ([`8e2e2d5b`](https://github.com/vectorize-io/hindsight/commit/8e2e2d5b))
- Fix Claude Code integration session startup by pre-starting the daemon in the background. ([`26944e25`](https://github.com/vectorize-io/hindsight/commit/26944e25))
- Fix Claude Code integration installation and configuration experience so setup is more reliable. ([`35b2cbb6`](https://github.com/vectorize-io/hindsight/commit/35b2cbb6))
- Fix a memory leak in entity resolution that could grow over time under load. ([`e6333719`](https://github.com/vectorize-io/hindsight/commit/e6333719))
- Avoid crashes and retain failures when the Postgres pg_trgm extension is unavailable by handling detection/fallback correctly. ([`365fa3ce`](https://github.com/vectorize-io/hindsight/commit/365fa3ce))
- Strip Markdown code fences from model outputs across all LLM providers for more consistent parsing. ([`2f2db2a6`](https://github.com/vectorize-io/hindsight/commit/2f2db2a6))
- Return a clear 400 error for empty recall queries and fix a SQL parameterization issue. ([`5cdc714a`](https://github.com/vectorize-io/hindsight/commit/5cdc714a))
- Ensure file retain requests include authentication headers so uploads work in authenticated deployments. ([`78aa7c53`](https://github.com/vectorize-io/hindsight/commit/78aa7c53))
- Fix MCP tool calls when MCP_AUTH_TOKEN and TENANT_API_KEY differ. ([`8364b9c5`](https://github.com/vectorize-io/hindsight/commit/8364b9c5))
- Allow claude-agent-sdk to install correctly on Linux/Docker environments. ([`3f31cbf5`](https://github.com/vectorize-io/hindsight/commit/3f31cbf5))
- In LiteLLM mode, fall back to the last user message when no explicit hindsight query is provided. ([`5e8952c5`](https://github.com/vectorize-io/hindsight/commit/5e8952c5))
- Fix non-atomic async operation creation to prevent inconsistent operation records. ([`94cf89b5`](https://github.com/vectorize-io/hindsight/commit/94cf89b5))
- Prevent orphaned parent operations when a batch retain child fails unexpectedly. ([`43942455`](https://github.com/vectorize-io/hindsight/commit/43942455))
- Fix failures for non-ASCII entity names by ensuring entity IDs are set correctly. ([`438ce98b`](https://github.com/vectorize-io/hindsight/commit/438ce98b))
- Correctly store LLM facts labeled as "assistant" as "experience" in the database. ([`446c75f3`](https://github.com/vectorize-io/hindsight/commit/446c75f3))

## [0.4.19](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.19)

**Features**

- TypeScript client now works in Deno environments. ([`72c25c97`](https://github.com/vectorize-io/hindsight/commit/72c25c97))
- Added Agno integration to use Hindsight as a memory toolkit. ([`8c378b98`](https://github.com/vectorize-io/hindsight/commit/8c378b98))
- Added Hermes Agent integration (hindsight-hermes) for persistent memory. ([`ef90842f`](https://github.com/vectorize-io/hindsight/commit/ef90842f))
- Expanded retain behavior with new `verbatim` and `chunks` extraction modes and named retain strategies. ([`e4f8a157`](https://github.com/vectorize-io/hindsight/commit/e4f8a157))

**Improvements**

- Improved local reranker performance/efficiency with FP16 and bucketed batching, plus compatibility with Transformers 5.x. ([`e7da7d0e`](https://github.com/vectorize-io/hindsight/commit/e7da7d0e))

**Bug Fixes**

- Prevented silent memory loss when consolidation fails (failed consolidations are tracked and can be recovered). ([`28dac7c7`](https://github.com/vectorize-io/hindsight/commit/28dac7c7))
- Fixed Docker control-plane startup to respect the configured control-plane hostname. ([`8a64dc8d`](https://github.com/vectorize-io/hindsight/commit/8a64dc8d))
- Database cleanup migration now removes orphaned observation memory units to avoid inconsistent memory state. ([`f09ad9de`](https://github.com/vectorize-io/hindsight/commit/f09ad9de))
- Deleting a document now also deletes linked memory units to prevent leftover/stale memory entries. ([`f27bd953`](https://github.com/vectorize-io/hindsight/commit/f27bd953))
- Fixed MCP middleware to send an Accept header, preventing 406 response errors in some setups. ([`836fd81e`](https://github.com/vectorize-io/hindsight/commit/836fd81e))
- Improved compatibility with Gemini tool-calling by preserving thought signature metadata to avoid failures on gemini-3.1-flash-lite-preview. ([`21f9f46c`](https://github.com/vectorize-io/hindsight/commit/21f9f46c))

## [0.4.18](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.18)

**Features**

- Add compound tag filtering using tag groups. ([`5de793ee`](https://github.com/vectorize-io/hindsight/commit/5de793ee))
- Publish new slim Python packages (hindsight-api-slim and hindsight-all-slim) for smaller installs. ([`15ea23d5`](https://github.com/vectorize-io/hindsight/commit/15ea23d5))
- Add MiniMax as a supported LLM provider. ([`2344484f`](https://github.com/vectorize-io/hindsight/commit/2344484f))
- Add Jina MLX reranker provider optimized for Apple Silicon. ([`1caf5ec9`](https://github.com/vectorize-io/hindsight/commit/1caf5ec9))

**Improvements**

- Allow configuring maximum recall query tokens via an environment variable. ([`66dedb8d`](https://github.com/vectorize-io/hindsight/commit/66dedb8d))
- Improve retrieval performance by switching to per-bank HNSW indexes. ([`43b3efc4`](https://github.com/vectorize-io/hindsight/commit/43b3efc4))

**Bug Fixes**

- Prevent reranking failures by truncating long documents that exceed LiteLLM reranker context limits. ([`eeb938fc`](https://github.com/vectorize-io/hindsight/commit/eeb938fc))
- Ensure recalled memories are injected as system context for OpenClaw. ([`b17f338e`](https://github.com/vectorize-io/hindsight/commit/b17f338e))
- Ensure embedded profiles are registered in CLI metadata when the daemon starts. ([`06b0f74a`](https://github.com/vectorize-io/hindsight/commit/06b0f74a))
- Cancel in-flight async operations when a bank is deleted to avoid dangling work. ([`0560f626`](https://github.com/vectorize-io/hindsight/commit/0560f626))

## [0.4.17](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.17)

**Features**

- Added a manual retry option for failed asynchronous operations. ([`dcaacbe4`](https://github.com/vectorize-io/hindsight/commit/dcaacbe4))
- You can now change/update tags on an existing document. ([`1b4ad7f4`](https://github.com/vectorize-io/hindsight/commit/1b4ad7f4))
- Added history tracking and a diff view for mental model changes. ([`e2baca8b`](https://github.com/vectorize-io/hindsight/commit/e2baca8b))
- Added observation history tracking with a UI diff view to review changes over time. ([`576473b6`](https://github.com/vectorize-io/hindsight/commit/576473b6))
- File uploads can now choose a parser per request, with configurable fallback chains. ([`99220d05`](https://github.com/vectorize-io/hindsight/commit/99220d05))
- Added an extension hook that runs after file-to-Markdown conversion completes. ([`1d17dea2`](https://github.com/vectorize-io/hindsight/commit/1d17dea2))

**Improvements**

- Operations view now supports filtering by operation type and has more reliable auto-refresh behavior. ([`f7a60f89`](https://github.com/vectorize-io/hindsight/commit/f7a60f89))
- Added token limits for “source facts” used during consolidation and recall to better control context usage. ([`5d05962d`](https://github.com/vectorize-io/hindsight/commit/5d05962d))
- Improved bank selector usability by truncating very long bank names in the dropdown. ([`1e40cd22`](https://github.com/vectorize-io/hindsight/commit/1e40cd22))

**Bug Fixes**

- Fixed webhook schema issues affecting multi-tenant retain webhooks. ([`32a4882a`](https://github.com/vectorize-io/hindsight/commit/32a4882a))
- Fixed file ingestion failures by stripping null bytes from parsed file content before retaining. ([`cd3a6a22`](https://github.com/vectorize-io/hindsight/commit/cd3a6a22))
- Fixed tool selection handling for OpenAI-compatible providers when using named tool_choice. ([`1cdfb7c2`](https://github.com/vectorize-io/hindsight/commit/1cdfb7c2))
- Improved consolidation behavior to prioritize a bank’s mission over an ephemeral-state heuristic. ([`00ccf0b2`](https://github.com/vectorize-io/hindsight/commit/00ccf0b2))
- Fixed database migrations to correctly handle mental model embedding dimension changes. ([`7accac94`](https://github.com/vectorize-io/hindsight/commit/7accac94))
- Fixed file upload failures caused by an Iris parser httpx read timeout. ([`fa3501d4`](https://github.com/vectorize-io/hindsight/commit/fa3501d4))
- Improved reliability of running migrations by serializing Alembic upgrades within the process. ([`f88b50a4`](https://github.com/vectorize-io/hindsight/commit/f88b50a4))
- Fixed Google Cloud Storage authentication when using Workload Identity Federation credentials. ([`d2504ac5`](https://github.com/vectorize-io/hindsight/commit/d2504ac5))
- Fixed the bank selector to refresh the bank list when the dropdown is opened. ([`0ad8c2d0`](https://github.com/vectorize-io/hindsight/commit/0ad8c2d0))

## [0.4.16](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.16)

**Features**

- Added Webhooks with `consolidation.completed` and `retain.completed` events. ([`abbf874d`](https://github.com/vectorize-io/hindsight/commit/abbf874d))

**Improvements**

- Improved OpenClaw recall/retention controls. ([`d425e93c`](https://github.com/vectorize-io/hindsight/commit/d425e93c))
- Improved search/reranking quality by switching combined scoring to multiplicative boosts. ([`aa8e5475`](https://github.com/vectorize-io/hindsight/commit/aa8e5475))
- Improved performance of observation recall by 40x on large banks. ([`ad2cf72a`](https://github.com/vectorize-io/hindsight/commit/ad2cf72a))
- Improved server shutdown behavior by capping graceful shutdown time and allowing a forced kill on a second Ctrl+C. ([`4c058b4b`](https://github.com/vectorize-io/hindsight/commit/4c058b4b))

**Bug Fixes**

- Fixed an async deadlock risk by running database schema migrations in a background thread during startup. ([`e0a2ac63`](https://github.com/vectorize-io/hindsight/commit/e0a2ac63))
- Fixed webhook delivery/outbox processing so transactions don’t silently roll back due to using the wrong database schema name. ([`75b95106`](https://github.com/vectorize-io/hindsight/commit/75b95106))
- Fixed observation results to correctly resolve and return related chunks using source_memory_ids. ([`cb6d1c46`](https://github.com/vectorize-io/hindsight/commit/cb6d1c46))
- Fixed MCP bank-level tool filtering compatibility with FastMCP 3.x. ([`f17406fd`](https://github.com/vectorize-io/hindsight/commit/f17406fd))
- Fixed crashes when an LLM returns invalid JSON across all retries (now handled cleanly instead of raising a TypeError). ([`66423b85`](https://github.com/vectorize-io/hindsight/commit/66423b85))
- Fixed observations without source dates to preserve missing (None) temporal fields instead of incorrectly populating them. ([`891c33b1`](https://github.com/vectorize-io/hindsight/commit/891c33b1))

## [0.4.15](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.15)

**Features**

- Added observation_scopes to control the granularity/visibility of observations. ([`55af4681`](https://github.com/vectorize-io/hindsight/commit/55af4681))
- List documents API now supports filtering by tags (and fixes the q parameter description). ([`1d70abfe`](https://github.com/vectorize-io/hindsight/commit/1d70abfe))
- Added PydanticAI integration for persistent agent memory. ([`cab5a40f`](https://github.com/vectorize-io/hindsight/commit/cab5a40f))
- Added richer entity label support (optional labels, free-form values, multi-value fields, and UI polish). ([`9b96becc`](https://github.com/vectorize-io/hindsight/commit/9b96becc))
- Added support for timestamp="unset" so content can be retained without a date. ([`f903948a`](https://github.com/vectorize-io/hindsight/commit/f903948a))
- OpenClaw can now automatically retain the last n+2 turns every n turns (default n=10). ([`ad1660b3`](https://github.com/vectorize-io/hindsight/commit/ad1660b3))
- Added configurable Gemini/Vertex AI safety settings for LLM calls. ([`73ef99e7`](https://github.com/vectorize-io/hindsight/commit/73ef99e7))
- Added extension hooks to customize root routing and error headers. ([`e407f4bc`](https://github.com/vectorize-io/hindsight/commit/e407f4bc))

**Improvements**

- Improved recall performance by fetching all recall chunks in a single query. ([`61bf428b`](https://github.com/vectorize-io/hindsight/commit/61bf428b))
- Improved recall/retain performance and scalability for large memory banks. ([`7942f181`](https://github.com/vectorize-io/hindsight/commit/7942f181))

**Bug Fixes**

- Fixed the TypeScript SDK to send null (not undefined) when includeEntities is false. ([`15f4b876`](https://github.com/vectorize-io/hindsight/commit/15f4b876))
- Prevented reflect from failing with context_length_exceeded on large memory banks. ([`77defd96`](https://github.com/vectorize-io/hindsight/commit/77defd96))
- Fixed a consolidation deadlock caused by retrying after zombie processing tasks. ([`c2876490`](https://github.com/vectorize-io/hindsight/commit/c2876490))
- Fixed observations count in the control plane that always showed 0. ([`eaeaa1f2`](https://github.com/vectorize-io/hindsight/commit/eaeaa1f2))
- Fixed ZeroEntropy rerank endpoint URL and ensured the MCP retain async_processing parameter is handled correctly. ([`f6f1a7d8`](https://github.com/vectorize-io/hindsight/commit/f6f1a7d8))
- Fixed JSON serialization issues and logging-related exception propagation when using the claude_code LLM provider. ([`ecb833f4`](https://github.com/vectorize-io/hindsight/commit/ecb833f4))
- Added bank-scoped request validation to prevent cross-bank/invalid bank operations. ([`5270aa5a`](https://github.com/vectorize-io/hindsight/commit/5270aa5a))

## [0.4.14](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.14)

**Features**

- Add Chat SDK integration to give chatbots persistent memory. ([`fed987f9`](https://github.com/vectorize-io/hindsight/commit/fed987f9))
- Allow configuring which MCP tools are exposed per memory bank, and expand the MCP tool set with additional tools and parameters. ([`3ffec650`](https://github.com/vectorize-io/hindsight/commit/3ffec650))
- Enable the bank configuration API by default. ([`4d030707`](https://github.com/vectorize-io/hindsight/commit/4d030707))
- Support filtering graph-based memory retrieval by tags. ([`0bb5ca4c`](https://github.com/vectorize-io/hindsight/commit/0bb5ca4c))
- Add batch observations consolidation to process multiple observations more efficiently. ([`0aa7c2b3`](https://github.com/vectorize-io/hindsight/commit/0aa7c2b3))
- Add OpenClaw options to toggle autoRecall and exclude specific providers. ([`3f9eb27c`](https://github.com/vectorize-io/hindsight/commit/3f9eb27c))
- Add a ZeroEntropy reranker provider option. ([`17259675`](https://github.com/vectorize-io/hindsight/commit/17259675))

**Improvements**

- Increase customization options for reflect, retain, and consolidation behavior. ([`2a322732`](https://github.com/vectorize-io/hindsight/commit/2a322732))
- Include source document metadata in fact extraction results. ([`87219b73`](https://github.com/vectorize-io/hindsight/commit/87219b73))

**Bug Fixes**

- Raise a clear error when embedding dimensions exceed pgvector HNSW limits (instead of failing later at runtime). ([`8cd65b98`](https://github.com/vectorize-io/hindsight/commit/8cd65b98))
- Fix multi-tenant schema isolation issues in storage and the bank config API. ([`b180b3ad`](https://github.com/vectorize-io/hindsight/commit/b180b3ad))
- Ensure LiteLLM embedding calls use the correct float encoding format to prevent embedding failures. ([`58f2de70`](https://github.com/vectorize-io/hindsight/commit/58f2de70))
- Improve recall performance by reducing memory usage during retrieval. ([`9f0c031d`](https://github.com/vectorize-io/hindsight/commit/9f0c031d))
- Handle observation regeneration correctly when underlying memories are deleted. ([`ac9a94ad`](https://github.com/vectorize-io/hindsight/commit/ac9a94ad))
- Fix reflect retrieval to correctly populate dependencies and enforce full hierarchical retrieval. ([`8b1a4658`](https://github.com/vectorize-io/hindsight/commit/8b1a4658))
- Fix OpenClaw health checks by passing the auth token to the health endpoint. ([`40b02645`](https://github.com/vectorize-io/hindsight/commit/40b02645))

## [0.4.13](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.13)

**Features**

- Switched the default OpenAI LLM to gpt-4o-mini. ([`325b5cc1`](https://github.com/vectorize-io/hindsight/commit/325b5cc1))
- Observation recall now includes the source facts behind recalled observations. ([`5569d4ad`](https://github.com/vectorize-io/hindsight/commit/5569d4ad))
- Added CrewAI integration to enable persistent memory. ([`41db2960`](https://github.com/vectorize-io/hindsight/commit/41db2960))

**Bug Fixes**

- Fixed npx hindsight-control-plane failing to run. ([`0758827d`](https://github.com/vectorize-io/hindsight/commit/0758827d))
- Improved MCP compatibility by aligning the local MCP implementation with the server and removing the deprecated stateless parameter. ([`ea8163c5`](https://github.com/vectorize-io/hindsight/commit/ea8163c5))
- Fixed Docker startup failures when using named Docker volumes. ([`ac739487`](https://github.com/vectorize-io/hindsight/commit/ac739487))
- Prevented reranker crashes when an upstream provider returns an error. ([`58c4d657`](https://github.com/vectorize-io/hindsight/commit/58c4d657))
- Improved accuracy of fact temporal ordering by reducing per-fact time offsets. ([`c3ef1555`](https://github.com/vectorize-io/hindsight/commit/c3ef1555))
- Client timeout settings are now properly respected. ([`dcaa9f14`](https://github.com/vectorize-io/hindsight/commit/dcaa9f14))
- Fixed documents not being tracked when fact extraction returns zero facts. ([`f78278ea`](https://github.com/vectorize-io/hindsight/commit/f78278ea))

## [0.4.12](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.12)

**Features**

- Accept and ingest PDFs, images, and common Office documents as inputs. ([`224b7b74`](https://github.com/vectorize-io/hindsight/commit/224b7b74))
- Add the Iris file parser for improved document parsing support. ([`7eafba66`](https://github.com/vectorize-io/hindsight/commit/7eafba66))
- Add async Retain support via provider Batch APIs (e.g., OpenAI and Groq) for higher-throughput ingestion. ([`40d42c58`](https://github.com/vectorize-io/hindsight/commit/40d42c58))
- Allow Recall to return chunks only (no memories) by setting max_tokens=0. ([`7dad9da0`](https://github.com/vectorize-io/hindsight/commit/7dad9da0))
- Add a Go client SDK for the Hindsight API. ([`2a47389f`](https://github.com/vectorize-io/hindsight/commit/2a47389f))
- Add support for the pgvectorscale (DiskANN) vector index backend. ([`95c42204`](https://github.com/vectorize-io/hindsight/commit/95c42204))
- Add support for Azure pg_diskann vector indexing. ([`476726c2`](https://github.com/vectorize-io/hindsight/commit/476726c2))

**Improvements**

- Improve reliability of async batch Retain when ingesting large payloads. ([`aefb3fcf`](https://github.com/vectorize-io/hindsight/commit/aefb3fcf))
- Improve AI SDK tooling to make it easier to work with Hindsight programmatically. ([`d06a0259`](https://github.com/vectorize-io/hindsight/commit/d06a0259))

**Bug Fixes**

- Ensure document tags are preserved when using the async Retain flow. ([`b4b5c44a`](https://github.com/vectorize-io/hindsight/commit/b4b5c44a))
- Fix OpenClaw ingestion failures for very large content (E2BIG). ([`6bad6673`](https://github.com/vectorize-io/hindsight/commit/6bad6673))
- Harden OpenClaw behavior (safer shell usage, better HTTP mode handling, and more reliable initialization), including per-user banks support. ([`c4610130`](https://github.com/vectorize-io/hindsight/commit/c4610130))
- Improve Python client async API consistency and reduce connection drop issues via keepalive timeout fixes. ([`8114ef44`](https://github.com/vectorize-io/hindsight/commit/8114ef44))

## [0.4.11](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.11)

**Features**

- Added support for LiteLLM SDK as an embeddings and reranking provider. ([`e408b7e`](https://github.com/vectorize-io/hindsight/commit/e408b7e))
- Expanded Postgres search support with additional text/vector extensions, including TimescaleDB pg_textsearch and vchord/pgvector options. ([`d871c30`](https://github.com/vectorize-io/hindsight/commit/d871c30))
- Added hierarchical configuration scopes (system, tenant, bank) for more flexible multi-tenant setup and overrides. ([`8d731f2`](https://github.com/vectorize-io/hindsight/commit/8d731f2))
- Added reverse proxy/base-path support for running Hindsight behind a proxy. ([`93ddd41`](https://github.com/vectorize-io/hindsight/commit/93ddd41))
- Added MCP tools to create, read, update, and delete mental models. ([`f641b30`](https://github.com/vectorize-io/hindsight/commit/f641b30))
- Added a "docs" skill for agents/tools to access documentation-oriented capabilities. ([`dd1e098`](https://github.com/vectorize-io/hindsight/commit/dd1e098))
- Added an OpenClaw configuration option to skip recall/retain for specific providers. ([`fb7be3e`](https://github.com/vectorize-io/hindsight/commit/fb7be3e))

**Improvements**

- Improved LiteLLM gateway model configuration for more reliable provider/model selection. ([`7d95a00`](https://github.com/vectorize-io/hindsight/commit/7d95a00))
- Exposed actual LLM token usage in retain results to improve cost/usage visibility. ([`83ca669`](https://github.com/vectorize-io/hindsight/commit/83ca669))
- Added user-initiated attribution to request context to improve async task and usage attribution. ([`90be7c6`](https://github.com/vectorize-io/hindsight/commit/90be7c6))
- Added OpenTelemetry tracing for improved request traceability and observability. ([`69dec8e`](https://github.com/vectorize-io/hindsight/commit/69dec8e))
- Helm chart: split TEI embedding and reranker into separate deployments for independent scaling and rollout. ([`43f9a8b`](https://github.com/vectorize-io/hindsight/commit/43f9a8b))
- Helm chart: added PodDisruptionBudgets and per-component affinity controls for more resilient scheduling. ([`9943957`](https://github.com/vectorize-io/hindsight/commit/9943957))

**Bug Fixes**

- Fixed a recursion issue in memory retention that could cause failures or runaway memory usage. ([`4f11210`](https://github.com/vectorize-io/hindsight/commit/4f11210))
- Fixed Reflect API serialization/schema issues for "based_on" so reflections are returned and stored correctly. ([`f9a8a8e`](https://github.com/vectorize-io/hindsight/commit/f9a8a8e))
- Improved MCP server compatibility by allowing extra tool arguments when appropriate and fixing bank ID resolution priority. ([`7ee229b`](https://github.com/vectorize-io/hindsight/commit/7ee229b))
- Added missing trust_code environment configuration support. ([`60574ee`](https://github.com/vectorize-io/hindsight/commit/60574ee))
- Hardened the MCP server with fixes to routing/validation and more accurate usage metering. ([`e798979`](https://github.com/vectorize-io/hindsight/commit/e798979))
- Fixed the slim Docker image to include tiktoken to prevent runtime tokenization errors. ([`6eec83b`](https://github.com/vectorize-io/hindsight/commit/6eec83b))
- Fixed MCP operations not being tracked correctly for usage metering. ([`888b50d`](https://github.com/vectorize-io/hindsight/commit/888b50d))
- Helm chart: fixed GKE deployments overriding the configured HINDSIGHT_API_PORT. ([`03f47e2`](https://github.com/vectorize-io/hindsight/commit/03f47e2))

## [0.4.10](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.10)

**Features**

- Provided a slimmer Docker distribution to reduce image size and speed up pulls. ([`f648178`](https://github.com/vectorize-io/hindsight/commit/f648178))
- Added Markdown support in Reflect and Mental Models content. ([`c4ef090`](https://github.com/vectorize-io/hindsight/commit/c4ef090))
- Added built-in Supabase tenant extension for running Hindsight with Supabase-backed multi-tenancy. ([`e99ee0f`](https://github.com/vectorize-io/hindsight/commit/e99ee0f))
- Added TenantExtension authentication support to the MCP endpoint. ([`fedfb49`](https://github.com/vectorize-io/hindsight/commit/fedfb49))

**Improvements**

- Improved MCP tool availability/routing based on the endpoint being used. ([`d90588b`](https://github.com/vectorize-io/hindsight/commit/d90588b))

**Bug Fixes**

- Stopped logging database usernames and passwords to prevent credential leaks in logs. ([`c568094`](https://github.com/vectorize-io/hindsight/commit/c568094))
- Fixed OpenClaw sessions wiping memory on each new session. ([`981cf60`](https://github.com/vectorize-io/hindsight/commit/981cf60))
- Fixed hindsight-embed profiles not loading correctly. ([`0430588`](https://github.com/vectorize-io/hindsight/commit/0430588))
- Fixed tagged directives so they correctly apply to tagged mental models. ([`278718d`](https://github.com/vectorize-io/hindsight/commit/278718d))
- Fixed a cast error that could cause failures at runtime. ([`093ecff`](https://github.com/vectorize-io/hindsight/commit/093ecff))

**Other**

- Added a docker-compose example to simplify local deployment and testing. ([`5179d5f`](https://github.com/vectorize-io/hindsight/commit/5179d5f))

## [0.4.9](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.9)

**Features**

- New AI SDK integration. ([`7e339e1`](https://github.com/vectorize-io/hindsight/commit/7e339e1))
- Add a Python SDK for running Hindsight in embedded mode (HindsightEmbedded). ([`d3302c9`](https://github.com/vectorize-io/hindsight/commit/d3302c9))
- Add streaming support to the hindsight-litellm wrappers. ([`665877b`](https://github.com/vectorize-io/hindsight/commit/665877b))
- Add OpenClaw support for connecting to an external Hindsight API and using dynamic per-channel memory banks. ([`6b34692`](https://github.com/vectorize-io/hindsight/commit/6b34692))

**Improvements**

- Improve the mental models experience in the control plane UI. ([`7097716`](https://github.com/vectorize-io/hindsight/commit/7097716))
- Reduce noisy Hugging Face logging output. ([`34d9188`](https://github.com/vectorize-io/hindsight/commit/34d9188))

**Bug Fixes**

- Improve recall endpoint reliability by handling timeouts correctly and rejecting overly long queries. ([`dd621a6`](https://github.com/vectorize-io/hindsight/commit/dd621a6))
- Improve /reflect behavior with Claude Code and Codex providers. ([`a43d208`](https://github.com/vectorize-io/hindsight/commit/a43d208))
- Fix OpenClaw shell argument escaping for more reliable command execution. ([`63e2964`](https://github.com/vectorize-io/hindsight/commit/63e2964))

## [0.4.8](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.8)

**Features**

- Added profile support for `hindsight-embed`, enabling separate embedding configurations/workspaces. ([`6c7f057`](https://github.com/vectorize-io/hindsight/commit/6c7f057))
- Added support for additional LLM backends, including OpenAI Codex and Claude Code. ([`539190b`](https://github.com/vectorize-io/hindsight/commit/539190b))

**Improvements**

- Enhanced OpenClaw and `hindsight-embed` parameter/config options for easier configuration and better defaults. ([`749478d`](https://github.com/vectorize-io/hindsight/commit/749478d))
- Added OpenClaw plugin configuration options to select LLM provider and model. ([`8564135`](https://github.com/vectorize-io/hindsight/commit/8564135))
- Server now prints its version during startup to simplify debugging and support requests. ([`1499ce5`](https://github.com/vectorize-io/hindsight/commit/1499ce5))
- Improved tracing/debuggability by propagating request context through asynchronous background tasks. ([`44d9125`](https://github.com/vectorize-io/hindsight/commit/44d9125))
- Added stronger validation and context for mental model create/refresh operations to prevent invalid requests. ([`35127d5`](https://github.com/vectorize-io/hindsight/commit/35127d5))

**Bug Fixes**

- Improved embedding CLI experience with richer logs and isolated profiles to avoid cross-contamination between runs. ([`794a743`](https://github.com/vectorize-io/hindsight/commit/794a743))
- Operation validation now runs correctly in the worker process, preventing invalid background operations from slipping through. ([`96f0e54`](https://github.com/vectorize-io/hindsight/commit/96f0e54))
- Fixed unreliable behavior when using a custom PostgreSQL schema. ([`3825506`](https://github.com/vectorize-io/hindsight/commit/3825506))

## [0.4.7](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.7)

**Features**

- Add extension hooks to validate and customize mental model operations. ([`9c3fda7`](https://github.com/vectorize-io/hindsight/commit/9c3fda7))
- Add support for using an external embedding API provider in OpenClaw plugin (with additional OpenClaw compatibility fixes). ([`4b57b82`](https://github.com/vectorize-io/hindsight/commit/4b57b82))

**Improvements**

- Speed up container startup by preloading the tiktoken encoding during Docker image builds. ([`039944c`](https://github.com/vectorize-io/hindsight/commit/039944c))

**Bug Fixes**

- Prevent PostgreSQL insert failures by stripping null bytes from text fields before saving. ([`ef9d3a1`](https://github.com/vectorize-io/hindsight/commit/ef9d3a1))
- Fix worker schema selection so it uses the correct default database schema. ([`d788a55`](https://github.com/vectorize-io/hindsight/commit/d788a55))
- Honor an already-set HINDSIGHT_API_DATABASE_URL instead of overwriting it in the hindsight-embed workflow. ([`f0cb192`](https://github.com/vectorize-io/hindsight/commit/f0cb192))

## [0.4.6](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.6)

**Improvements**

- Improved OpenClaw configuration setup to make embedding integration easier to configure. ([`27498f9`](https://github.com/vectorize-io/hindsight/commit/27498f9))

**Bug Fixes**

- Fixed OpenClaw embedding version binding/versioning to prevent mismatches when using the embed integration. ([`1163b1f`](https://github.com/vectorize-io/hindsight/commit/1163b1f))

## [0.4.5](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.5)

**Bug Fixes**

- Fixed occasional failures when retaining memories asynchronously with timestamps. ([`cbb8fc6`](https://github.com/vectorize-io/hindsight/commit/cbb8fc6))

## [0.4.4](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.4)

**Bug Fixes**

- Fixed async “retain” operations failing when a timestamp is provided. ([`35f0984`](https://github.com/vectorize-io/hindsight/commit/35f0984))
- Corrected the OpenClaw daemon integration name to “openclaw” (previously “openclawd”). ([`b364bc3`](https://github.com/vectorize-io/hindsight/commit/b364bc3))

## [0.4.3](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.3)

**Features**

- Add Vertex AI as a supported LLM provider. ([`c2ac7d0`](https://github.com/vectorize-io/hindsight/commit/c2ac7d0), [`49ae55a`](https://github.com/vectorize-io/hindsight/commit/49ae55a))
- Add Bearer token authentication for MCP and propagate tenant authentication across MCP requests. ([`0da77ce`](https://github.com/vectorize-io/hindsight/commit/0da77ce))

**Improvements**

- CLI: add a --wait flag for consolidate and a --date filter for listing documents. ([`ff20bf9`](https://github.com/vectorize-io/hindsight/commit/ff20bf9))

**Bug Fixes**

- Fix worker polling deadlocks to prevent background processing from stalling. ([`f4f86e3`](https://github.com/vectorize-io/hindsight/commit/f4f86e3))
- Improve reliability of Docker builds by retrying ML model downloads. ([`ecc590c`](https://github.com/vectorize-io/hindsight/commit/ecc590c))
- Fix tenant authentication handling for internal background tasks and ensure the control-plane forwards required auth to the dataplane. ([`03bf13e`](https://github.com/vectorize-io/hindsight/commit/03bf13e))
- Ensure tenant database migrations run at startup and workers use the correct tenant schema context. ([`657fe02`](https://github.com/vectorize-io/hindsight/commit/657fe02))
- Fix control-plane graph endpoint errors when upstream data is missing. ([`751f99a`](https://github.com/vectorize-io/hindsight/commit/751f99a))

**Other**

- Rename the default bot/user identity from "moltbot" to "openclaw". ([`728ce13`](https://github.com/vectorize-io/hindsight/commit/728ce13))

## [0.4.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.2)

**Features**

- Added Clawdbot/Moltbot/OpenClaw integration. ([`12e9a3d`](https://github.com/vectorize-io/hindsight/commit/12e9a3d))

**Improvements**

- Added additional configuration options to control LLM retry behavior. ([`3f211f0`](https://github.com/vectorize-io/hindsight/commit/3f211f0))
- Added real-time logs showing a detailed timing breakdown during consolidation runs. ([`8781c9f`](https://github.com/vectorize-io/hindsight/commit/8781c9f))

**Bug Fixes**

- Fixed hindsight-embed crashing on macOS. ([`c16ccc2`](https://github.com/vectorize-io/hindsight/commit/c16ccc2))

## [0.4.1](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.1)

**Features**

- Added support for using a non-default PostgreSQL schema by default. ([`2b72e1f`](https://github.com/vectorize-io/hindsight/commit/2b72e1f))

**Improvements**

- Improved memory consolidation performance (benchmarking and optimizations). ([`b43ef98`](https://github.com/vectorize-io/hindsight/commit/b43ef98))

**Bug Fixes**

- Fixed the /version endpoint returning an incorrect version. ([`cfcc23c`](https://github.com/vectorize-io/hindsight/commit/cfcc23c))
- Fixed mental model search failing due to UUID type mismatch after text-ID migration. ([`94cc0a1`](https://github.com/vectorize-io/hindsight/commit/94cc0a1))
- Added safer PyTorch device detection to prevent crashes on some environments. ([`67c4788`](https://github.com/vectorize-io/hindsight/commit/67c4788))
- Fixed Python packages exposing an incorrect __version__ value. ([`fccbdfe`](https://github.com/vectorize-io/hindsight/commit/fccbdfe))

## [0.4.0](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.0)

**Observations**, **Mental Models**, new **Agentic Reflect** and Directives, read the announcement.

**Features**

- Added support for providing a custom prompt for memory extraction. ([`3172e99`](https://github.com/vectorize-io/hindsight/commit/3172e99))
- Expanded the LiteLLM integration with async retain/reflect support, cleaner API, and support for tags/mission (including passing API keys correctly). ([`1d4879a`](https://github.com/vectorize-io/hindsight/commit/1d4879a))
- Added a new worker service to run background tasks at scale. ([`4c79240`](https://github.com/vectorize-io/hindsight/commit/4c79240))
- MCP retain now supports timestamps. ([`b378f68`](https://github.com/vectorize-io/hindsight/commit/b378f68))
- Added support for installing skills via `npx add-skill`. ([`ec22317`](https://github.com/vectorize-io/hindsight/commit/ec22317))

**Improvements**

- CLI retain-files now accepts more file types. ([`1eeced3`](https://github.com/vectorize-io/hindsight/commit/1eeced3))

**Bug Fixes**

- Fixed a macOS crash in the embed daemon caused by an XPC connection issue. ([`e5fc6ee`](https://github.com/vectorize-io/hindsight/commit/e5fc6ee))
- Fixed occasional extraction in the wrong language. ([`87d4a36`](https://github.com/vectorize-io/hindsight/commit/87d4a36))
- Fixed PyTorch model initialization issues that could cause startup failures (meta tensor/init problems). ([`ddaa5f5`](https://github.com/vectorize-io/hindsight/commit/ddaa5f5))


**Features**

- Add memory tags so you can label and filter memories during recall/reflect. ([`20c8f8b`](https://github.com/vectorize-io/hindsight/commit/20c8f8b))
- Allow choosing different AI providers/models per operation. ([`e6709d5`](https://github.com/vectorize-io/hindsight/commit/e6709d5))
- Add Cohere support for embeddings and reranking. ([`4de0730`](https://github.com/vectorize-io/hindsight/commit/4de0730))
- Add configurable embedding dimensions and OpenAI embeddings support. ([`70de23e`](https://github.com/vectorize-io/hindsight/commit/70de23e))
- Support custom base URLs for OpenAI-style embeddings and Cohere endpoints. ([`fa53917`](https://github.com/vectorize-io/hindsight/commit/fa53917))
- Add LiteLLM gateway support for routing LLM/embedding requests. ([`d47c8a2`](https://github.com/vectorize-io/hindsight/commit/d47c8a2))
- Add multilingual content support to improve handling and retrieval across languages. ([`c65c6a9`](https://github.com/vectorize-io/hindsight/commit/c65c6a9))
- Add delete memory bank capability. ([`4b82d2d`](https://github.com/vectorize-io/hindsight/commit/4b82d2d))
- Add backup/restore tooling for memory banks. ([`67b273d`](https://github.com/vectorize-io/hindsight/commit/67b273d))

**Improvements**

- Add retention modes to control how memories are extracted and stored. ([`fb31a35`](https://github.com/vectorize-io/hindsight/commit/fb31a35))
- Add offline (optional) database migrations to support restricted/air-gapped deployments. ([`233bd2e`](https://github.com/vectorize-io/hindsight/commit/233bd2e))
- Add database connection configuration options for more flexible deployments. ([`33fac2c`](https://github.com/vectorize-io/hindsight/commit/33fac2c))
- Load .env automatically on startup to simplify configuration. ([`c06d9b4`](https://github.com/vectorize-io/hindsight/commit/c06d9b4))
- Expose an operation ID from retain requests so async/background processing can be tracked. ([`1dacd0e`](https://github.com/vectorize-io/hindsight/commit/1dacd0e))
- Add per-request LLM token usage metrics for monitoring and cost tracking. ([`29a542d`](https://github.com/vectorize-io/hindsight/commit/29a542d))
- Add LLM call latency metrics for performance monitoring. ([`5e1f13e`](https://github.com/vectorize-io/hindsight/commit/5e1f13e))
- Include tenant in metrics labels for better multi-tenant observability. ([`1ffc2a4`](https://github.com/vectorize-io/hindsight/commit/1ffc2a4))
- Add async processing option to MCP retain tool for background retention workflows. ([`37fc7fb`](https://github.com/vectorize-io/hindsight/commit/37fc7fb))

**Bug Fixes**

- Fix extension loading in multi-worker deployments so all workers load extensions correctly. ([`f5f3fca`](https://github.com/vectorize-io/hindsight/commit/f5f3fca))
- Improve recall performance by batching recall queries. ([`5991308`](https://github.com/vectorize-io/hindsight/commit/5991308))
- Improve retrieval quality and stability for large memory banks (graph/MPFP retrieval fixes). ([`6232e69`](https://github.com/vectorize-io/hindsight/commit/6232e69))
- Fix entities list being limited to 100 entities. ([`26bf571`](https://github.com/vectorize-io/hindsight/commit/26bf571))
- Fix UI only showing the first 1000 memories. ([`67c1a42`](https://github.com/vectorize-io/hindsight/commit/67c1a42))
- Fix duplicated causal relationships and improve token usage during processing. ([`49e233c`](https://github.com/vectorize-io/hindsight/commit/49e233c))
- Improve causal link detection accuracy. ([`2a00df0`](https://github.com/vectorize-io/hindsight/commit/2a00df0))
- Make retain max completion tokens configurable to prevent truncation issues. ([`7715a51`](https://github.com/vectorize-io/hindsight/commit/7715a51))
- Fix Python SDK not sending the Authorization header, preventing authenticated requests. ([`39e3f7c`](https://github.com/vectorize-io/hindsight/commit/39e3f7c))
- Fix stats endpoint missing tenant authentication in multi-tenant setups. ([`d6ff191`](https://github.com/vectorize-io/hindsight/commit/d6ff191))
- Fix embedding dimension handling for tenant schemas in multi-tenant databases. ([`6fe9314`](https://github.com/vectorize-io/hindsight/commit/6fe9314))
- Fix Groq free-tier compatibility so requests work correctly. ([`d899d18`](https://github.com/vectorize-io/hindsight/commit/d899d18))
- Fix security vulnerability (qs / CVE-2025-15284). ([`b3becb6`](https://github.com/vectorize-io/hindsight/commit/b3becb6))
- Restore MCP tools for listing and creating memory banks. ([`9fd5679`](https://github.com/vectorize-io/hindsight/commit/9fd5679))

## [0.2.0](https://github.com/vectorize-io/hindsight/releases/tag/v0.2.0)

**Features**

- Add additional model provider support, including Anthropic Claude and LM Studio. ([`787ed60`](https://github.com/vectorize-io/hindsight/commit/787ed60))
- Add multi-bank access and new MCP tools for interacting with multiple memory banks via MCP. ([`6b5f593`](https://github.com/vectorize-io/hindsight/commit/6b5f593))
- Allow supplying custom entities when retaining memories via the retain endpoint. ([`dd59bc8`](https://github.com/vectorize-io/hindsight/commit/dd59bc8))
- Enhance the /reflect endpoint with max_tokens control and optional structured output responses. ([`d49e820`](https://github.com/vectorize-io/hindsight/commit/d49e820))


**Improvements**

- Improve local LLM support for reasoning-capable models and streamline Docker startup for local deployments. ([`eea0f27`](https://github.com/vectorize-io/hindsight/commit/eea0f27))
- Support operation validator extensions and return proper HTTP errors when validation fails. ([`ce45d30`](https://github.com/vectorize-io/hindsight/commit/ce45d30))
- Add configurable observation thresholds to control when observations are created/updated. ([`54e2df0`](https://github.com/vectorize-io/hindsight/commit/54e2df0))
- Improve graph visualization to the control plane for exploring memory relationships. ([`1a62069`](https://github.com/vectorize-io/hindsight/commit/1a62069))

**Bug Fixes**

- Fix MCP server lifecycle handling so MCP lifespan is correctly tied to the FastAPI app lifespan. ([`6b78f7d`](https://github.com/vectorize-io/hindsight/commit/6b78f7d))

## [0.1.15](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.15)

**Features**

- Add the ability to delete documents from the web UI. ([`f7ff32d`](https://github.com/vectorize-io/hindsight/commit/f7ff32d))

**Improvements**

- Improve the API health check endpoint and update the generated client APIs/types accordingly. ([`e06a612`](https://github.com/vectorize-io/hindsight/commit/e06a612))

## [0.1.14](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.14)

**Bug Fixes**

- Fixes the embedded “get-skill” installer so installing skills works correctly. ([`0b352d1`](https://github.com/vectorize-io/hindsight/commit/0b352d1))

## [0.1.13](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.13)

**Improvements**

- Improve reliability by surfacing task handler failures so retries can occur when processing fails. ([`904ea4d`](https://github.com/vectorize-io/hindsight/commit/904ea4d))
- Revamp the hindsight-embed component architecture, including a new daemon/client model and CLI updates for embedding workflows. ([`e6511e7`](https://github.com/vectorize-io/hindsight/commit/e6511e7))

**Bug Fixes**

- Fix memory retention so timestamps are correctly taken into account. ([`234d426`](https://github.com/vectorize-io/hindsight/commit/234d426))

## [0.1.12](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.12)

**Features**

- Added an extensions system for plugging in new operations/skills (including built-in tenant support). ([`2a0c490`](https://github.com/vectorize-io/hindsight/commit/2a0c490))
- Introduced the hindsight-embed tool and a native agentic skill for embedding/agent workflows. ([`da44a5e`](https://github.com/vectorize-io/hindsight/commit/da44a5e))

**Improvements**

- Improved reliability when parsing LLM JSON by retrying on parse errors and adding clearer diagnostics. ([`a831a7b`](https://github.com/vectorize-io/hindsight/commit/a831a7b))

**Bug Fixes**

- Fixed structured-output support for Ollama-based LLM providers. ([`32bca12`](https://github.com/vectorize-io/hindsight/commit/32bca12))
- Adjusted LLM validation to cap max completion tokens at 100 to prevent validation failures. ([`b94b5cf`](https://github.com/vectorize-io/hindsight/commit/b94b5cf))

## [0.1.11](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.11)

**Bug Fixes**

- Fixed the standalone Docker image and control plane standalone build process so standalone deployments build correctly. ([`2948cb6`](https://github.com/vectorize-io/hindsight/commit/2948cb6))

## [0.1.10](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.10)

*This release contains internal maintenance and infrastructure changes only.*


## [0.1.9](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.9)

**Features**

- Simplified local MCP installation and added a standalone UI option for easier setup. ([`1c6acc3`](https://github.com/vectorize-io/hindsight/commit/1c6acc3))

**Bug Fixes**

- Fixed the standalone Docker image so it builds and starts reliably. ([`b52eb90`](https://github.com/vectorize-io/hindsight/commit/b52eb90))
- Improved Docker runtime reliability by adding required system utilities (procps). ([`ae80876`](https://github.com/vectorize-io/hindsight/commit/ae80876))

## [0.1.8](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.8)

**Bug Fixes**

- Fix bank list responses when a bank has no name. ([`04f01ab`](https://github.com/vectorize-io/hindsight/commit/04f01ab))
- Fix failures when retaining memories asynchronously. ([`63f5138`](https://github.com/vectorize-io/hindsight/commit/63f5138))
- Fix a race condition in the bank selector when switching banks. ([`e468a4e`](https://github.com/vectorize-io/hindsight/commit/e468a4e))

## [0.1.7](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.7)

*This release contains internal maintenance and infrastructure changes only.*

## [0.1.6](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.6)

**Features**

- Added support for the Gemini 3 Pro and GPT-5.2 models. ([`bb1f9cb`](https://github.com/vectorize-io/hindsight/commit/bb1f9cb))
- Added a local MCP server option for running/connecting to Hindsight via MCP without a separate remote service. ([`7dd6853`](https://github.com/vectorize-io/hindsight/commit/7dd6853))

**Improvements**

- Updated the Postgres/pg0 dependency to a newer 0.11.x series for improved compatibility and stability. ([`47be07f`](https://github.com/vectorize-io/hindsight/commit/47be07f))

## [0.1.5](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.5)

**Features**

- Added LiteLLM integration so Hindsight can capture and manage memories from LiteLLM-based LLM calls. ([`dfccbf2`](https://github.com/vectorize-io/hindsight/commit/dfccbf2))
- Added an optional graph-based retriever (MPFP) to improve recall by leveraging relationships between memories. ([`7445cef`](https://github.com/vectorize-io/hindsight/commit/7445cef))

**Improvements**

- Switched the embedded Postgres layer to pg0-embedded for a smoother local/standalone experience. ([`94c2b85`](https://github.com/vectorize-io/hindsight/commit/94c2b85))

**Bug Fixes**

- Fixed repeated retries on 400 errors from the LLM, preventing unnecessary request loops and failures. ([`70983f5`](https://github.com/vectorize-io/hindsight/commit/70983f5))
- Fixed recall trace visualization in the control plane so search/recall debugging displays correctly. ([`922164e`](https://github.com/vectorize-io/hindsight/commit/922164e))
- Fixed the CLI installer to make installation more reliable. ([`158a6aa`](https://github.com/vectorize-io/hindsight/commit/158a6aa))
- Updated Next.js to patch security vulnerabilities (CVE-2025-55184, CVE-2025-55183). ([`f018cc5`](https://github.com/vectorize-io/hindsight/commit/f018cc5))

## [0.1.3](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.3)

**Improvements**

- Improved CLI and UI branding/polish, including new banner/logo assets and updated interface styling. ([`fa554b8`](https://github.com/vectorize-io/hindsight/commit/fa554b8))


## [0.1.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.2)

**Bug Fixes**

- Fixed the standalone Docker image so it builds/runs correctly. ([`1056a20`](https://github.com/vectorize-io/hindsight/commit/1056a20))

## Integration Changelogs

| Integration | Package | Description |
|---|---|---|
| [LiteLLM](integrations/litellm.md) | `hindsight-litellm` | Universal LLM memory via LiteLLM (100+ providers) |
| [Pydantic AI](integrations/pydantic-ai.md) | `hindsight-pydantic-ai` | Persistent memory tools for Pydantic AI agents |
| [CrewAI](integrations/crewai.md) | `hindsight-crewai` | Persistent memory for CrewAI agents |
| [AI SDK](integrations/ai-sdk.md) | `@vectorize-io/hindsight-ai-sdk` | Memory integration for Vercel AI SDK |
| [Chat SDK](integrations/chat.md) | `@vectorize-io/hindsight-chat` | Memory integration for Vercel Chat SDK |
| [OpenClaw](integrations/openclaw.md) | `@vectorize-io/hindsight-openclaw` | Hindsight memory plugin for OpenClaw |
