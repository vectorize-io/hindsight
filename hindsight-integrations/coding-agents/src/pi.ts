/**
 * pi's extension entrypoint (`pi`, npm `@earendil-works/pi-coding-agent`).
 *
 * pi loads extensions listed in the `extensions` array of `~/.pi/agent/settings.json` and calls each
 * one's default export with its `pi` API. The whole adapter lives in src/harness/pi-extension.ts,
 * shared with Prime Agent — pi's fork — which registers the same default-export shape from its own
 * settings file.
 *
 * pi has no hooks system, which is why it is a persistent-extension harness rather than a
 * hook-binary harness like claude-code/cursor/codex.
 *
 * This entry exists — rather than pointing pi at dist/prime-agent.js — purely so the harness reports
 * as "pi": that selects the `harnesses.pi` config section, feeds `{harness}` bank templating, and
 * keeps pi sessions attributable in diagnostics separately from Prime Agent's.
 */
import { createPiExtension, type ExtensionFactory } from "./harness/pi-extension";

const extension: ExtensionFactory = createPiExtension("pi");

export { extension };
export default extension;
