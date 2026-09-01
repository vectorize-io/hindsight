/**
 * Prime Agent's extension entrypoint (`prime-agent`, npm `prime-agent`).
 *
 * Prime Agent (PrimeIntellect) is a FORK OF PI: same extension API, same `extensions` array in
 * `settings.json` (under `~/.prime/agent/` rather than `~/.pi/agent/`), same `before_agent_start` /
 * `agent_end` events and `registerTool`. So it needs no adapter of its own and drives the exact pi
 * runtime — see src/harness/pi-extension.ts.
 *
 * Prime Agent has no hooks system, which is why it is a persistent-extension harness rather than a
 * hook-binary harness like claude-code/cursor/codex.
 *
 * This entry exists — rather than pointing Prime Agent at dist/pi.js — purely so the harness reports
 * as "prime-agent": that selects the `harnesses.prime-agent` config section, feeds `{harness}` bank
 * templating, and keeps Prime Agent sessions attributable in diagnostics separately from pi's.
 */
import { createPiExtension, type ExtensionFactory } from "./harness/pi-extension";

const extension: ExtensionFactory = createPiExtension("prime-agent");

export { extension };
export default extension;
