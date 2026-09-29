/**
 * Kilo Code CLI entrypoint (`kilo`, npm `@kilocode/cli`).
 *
 * Kilo is a FORK OF OPENCODE: same config schema (it still reads legacy `opencode.json`), the same
 * `plugin` array, and `@kilocode/plugin` re-exports opencode's Plugin/Hooks/PluginInput types. So
 * Kilo needs no adapter of its own — it loads this default export as a persistent plugin and drives
 * the exact opencode runtime (recall+inject, cold seed, native hindsight_* tools, write-back).
 *
 * Kilo has NO hooks system, which is why it is a plugin harness rather than a hook-binary harness
 * like claude-code/cursor/codex.
 *
 * This entry exists — rather than pointing Kilo at dist/index.js — purely so the harness reports as
 * "kilo": that selects the `harnesses.kilo` config section, feeds `{harness}` bank templating, and
 * keeps Kilo sessions attributable in diagnostics separately from opencode's.
 */
import { createPluginEntry } from "./harness/plugin-entry";

const HindsightKiloPlugin = createPluginEntry("kilo");

export default HindsightKiloPlugin;
export { HindsightKiloPlugin };
