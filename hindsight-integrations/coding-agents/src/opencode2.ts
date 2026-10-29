/**
 * opencode2 entrypoint — the plugin opencode v2 (`@opencode/cli`, binary `opencode2`) loads.
 *
 * It is a SEPARATE entry from src/index.ts because v2's plugin contract shares nothing with v1's:
 * v1 loads a function returning named hooks, v2 loads a `{id, setup(ctx)}` object. Handing either
 * host the other's export loads a plugin that registers nothing and reports no error.
 *
 * The two hosts nonetheless share one registered path. opencode v1 and v2 read the SAME
 * `~/.config/opencode/opencode.json`, and v1 REJECTS the whole file when it sees v2's `plugins` key
 * ("Configuration is invalid … Unrecognized key: plugins"), so a second config entry is not an
 * option. What makes one entry serve both is that they resolve a plugin DIRECTORY differently:
 * v1 follows `package.json` `main` (→ dist/index.js, the v1 plugin), v2 looks for `<dir>/index.js`
 * and ignores `main`. The package therefore ships a root `index.js` re-exporting this file, and the
 * package root already registered in the user's `plugin` array — which v2 migrates to `plugins` for
 * itself — drives the right plugin under each host. Written against opencode 1.18.9 and the v2
 * beta, then re-checked on stable 2.0.3, whose loader still resolves `<dir>/index.js` and whose
 * live session registers the native tools and answers from injected memory. From 2.2 v2 no longer
 * looks for `<dir>/index.js`: a directory whose package.json has a `name` and an `exports` map is
 * resolved by name (see below), which reaches the same file through `exports["./server"]` (#6138).
 *
 * The harness reports as "opencode2" rather than "opencode": that selects the `harnesses.opencode2`
 * config section and keeps v2 sessions attributable in diagnostics separately from v1's. The bank
 * is unaffected — the default id template (`coding-agent::{gitProject}`) is harness-neutral, so
 * both hosts compound into the same repo memory.
 *
 * The body lives in harness/opencode2.ts; everything behind it is the shared RuntimeCore.
 */
import { createOpencode2PluginEntry } from "./harness/opencode2";
import { createPluginEntry } from "./harness/plugin-entry";

/**
 * The v2 definition, plus v1's plugin function as `server`. A host that looks this package up by
 * NAME — v2 given a non-absolute entry, v2 2.2+ given any package directory, or v1 1.18+ — reads
 * package.json `exports["./server"]` (→ the root index.js → this file) before `main`. Without it v2
 * fell through to `exports["."]`, got v1's function and refused to load the plugin (#5823); v1 1.18+
 * reads the same export and accepts an object default that has `server()`. v2's schema ignores the
 * extra key, and older v1 never reads `./server`, so it still follows `main` to dist/index.js.
 */
const HindsightOpencode2Plugin = {
  ...createOpencode2PluginEntry("opencode2"),
  server: createPluginEntry("opencode"),
};

export default HindsightOpencode2Plugin;
export { HindsightOpencode2Plugin };
