/**
 * opencode v2 (`opencode2`) plugin entrypoint.
 *
 * opencode v1 and v2 share `~/.config/opencode/opencode.json` and v1 rejects the whole file when it
 * sees v2's `plugins` key, so both hosts have to load from the ONE package path already registered
 * in the user's `plugin` array. They resolve a plugin directory differently, which is what makes
 * that work: v1 follows package.json `main` (→ dist/index.js, the v1 plugin), while v2 ignores
 * `main` and loads `<dir>/index.js` — this file. See src/opencode2.ts.
 */
export { default, HindsightOpencode2Plugin } from "./dist/opencode2.js";
