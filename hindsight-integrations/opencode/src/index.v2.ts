/**
 * Hindsight OpenCode V2 plugin entrypoint.
 *
 * OpenCode V2 uses a different plugin API from V1, so it gets its own entry
 * rather than restructuring the default export (the V1 entry must stay
 * function-only for the legacy loader). Add it as a separate plugin:
 *
 * ```jsonc
 * // opencode.json (OpenCode v2)
 * { "plugins": ["@vectorize-io/opencode-hindsight/v2"] }
 * ```
 *
 * With options:
 *
 * ```jsonc
 * {
 *   "plugins": [
 *     { "package": "@vectorize-io/opencode-hindsight/v2",
 *       "options": { "bankId": "my-bank" } }
 *   ]
 * }
 * ```
 */

export { HindsightV2Plugin as default } from "./v2.js";
