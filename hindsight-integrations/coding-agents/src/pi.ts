/**
 * pi's extension entrypoint (`pi`, npm `@earendil-works/pi-coding-agent`).
 *
 * pi loads extensions listed in the `extensions` array of `~/.pi/agent/settings.json` and calls each
 * one's default export with its `pi` API. The whole adapter lives in src/harness/pi-extension.ts.
 *
 * pi has no hooks system, which is why it is a persistent-extension harness rather than a
 * hook-binary harness like claude-code/cursor/codex.
 *
 * The harness it reports ("pi") selects the `harnesses.pi` config section and feeds `{harness}` bank
 * templating.
 */
import { createPiExtension, type ExtensionFactory } from "./harness/pi-extension";

const extension: ExtensionFactory = createPiExtension("pi");

export { extension };
export default extension;
