import type { HarnessDockerSetup } from "./harness";

/**
 * ZCode — hook harness. `--prompt` runs one prompt non-interactively and prints the reply.
 *
 * Driven through the stub model, and like Factory Droid it needs NO vendor account to do it: a
 * `provider` entry in `~/.zcode/cli/config.json` with `kind: "anthropic"` and a `baseURL` replaces
 * Z.AI's backend outright, and `zcode --prompt` then runs with no login on disk. (Verified against
 * zcode 0.16.5 / zcode-app-cli 3.11.2-22, which vendors the same agent runtime ZCode Desktop ships.)
 *
 * The route is rendered by an entry script baked into the image rather than passed as environment,
 * because ZCode reads its provider from that config file and nothing else — and, unlike Droid's,
 * that file is the SAME one the installer just wrote hooks and the MCP server into. The script
 * therefore merges rather than overwrites; clobbering it would delete the very wiring under test.
 *
 * Fully asserted, injection included: unlike grok-build (passive hook) and factory-droid/qwen-code
 * (no UserPromptSubmit, or an echo the system prompt crowds out), ZCode fires all three events in
 * headless mode and its `hookSpecificOutput.additionalContext` reaches the model.
 */
export const zcodeDockerSetup: HarnessDockerSetup = {
  name: "zcode",
  hindsightHarness: "zcode",
  installCommand: "hindsight-coding-agents install zcode",
  stubModelEnv: (baseUrl) => ({ HINDSIGHT_STUB_BASE_URL: baseUrl }),
  command: (prompt) => ["hindsight-zcode-e2e-exec", prompt],
};
