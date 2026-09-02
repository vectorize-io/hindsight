import type { HarnessDockerSetup } from "./harness";

/**
 * DeepAgents Dcode — `-n` runs one prompt non-interactively and exits.
 *
 * Driven through the stub model. Dcode has no vendor account of its own: it is model-agnostic and
 * reads whichever provider key is in the environment, so `OPENAI_BASE_URL` retargets it at the stub
 * with no credential to mount. The model must be given as `provider:model` — a bare stub id makes
 * Dcode stop with "Unable to infer a model provider" before it ever calls out.
 *
 * Unlike every other harness here, the install is not a config patch: `install dcode` registers a
 * local marketplace and shells out to Dcode's own `plugin install`, so this E2E is also the only
 * check that the published tarball's `plugin.json` + `hooks/` survive packaging and that Dcode's
 * plugin manager accepts them.
 */
export const dcodeDockerSetup: HarnessDockerSetup = {
  name: "dcode",
  hindsightHarness: "dcode",
  installCommand: "hindsight-coding-agents install dcode",
  stubModelEnv: (baseUrl) => ({
    OPENAI_BASE_URL: `${baseUrl}/v1`,
    OPENAI_API_KEY: "hindsight-e2e",
  }),
  command: (prompt) => ["dcode", "-n", prompt, "-M", "openai:hindsight-e2e-stub"],
};
