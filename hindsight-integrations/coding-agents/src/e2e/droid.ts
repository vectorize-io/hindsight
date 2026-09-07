import type { HarnessDockerSetup } from "./harness";

/**
 * Factory Droid — `droid exec` runs one prompt non-interactively and prints the reply on stdout.
 *
 * Driven through the stub model, and unusually for a subscription CLI it needs NO account to do it.
 * Droid's BYOK path (a `customModels` entry in ~/.factory/settings.json) bypasses Factory's backend
 * entirely: with one configured, `droid exec` runs to completion with no login on disk at all. That
 * is what makes this harness containerisable — its real credential is an encrypted
 * `auth.v2.loginkeychain` blob that cannot be handed to a Linux container, which is exactly why the
 * PR that added the harness shipped without an E2E.
 *
 * The route is baked into the image (e2e/droid-e2e-exec.sh) rather than passed as environment, for
 * the same reason dsh needs an overlay file: Droid takes its base URL from settings.json and expands
 * `${VAR}` only in `apiKey`, never in `baseUrl` — a `${...}` there is sent to the network verbatim
 * and the run dies with "Exec failed". The entry script renders the file from this setup's
 * environment instead, and marks /workspace trusted so a fresh container does not stop on the
 * workspace-trust gate.
 *
 * `injectsIntoModel` is false because `droid exec` fires SessionStart and Stop but NOT
 * UserPromptSubmit — verified by probing all four events against a real Droid 0.213.0. Recall is
 * injected by the UserPromptSubmit hook, so the non-interactive path cannot demonstrate injection no
 * matter how it is driven (the interactive TUI does fire it, and does inject). Retention — the Stop
 * write-back, the transcript parser, the bank wiring — is fully asserted, exactly as for Grok Build.
 */
export const factoryDroidDockerSetup: HarnessDockerSetup = {
  name: "factory-droid",
  hindsightHarness: "factory-droid",
  installCommand: "hindsight-coding-agents install factory-droid",
  stubModelEnv: (baseUrl) => ({
    HINDSIGHT_STUB_BASE_URL: `${baseUrl}/v1`,
  }),
  injectsIntoModel: false,
  command: (prompt) => ["hindsight-droid-e2e-exec", prompt],
};
