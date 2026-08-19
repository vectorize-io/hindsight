/**
 * Docker E2E setups for every supported CLI harness.
 *
 * Each entry is small and declarative: where the host keeps that CLI's SUBSCRIPTION credentials,
 * where the CLI expects them inside the container, how to wire Hindsight into it, and how to drive
 * it non-interactively. Everything else — the seeded bank, the git fixture, the assertions — is
 * shared in ./harness.
 *
 * Credentials are subscription logins (the same ones a developer uses locally), not API keys, so
 * every path defaults to that CLI's real credential location and stays overridable via
 * `<HARNESS>_E2E_AUTH_PATH` for machines and CI runners that keep them elsewhere. A harness whose
 * credential is absent is SKIPPED, never failed — no one has all nine subscriptions.
 *
 * Antigravity is intentionally absent: its harness E2E is covered separately.
 */
import { homedir } from "node:os";
import { join } from "node:path";
import type { HarnessDockerSetup } from "./harness";
import { codexDockerSetup } from "./codex";

const home = (...parts: string[]) => join(homedir(), ...parts);

/** Host credential location, overridable per harness for CI runners that stage secrets elsewhere. */
const authPath = (envVar: string, ...parts: string[]) => process.env[envVar] || home(...parts);

/**
 * opencode — persistent plugin. `run` executes one prompt non-interactively and prints the reply,
 * which the shared runner captures from stdout.
 */
export const opencodeDockerSetup: HarnessDockerSetup = {
  name: "opencode",
  hindsightHarness: "opencode",
  credentialPath: () =>
    authPath("OPENCODE_E2E_AUTH_PATH", ".local", "share", "opencode", "auth.json"),
  credentialTarget: "/root/.local/share/opencode/auth.json",
  installCommand: "hindsight-coding-agents install opencode",
  command: (prompt) => ["opencode", "run", prompt],
};

/**
 * Kilo CLI — an opencode fork, so same shape; its own plugin entry.
 *
 * Borrows the OPENCODE subscription file. Kilo's own `auth.json` is account-bound and reports zero
 * stored credentials; because Kilo is an opencode fork its auth file has the same format, so
 * opencode's credentials drop straight in. (Codex's `auth.json` does NOT work here — Kilo ignores
 * it, falls back to a default model and stops with "You need to sign in to use this model".)
 */
export const kiloDockerSetup: HarnessDockerSetup = {
  name: "kilo",
  hindsightHarness: "kilo",
  credentialPath: () => authPath("KILO_E2E_AUTH_PATH", ".local", "share", "opencode", "auth.json"),
  credentialTarget: "/root/.local/share/kilo/auth.json",
  installCommand: "hindsight-coding-agents install kilo",
  command: (prompt) => ["kilo", "run", prompt],
};

/**
 * Claude Code — hook harness. `--print` is its non-interactive mode.
 *
 * Deliberately WITHOUT `--dangerously-skip-permissions`: the container runs as root and Claude
 * refuses that flag under root ("cannot be used with root/sudo privileges"), exiting before it ever
 * calls a model. The E2E prompt only asks a question, so no tool permissions are needed anyway.
 *
 * Driven through the stub model: macOS keeps the subscription in the Keychain, so there is no file
 * to hand a container. `ANTHROPIC_BASE_URL` retargets the API, which is enough to exercise our hook
 * lifecycle without an account.
 */
export const claudeCodeDockerSetup: HarnessDockerSetup = {
  name: "claude-code",
  hindsightHarness: "claude-code",
  unsupported:
    "hangs to the timeout with the stub serving 0 requests: ANTHROPIC_BASE_URL alone does not get " +
    "it to the model, and it stalls before the first request — apparently waiting on login / " +
    "onboarding that a fresh container has never completed. (An earlier attempt also showed " +
    "--dangerously-skip-permissions is rejected under root, which is why that flag is absent.) " +
    "The macOS subscription lives in the Keychain, so there is no credential file to mount either.",
  installCommand: "hindsight-coding-agents install claude-code",
  stubModelEnv: (baseUrl) => ({
    ANTHROPIC_BASE_URL: baseUrl,
    ANTHROPIC_API_KEY: "hindsight-e2e",
  }),
  command: (prompt) => ["claude", "--print", prompt],
};

/**
 * Cursor CLI — `-p` prints the reply for scripts.
 *
 * Driven through the stub model: Cursor's session is account-bound, so copying `cli-config.json`
 * into a container still yields "Authentication required". `--endpoint`/`CURSOR_API_ENDPOINT` is
 * its documented way to target a different API, which lets the E2E exercise our hooks without an
 * account.
 */
export const cursorDockerSetup: HarnessDockerSetup = {
  name: "cursor-cli",
  hindsightHarness: "cursor-cli",
  installCommand: "hindsight-coding-agents install cursor-cli",
  unsupported:
    "cursor-agent never contacts a custom endpoint — the stub served 0 requests via both " +
    "CURSOR_API_ENDPOINT and the --endpoint/--api-key flags, and the run hangs to the timeout " +
    "instead. It appears to authenticate against Cursor's own service before any model call. " +
    "Its account session is also machine-bound, so mounting cli-config.json yields " +
    '"Authentication required". Re-enable by clearing this field once either path works.',
  stubModelEnv: (baseUrl) => ({ CURSOR_API_ENDPOINT: baseUrl, CURSOR_API_KEY: "hindsight-e2e" }),
  command: (prompt, { stubUrl }) => [
    "cursor-agent",
    "-p",
    "--force",
    ...(stubUrl ? ["--endpoint", stubUrl, "--api-key", "hindsight-e2e"] : []),
    prompt,
  ],
};

/**
 * GitHub Copilot CLI, driven through the stub model.
 *
 * Its GitHub token lives in the system keyring, so there is nothing to mount. Copilot's own BYOK
 * mode solves this exactly: setting `COPILOT_PROVIDER_BASE_URL` targets a custom provider and, per
 * its docs, "GitHub authentication is not required when using a custom provider". A model must be
 * named for BYOK, hence COPILOT_MODEL.
 */
export const copilotDockerSetup: HarnessDockerSetup = {
  name: "copilot-cli",
  hindsightHarness: "copilot-cli",
  unsupported:
    "hangs to the timeout without ever calling the model — the stub served 0 requests despite the " +
    "documented BYOK variables (COPILOT_PROVIDER_BASE_URL/TYPE/API_KEY + COPILOT_MODEL), which are " +
    "supposed to make GitHub auth unnecessary. It stalls before the first request, so the block is " +
    "upstream of the provider override, not a response-shape mismatch. Its real token lives in the " +
    "system keyring, so there is no file to mount either.",
  installCommand: "hindsight-coding-agents install copilot-cli",
  stubModelEnv: (baseUrl) => ({
    COPILOT_PROVIDER_BASE_URL: `${baseUrl}/v1`,
    COPILOT_PROVIDER_TYPE: "openai",
    COPILOT_PROVIDER_API_KEY: "hindsight-e2e",
    COPILOT_MODEL: "hindsight-e2e-stub",
  }),
  command: (prompt) => ["copilot", "-p", prompt, "--allow-all-tools"],
};

/**
 * Grok Build (xAI) — `-p/--single` runs one prompt and exits.
 *
 * Retention-only: Grok's prompt hook is passive (it ignores hook stdout), so injected memory can
 * never reach the model — a platform limitation, not a wiring defect. Asserting injection here
 * would keep the harness permanently red for something no change on our side can fix.
 */
export const grokDockerSetup: HarnessDockerSetup = {
  name: "grok-build",
  hindsightHarness: "grok-build",
  credentialPath: () => authPath("GROK_E2E_AUTH_PATH", ".grok", "auth.json"),
  credentialTarget: "/root/.grok/auth.json",
  installCommand: "hindsight-coding-agents install grok-build",
  injectsIntoModel: false,
  command: (prompt) => ["grok", "-p", prompt],
};

/**
 * Devin CLI — `-p` takes the prompt inline and exits.
 *
 * Credentials live in `~/.local/share/devin/credentials.toml`, NOT the `~/.config/devin/config.json`
 * that holds settings — mounting the latter yields "Not logged in" from an otherwise healthy run.
 */
export const devinDockerSetup: HarnessDockerSetup = {
  name: "devin-cli",
  hindsightHarness: "devin-cli",
  unsupported:
    "authenticates and runs cleanly (✓ Organization, no trust error) but its reply never reaches " +
    "us: the captured stdout holds only the runner's npm output, and it writes no " +
    "/results/last-message.txt. Needs a way to capture Devin's answer — an output-file flag or " +
    "stderr capture — before the injection assertion can mean anything.",
  credentialPath: () =>
    authPath("DEVIN_E2E_AUTH_PATH", ".local", "share", "devin", "credentials.toml"),
  credentialTarget: "/root/.local/share/devin/credentials.toml",
  installCommand: "hindsight-coding-agents install devin-cli",
  // A fresh container has never trusted the workspace, and Devin's own help is explicit that
  // non-interactive mode "cannot show the trust prompt and fails in an untrusted directory".
  command: (prompt) => ["devin", "-p", prompt, "--respect-workspace-trust", "false"],
};

/**
 * Cline CLI — the prompt is positional and act mode auto-approves tools by default. Like Copilot,
 * its state is a directory (SQLite session/connector stores).
 */
export const clineDockerSetup: HarnessDockerSetup = {
  name: "cline-cli",
  hindsightHarness: "cline-cli",
  credentialPath: () => authPath("CLINE_E2E_AUTH_PATH", ".cline"),
  credentialTarget: "/root/.cline",
  installCommand: "hindsight-coding-agents install cline-cli",
  command: (prompt) => ["cline", "--auto-approve", "true", "--cwd", "/workspace", prompt],
};

/**
 * Prime Agent — extension host. `-p` runs one prompt non-interactively and prints the reply.
 * Auth is the whole `~/.prime/agent` directory (auth.json plus the kernel venv it provisions on
 * first login), so the mount is the directory rather than a single credential file.
 */
export const primeAgentDockerSetup: HarnessDockerSetup = {
  name: "prime-agent",
  hindsightHarness: "prime-agent",
  credentialPath: () => authPath("PRIME_AGENT_E2E_AUTH_PATH", ".prime", "agent", "auth.json"),
  credentialTarget: "/root/.prime/agent/auth.json",
  installCommand: "hindsight-coding-agents install prime-agent",
  command: (prompt) => ["prime-agent", "-p", prompt],
};

/**
 * DeepSeek Harness — a native Cordis plugin, driven through the one-shot `headless` profile, which
 * prints the final answer on stdout for the shared runner to capture.
 *
 * Driven through the stub model, but for a different reason from the CLIs above: dsh authenticates
 * fine with a plain API key — it simply must not need a paid DeepSeek account to run in CI. Unlike
 * those CLIs, dsh takes no base-URL environment variable, so the route is a composition overlay
 * baked into the image (e2e/dsh-stub-model.cordis.yml) that reads the stub's ephemeral URL from the
 * environment this setup supplies.
 */
export const dshDockerSetup: HarnessDockerSetup = {
  name: "dsh",
  hindsightHarness: "dsh",
  installCommand: "hindsight-coding-agents install dsh",
  stubModelEnv: (baseUrl) => ({
    HINDSIGHT_STUB_BASE_URL: `${baseUrl}/v1`,
    HINDSIGHT_STUB_KEY: "hindsight-e2e",
  }),
  command: (prompt) => [
    "dsh",
    "--profile",
    "headless",
    "--patch",
    "/dsh/stub-model.cordis.yml",
    prompt,
  ],
};

/** Every harness the unified Docker E2E can drive, in a stable order. */
export const ALL_HARNESS_SETUPS: HarnessDockerSetup[] = [
  codexDockerSetup,
  opencodeDockerSetup,
  kiloDockerSetup,
  claudeCodeDockerSetup,
  cursorDockerSetup,
  copilotDockerSetup,
  grokDockerSetup,
  devinDockerSetup,
  clineDockerSetup,
  primeAgentDockerSetup,
  dshDockerSetup,
];
