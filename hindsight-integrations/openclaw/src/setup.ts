#!/usr/bin/env node
/**
 * Setup wizard for the Hindsight OpenClaw plugin.
 *
 * Two modes of operation:
 *
 *   1. Interactive — no flags, just run `hindsight-openclaw-setup`. Walks the
 *      user through picking a mode (Cloud / External API / Embedded daemon)
 *      via @clack/prompts and writes openclaw.json.
 *
 *   2. Non-interactive — pass `--mode cloud|api|embedded` plus the relevant
 *      flags for that mode. No prompts, intended for CI and scripted installs.
 *
 * Scanner-safe: does not import subprocess APIs and does not read environment
 * variables directly. Pure config manipulation lives in setup-lib.ts.
 */

import * as p from "@clack/prompts";
import { realpathSync } from "fs";
import { resolve } from "path";
import { fileURLToPath } from "url";
import {
  DEFAULT_OPENCLAW_CONFIG_PATH,
  HINDSIGHT_CLOUD_URL,
  NO_KEY_PROVIDERS,
  type ApiSetupInput,
  type CloudSetupInput,
  type EmbeddedSetupInput,
  type SetupMode,
  applyApiMode,
  applyCloudMode,
  applyEmbeddedMode,
  defaultApiKeyEnvVar,
  ensurePluginConfig,
  isValidEnvVarName,
  loadConfig,
  saveConfig,
  summarizeApi,
  summarizeCloud,
  summarizeEmbedded,
} from "./setup-lib.js";

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

export interface ParsedCliArgs {
  help: boolean;
  configPath?: string;
  mode?: SetupMode;
  apiUrl?: string;
  token?: string;
  tokenEnv?: string;
  noToken: boolean;
  provider?: string;
  apiKey?: string;
  apiKeyEnv?: string;
  model?: string;
  positional?: string;
}

function usage(): string {
  return [
    "Usage: hindsight-openclaw-setup [options] [config-path]",
    "",
    "Interactive mode (no flags): walks through a TUI picker for Cloud /",
    "External API / Embedded daemon and writes the resulting plugin config",
    `to ${DEFAULT_OPENCLAW_CONFIG_PATH} (or the positional config-path arg).`,
    "",
    "Non-interactive mode: pass --mode and the relevant flags to skip the",
    "TUI. Suitable for CI and scripted setups.",
    "",
    "Options:",
    "  --config-path <path>    Path to openclaw.json (default: ~/.openclaw/openclaw.json)",
    "  --mode <mode>           cloud | api | embedded (enables non-interactive mode)",
    "",
    "Cloud mode (must pass exactly one of --token or --token-env):",
    `  --api-url <url>         Override the Hindsight Cloud URL (default: ${HINDSIGHT_CLOUD_URL})`,
    "  --token <value>         Store the token inline in openclaw.json (simple)",
    "  --token-env <VAR>       Reference an env var instead — resolved at startup",
    "                          via SecretRef (keeps the secret off disk)",
    "",
    "External API mode (token is optional; pass at most one of --token / --token-env / --no-token):",
    "  --api-url <url>         Hindsight API URL (required)",
    "  --token <value>         Inline token value in openclaw.json",
    "  --token-env <VAR>       Env var holding the token (SecretRef)",
    "  --no-token              Explicitly disable token auth",
    "",
    "Embedded mode (for providers that need a key, pass exactly one of --api-key or --api-key-env):",
    `  --provider <id>         LLM provider: ${["openai", "anthropic", "gemini", "groq", ...NO_KEY_PROVIDERS].join(" | ")}`,
    "  --api-key <value>       Store the LLM API key inline in openclaw.json",
    "  --api-key-env <VAR>     Env var holding the LLM API key (SecretRef)",
    "  --model <id>            Optional model override (otherwise uses the provider default)",
    "",
    "  -h, --help              Show this help",
    "",
    "Examples:",
    "  hindsight-openclaw-setup",
    "  hindsight-openclaw-setup --mode cloud --token hsk_...",
    "  hindsight-openclaw-setup --mode cloud --token-env HINDSIGHT_CLOUD_TOKEN",
    "  hindsight-openclaw-setup --mode api --api-url https://mcp.hindsight.example.com --no-token",
    "  hindsight-openclaw-setup --mode embedded --provider openai --api-key-env OPENAI_API_KEY",
    "  hindsight-openclaw-setup --mode embedded --provider claude-code",
  ].join("\n");
}

export function parseCliArgs(argv: string[]): ParsedCliArgs {
  const args: ParsedCliArgs = { help: false, noToken: false };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const value = argv[++i];
      if (value === undefined) {
        throw new Error(`missing value for ${arg}`);
      }
      return value;
    };

    switch (arg) {
      case "-h":
      case "--help":
        args.help = true;
        break;
      case "--config-path":
        args.configPath = next();
        break;
      case "--mode": {
        const value = next();
        if (value !== "cloud" && value !== "api" && value !== "embedded") {
          throw new Error(`invalid --mode: ${value} (expected cloud | api | embedded)`);
        }
        args.mode = value;
        break;
      }
      case "--api-url":
        args.apiUrl = next();
        break;
      case "--token":
        args.token = next();
        break;
      case "--token-env":
        args.tokenEnv = next();
        break;
      case "--no-token":
        args.noToken = true;
        break;
      case "--provider":
        args.provider = next();
        break;
      case "--api-key":
        args.apiKey = next();
        break;
      case "--api-key-env":
        args.apiKeyEnv = next();
        break;
      case "--model":
        args.model = next();
        break;
      default:
        if (arg.startsWith("-")) {
          throw new Error(`unknown argument: ${arg}`);
        }
        if (args.positional) {
          throw new Error(`unexpected extra positional argument: ${arg}`);
        }
        args.positional = arg;
    }
  }

  return args;
}

// ---------------------------------------------------------------------------
// Non-interactive execution
// ---------------------------------------------------------------------------

function buildCloudInput(args: ParsedCliArgs): CloudSetupInput {
  if (!args.token && !args.tokenEnv) {
    throw new Error("--mode cloud requires --token <value> or --token-env <VAR>");
  }
  if (args.token && args.tokenEnv) {
    throw new Error("--token and --token-env are mutually exclusive — pick one");
  }
  if (args.tokenEnv && !isValidEnvVarName(args.tokenEnv)) {
    throw new Error(`--token-env must be an UPPER_SNAKE_CASE env var name, got: ${args.tokenEnv}`);
  }
  return {
    apiUrl: args.apiUrl,
    token: args.token,
    tokenEnvVar: args.tokenEnv,
  };
}

function buildApiInput(args: ParsedCliArgs): ApiSetupInput {
  if (!args.apiUrl) {
    throw new Error("--mode api requires --api-url <url>");
  }
  const credFlagCount = (args.token ? 1 : 0) + (args.tokenEnv ? 1 : 0) + (args.noToken ? 1 : 0);
  if (credFlagCount > 1) {
    throw new Error(
      "--token, --token-env, and --no-token are mutually exclusive — pick at most one"
    );
  }
  if (args.tokenEnv && !isValidEnvVarName(args.tokenEnv)) {
    throw new Error(`--token-env must be an UPPER_SNAKE_CASE env var name, got: ${args.tokenEnv}`);
  }
  return {
    apiUrl: args.apiUrl,
    token: args.token,
    tokenEnvVar: args.tokenEnv,
  };
}

function buildEmbeddedInput(args: ParsedCliArgs): EmbeddedSetupInput {
  if (!args.provider) {
    throw new Error("--mode embedded requires --provider <id>");
  }
  if (args.apiKey && args.apiKeyEnv) {
    throw new Error("--api-key and --api-key-env are mutually exclusive — pick one");
  }
  const needsKey = !NO_KEY_PROVIDERS.has(args.provider);
  if (needsKey) {
    if (!args.apiKey && !args.apiKeyEnv) {
      throw new Error(
        `--provider ${args.provider} requires --api-key <value> or --api-key-env <VAR> (providers that need no key: ${[...NO_KEY_PROVIDERS].join(", ")})`
      );
    }
    if (args.apiKeyEnv && !isValidEnvVarName(args.apiKeyEnv)) {
      throw new Error(
        `--api-key-env must be an UPPER_SNAKE_CASE env var name, got: ${args.apiKeyEnv}`
      );
    }
  }
  return {
    llmProvider: args.provider,
    apiKey: args.apiKey,
    apiKeyEnvVar: args.apiKeyEnv,
    llmModel: args.model,
  };
}

export async function runNonInteractive(
  args: ParsedCliArgs,
  configPath: string
): Promise<{ summary: string; configPath: string }> {
  if (!args.mode) {
    throw new Error("runNonInteractive called without --mode");
  }

  const cfg = await loadConfig(configPath);
  const pluginConfig = ensurePluginConfig(cfg);

  let summary: string;
  if (args.mode === "cloud") {
    const input = buildCloudInput(args);
    applyCloudMode(pluginConfig, input);
    summary = summarizeCloud(input);
  } else if (args.mode === "api") {
    const input = buildApiInput(args);
    applyApiMode(pluginConfig, input);
    summary = summarizeApi(input);
  } else {
    const input = buildEmbeddedInput(args);
    applyEmbeddedMode(pluginConfig, input);
    summary = summarizeEmbedded(input);
  }

  await saveConfig(configPath, cfg);
  return { summary, configPath };
}

// ---------------------------------------------------------------------------
// Interactive (TUI) execution
// ---------------------------------------------------------------------------

const validateEnvVar = (value: string | undefined): string | undefined =>
  isValidEnvVarName(value) ? undefined : "Must be an UPPER_SNAKE_CASE env var name";

const validateRequired =
  (msg: string) =>
  (value: string | undefined): string | undefined =>
    value && value.trim().length > 0 ? undefined : msg;

function assertNotCancelled<T>(value: T | symbol): asserts value is T {
  if (p.isCancel(value)) {
    p.cancel("Setup cancelled.");
    process.exit(1);
  }
}

async function promptCloud(pluginConfig: Record<string, unknown>): Promise<string> {
  const useDefaultUrl = await p.confirm({
    message: `Use the default Hindsight Cloud URL (${HINDSIGHT_CLOUD_URL})?`,
    initialValue: true,
  });
  assertNotCancelled(useDefaultUrl);

  let apiUrl: string | undefined;
  if (!useDefaultUrl) {
    const custom = await p.text({
      message: "Hindsight Cloud URL",
      placeholder: HINDSIGHT_CLOUD_URL,
      validate: validateRequired("URL is required"),
    });
    assertNotCancelled(custom);
    apiUrl = custom;
  }

  const token = await p.password({
    message: "Hindsight Cloud API token (paste the value, it will be masked)",
    validate: validateRequired("Token is required"),
  });
  assertNotCancelled(token);

  const input = { apiUrl, token };
  applyCloudMode(pluginConfig, input);
  return summarizeCloud(input);
}

async function promptApi(pluginConfig: Record<string, unknown>): Promise<string> {
  const apiUrl = await p.text({
    message: "Hindsight API URL",
    placeholder: "https://mcp.hindsight.example.com",
    validate: validateRequired("URL is required"),
  });
  assertNotCancelled(apiUrl);

  const needsToken = await p.confirm({
    message: "Does this API require an auth token?",
    initialValue: false,
  });
  assertNotCancelled(needsToken);

  let token: string | undefined;
  if (needsToken) {
    const value = await p.password({
      message: "API token (paste the value, it will be masked)",
      validate: validateRequired("Token is required"),
    });
    assertNotCancelled(value);
    token = value;
  }

  const input = { apiUrl, token };
  applyApiMode(pluginConfig, input);
  return summarizeApi(input);
}

async function promptEmbedded(pluginConfig: Record<string, unknown>): Promise<string> {
  const provider = await p.select({
    message: "LLM provider used by the Hindsight memory daemon",
    options: [
      { value: "openai", label: "OpenAI", hint: "API key required" },
      { value: "anthropic", label: "Anthropic", hint: "API key required" },
      { value: "gemini", label: "Gemini", hint: "API key required" },
      { value: "groq", label: "Groq", hint: "API key required" },
      {
        value: "claude-code",
        label: "Claude Code",
        hint: "no API key needed (uses Claude Code CLI auth)",
      },
      {
        value: "openai-codex",
        label: "OpenAI Codex",
        hint: "no API key needed (uses codex auth login)",
      },
      { value: "ollama", label: "Ollama", hint: "no API key needed (local models)" },
    ],
  });
  assertNotCancelled(provider);
  const llmProvider = provider as string;

  let apiKey: string | undefined;
  if (!NO_KEY_PROVIDERS.has(llmProvider)) {
    const value = await p.password({
      message: `${llmProvider} API key (paste the value, it will be masked)`,
      validate: validateRequired("API key is required"),
    });
    assertNotCancelled(value);
    apiKey = value;
  }

  const overrideModel = await p.confirm({
    message: "Override the default model?",
    initialValue: false,
  });
  assertNotCancelled(overrideModel);

  let llmModel: string | undefined;
  if (overrideModel) {
    const value = await p.text({
      message: "Model id",
      placeholder: "gpt-4o-mini",
      validate: validateRequired("Model id is required"),
    });
    assertNotCancelled(value);
    llmModel = value;
  }

  const input = { llmProvider, apiKey, llmModel };
  applyEmbeddedMode(pluginConfig, input);
  return summarizeEmbedded(input);
}

async function runInteractive(
  configPath: string
): Promise<{ summary: string; configPath: string }> {
  p.intro("🦞 Hindsight Memory setup for OpenClaw");
  p.log.info(`Config file: ${configPath}`);

  const cfg = await loadConfig(configPath);
  const pluginConfig = ensurePluginConfig(cfg);

  const mode = await p.select({
    message: "How do you want to run Hindsight?",
    options: [
      { value: "cloud", label: "Cloud", hint: "managed Hindsight, no local setup" },
      { value: "api", label: "External API", hint: "your own running Hindsight deployment" },
      {
        value: "embedded",
        label: "Embedded daemon",
        hint: "spawn a local hindsight daemon on this machine",
      },
    ],
  });
  assertNotCancelled(mode);

  let summary: string;
  if ((mode as SetupMode) === "cloud") {
    summary = await promptCloud(pluginConfig);
  } else if ((mode as SetupMode) === "api") {
    summary = await promptApi(pluginConfig);
  } else {
    summary = await promptEmbedded(pluginConfig);
  }

  const spin = p.spinner();
  spin.start("Writing configuration");
  await saveConfig(configPath, cfg);
  spin.stop(`Saved to ${configPath}`);

  p.note(
    [
      summary,
      "",
      "Next steps:",
      "  1. Restart the gateway:  openclaw gateway restart",
      "  2. Verify config:        openclaw config validate",
      "",
      "Secrets were stored inline in openclaw.json. To reference an env var",
      "instead (recommended for CI/production), use:",
      "  openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiToken \\",
      "      --ref-source env --ref-id HINDSIGHT_CLOUD_TOKEN",
    ].join("\n"),
    "Hindsight Memory configured"
  );
  p.outro("Done.");
  return { summary, configPath };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let args: ParsedCliArgs;
  try {
    args = parseCliArgs(process.argv.slice(2));
  } catch (err) {
    console.error(`hindsight-openclaw-setup: ${err instanceof Error ? err.message : err}`);
    console.error();
    console.error(usage());
    process.exit(2);
  }

  if (args.help) {
    console.log(usage());
    return;
  }

  const configPath = args.configPath ?? args.positional ?? DEFAULT_OPENCLAW_CONFIG_PATH;

  if (args.mode) {
    // Non-interactive path for scripts and CI.
    try {
      const { summary } = await runNonInteractive(args, configPath);
      console.log(`Hindsight Memory configured: ${summary}`);
      console.log(`Saved to ${configPath}`);
    } catch (err) {
      console.error(`hindsight-openclaw-setup: ${err instanceof Error ? err.message : err}`);
      process.exit(1);
    }
    return;
  }

  // Interactive path (default).
  await runInteractive(configPath);
}

/**
 * Only run `main()` when this file is the Node entry point. Importing it from
 * a test (or any other module) should not trigger the interactive wizard.
 *
 * When invoked through `node_modules/.bin/hindsight-openclaw-setup` (npm-created
 * symlink), `process.argv[1]` points at the symlink while `import.meta.url`
 * resolves to the real file. Canonicalize both via `realpath` so the check
 * still matches — otherwise `main()` never runs on bin invocations and the
 * command silently exits with no output.
 */
function canonicalize(path: string): string {
  const resolved = resolve(path);
  try {
    return realpathSync(resolved);
  } catch {
    return resolved;
  }
}

function isDirectRun(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return canonicalize(entry) === canonicalize(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isDirectRun()) {
  main().catch((err: unknown) => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`hindsight-openclaw-setup failed: ${msg}`);
    process.exit(1);
  });
}
