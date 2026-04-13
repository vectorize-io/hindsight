/**
 * Pure helpers behind the Hindsight OpenClaw setup wizard. Kept separate from
 * setup.ts (the @clack/prompts entry point) so the mechanical bits are easy to
 * unit test without simulating an interactive terminal.
 *
 * Scanner-safe: imports no subprocess APIs and does not read any environment
 * variable. All config writing is an atomic rename over the OpenClaw config JSON.
 */

import { readFile, writeFile, mkdir, rename } from 'fs/promises';
import { homedir } from 'os';
import { join, dirname } from 'path';

export const PLUGIN_ID = 'hindsight-openclaw';

/**
 * Default Hindsight Cloud endpoint. Update this when the hosted service URL is
 * finalized, or users can override it at the prompt.
 */
export const HINDSIGHT_CLOUD_URL = 'https://api.hindsight.vectorize.io';

export const DEFAULT_OPENCLAW_CONFIG_PATH = join(homedir(), '.openclaw', 'openclaw.json');

export interface SecretRef {
  source: 'env' | 'file' | 'exec';
  provider: string;
  id: string;
}

export interface PluginEntry {
  enabled?: boolean;
  config?: Record<string, unknown>;
}

export interface OpenClawConfigShape {
  plugins?: {
    entries?: Record<string, PluginEntry>;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export type SetupMode = 'cloud' | 'api' | 'embedded';

export const NO_KEY_PROVIDERS: ReadonlySet<string> = new Set([
  'claude-code',
  'openai-codex',
  'ollama',
]);

export async function loadConfig(path: string): Promise<OpenClawConfigShape> {
  try {
    const raw = await readFile(path, 'utf8');
    return JSON.parse(raw) as OpenClawConfigShape;
  } catch (err: unknown) {
    if ((err as NodeJS.ErrnoException)?.code === 'ENOENT') return {};
    throw new Error(
      `Failed to read ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

export async function saveConfig(path: string, cfg: OpenClawConfigShape): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const serialized = `${JSON.stringify(cfg, null, 2)}\n`;
  const tmpPath = `${path}.tmp-${Date.now()}`;
  await writeFile(tmpPath, serialized, 'utf8');
  await rename(tmpPath, path);
}

/**
 * Ensure a `plugins.entries["hindsight-openclaw"].config` object exists, set
 * `enabled: true`, and return the mutable config record. Idempotent — safe to
 * call against a fresh or already-configured OpenClaw config.
 */
export function ensurePluginConfig(cfg: OpenClawConfigShape): Record<string, unknown> {
  const plugins = (cfg.plugins ??= {});
  const entries = (plugins.entries ??= {});
  const entry = (entries[PLUGIN_ID] ??= { enabled: true });
  entry.enabled = true;
  return (entry.config ??= {});
}

export function envSecretRef(id: string): SecretRef {
  return { source: 'env', provider: 'default', id };
}

export function clearCloudFields(pluginConfig: Record<string, unknown>): void {
  delete pluginConfig.hindsightApiUrl;
  delete pluginConfig.hindsightApiToken;
}

export function clearLocalLlmFields(pluginConfig: Record<string, unknown>): void {
  delete pluginConfig.llmProvider;
  delete pluginConfig.llmModel;
  delete pluginConfig.llmApiKey;
  delete pluginConfig.llmBaseUrl;
}

const ENV_VAR_RE = /^[A-Z][A-Z0-9_]*$/;

export function isValidEnvVarName(value: string | undefined): boolean {
  return !!value && ENV_VAR_RE.test(value.trim());
}

export function defaultApiKeyEnvVar(provider: string): string {
  return `${provider.toUpperCase().replace(/-/g, '_')}_API_KEY`;
}

/**
 * Cloud mode credential: either a direct token value (stored inline as a
 * plaintext string in openclaw.json) or an env var name (stored as a
 * SecretRef that OpenClaw resolves from `process.env` at startup).
 *
 * Interactive wizard defaults to the direct-value form — simpler UX for
 * users pasting a freshly-issued cloud token. CI / production flows should
 * prefer `tokenEnvVar` via `openclaw config set ... --ref-source env` or
 * `--token-env` to keep secrets off disk.
 */
export interface CloudSetupInput {
  apiUrl?: string;
  token?: string;
  tokenEnvVar?: string;
}

export interface ApiSetupInput {
  apiUrl: string;
  token?: string;
  tokenEnvVar?: string;
}

export interface EmbeddedSetupInput {
  llmProvider: string;
  apiKey?: string;
  apiKeyEnvVar?: string;
  llmModel?: string;
}

function pickCredential(
  token: string | undefined,
  tokenEnvVar: string | undefined,
): string | SecretRef | undefined {
  const hasToken = token && token.trim().length > 0;
  const hasEnvVar = tokenEnvVar && tokenEnvVar.trim().length > 0;
  if (hasToken && hasEnvVar) {
    throw new Error('provide either a direct value or an env var name — not both');
  }
  if (hasToken) return token!.trim();
  if (hasEnvVar) return envSecretRef(tokenEnvVar!.trim());
  return undefined;
}

/**
 * Apply the Cloud mode to a plugin config in place: sets `hindsightApiUrl` and
 * `hindsightApiToken` (either as a plaintext string or as a SecretRef —
 * whichever the caller provided), strips any leftover local-LLM fields so
 * mode switches don't carry stale state.
 */
export function applyCloudMode(
  pluginConfig: Record<string, unknown>,
  input: CloudSetupInput,
): void {
  const token = pickCredential(input.token, input.tokenEnvVar);
  if (token === undefined) {
    throw new Error('Cloud mode requires either `token` or `tokenEnvVar`');
  }
  clearLocalLlmFields(pluginConfig);
  pluginConfig.hindsightApiUrl = (input.apiUrl ?? HINDSIGHT_CLOUD_URL).trim();
  pluginConfig.hindsightApiToken = token;
}

/**
 * Apply the external-API mode to a plugin config in place: sets a required
 * `hindsightApiUrl`, optional `hindsightApiToken` (plaintext or SecretRef),
 * and strips any leftover local-LLM fields so mode switches don't carry
 * stale state.
 */
export function applyApiMode(
  pluginConfig: Record<string, unknown>,
  input: ApiSetupInput,
): void {
  const token = pickCredential(input.token, input.tokenEnvVar);
  clearLocalLlmFields(pluginConfig);
  pluginConfig.hindsightApiUrl = input.apiUrl.trim();
  if (token !== undefined) {
    pluginConfig.hindsightApiToken = token;
  } else {
    delete pluginConfig.hindsightApiToken;
  }
}

/**
 * Apply the embedded-daemon mode to a plugin config in place: sets
 * `llmProvider`, optional `llmApiKey` (plaintext or SecretRef), optional
 * `llmModel`, and strips any external-API settings so mode switches don't
 * carry stale state.
 */
export function applyEmbeddedMode(
  pluginConfig: Record<string, unknown>,
  input: EmbeddedSetupInput,
): void {
  const key = pickCredential(input.apiKey, input.apiKeyEnvVar);
  clearCloudFields(pluginConfig);
  pluginConfig.llmProvider = input.llmProvider;
  if (NO_KEY_PROVIDERS.has(input.llmProvider)) {
    delete pluginConfig.llmApiKey;
  } else {
    if (key === undefined) {
      throw new Error(`llmProvider "${input.llmProvider}" requires either \`apiKey\` or \`apiKeyEnvVar\``);
    }
    pluginConfig.llmApiKey = key;
  }
  if (input.llmModel && input.llmModel.trim().length > 0) {
    pluginConfig.llmModel = input.llmModel.trim();
  } else {
    delete pluginConfig.llmModel;
  }
}

function credentialSuffix(token: string | undefined, tokenEnvVar: string | undefined): string {
  if (tokenEnvVar && tokenEnvVar.trim().length > 0) {
    return ` (token from \${${tokenEnvVar.trim()}})`;
  }
  if (token && token.trim().length > 0) {
    return ' (token stored inline)';
  }
  return ' (no auth)';
}

export function summarizeCloud(input: CloudSetupInput): string {
  const url = (input.apiUrl ?? HINDSIGHT_CLOUD_URL).trim();
  return `Cloud → ${url}${credentialSuffix(input.token, input.tokenEnvVar)}`;
}

export function summarizeApi(input: ApiSetupInput): string {
  return `External API → ${input.apiUrl.trim()}${credentialSuffix(input.token, input.tokenEnvVar)}`;
}

export function summarizeEmbedded(input: EmbeddedSetupInput): string {
  if (NO_KEY_PROVIDERS.has(input.llmProvider)) {
    return `Embedded daemon → ${input.llmProvider}`;
  }
  const keyHint = input.apiKeyEnvVar ? ` (key from \${${input.apiKeyEnvVar.trim()}})` : ' (key stored inline)';
  return `Embedded daemon → ${input.llmProvider}${keyHint}`;
}
