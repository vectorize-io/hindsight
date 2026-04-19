import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, readFile, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import {
  HINDSIGHT_CLOUD_URL,
  PLUGIN_ID,
  applyApiMode,
  applyCloudMode,
  applyEmbeddedMode,
  defaultApiKeyEnvVar,
  ensurePluginConfig,
  envSecretRef,
  isValidEnvVarName,
  loadConfig,
  saveConfig,
  summarizeApi,
  summarizeCloud,
  summarizeEmbedded,
  type OpenClawConfigShape,
} from "./setup-lib.js";

describe("isValidEnvVarName", () => {
  it("accepts UPPER_SNAKE_CASE", () => {
    expect(isValidEnvVarName("OPENAI_API_KEY")).toBe(true);
    expect(isValidEnvVarName("HINDSIGHT_CLOUD_TOKEN")).toBe(true);
    expect(isValidEnvVarName("A")).toBe(true);
  });
  it("rejects lowercase, leading digits, empty, and undefined", () => {
    expect(isValidEnvVarName("lowercase")).toBe(false);
    expect(isValidEnvVarName("1LEADING_DIGIT")).toBe(false);
    expect(isValidEnvVarName("")).toBe(false);
    expect(isValidEnvVarName(undefined)).toBe(false);
    expect(isValidEnvVarName("has-dash")).toBe(false);
  });
});

describe("defaultApiKeyEnvVar", () => {
  it("UPPERs and snake_cases the provider id", () => {
    expect(defaultApiKeyEnvVar("openai")).toBe("OPENAI_API_KEY");
    expect(defaultApiKeyEnvVar("claude-code")).toBe("CLAUDE_CODE_API_KEY");
  });
});

describe("envSecretRef", () => {
  it("builds a default-provider env SecretRef", () => {
    expect(envSecretRef("OPENAI_API_KEY")).toEqual({
      source: "env",
      provider: "default",
      id: "OPENAI_API_KEY",
    });
  });
});

describe("ensurePluginConfig", () => {
  it("initializes the hindsight-openclaw entry on an empty config", () => {
    const cfg: OpenClawConfigShape = {};
    const pc = ensurePluginConfig(cfg);
    expect(cfg.plugins?.entries?.[PLUGIN_ID]).toEqual({ enabled: true, config: {} });
    expect(pc).toBe(cfg.plugins?.entries?.[PLUGIN_ID]?.config);
  });

  it("preserves existing config values and forces enabled=true", () => {
    const cfg: OpenClawConfigShape = {
      plugins: {
        entries: {
          [PLUGIN_ID]: {
            enabled: false,
            config: { llmProvider: "openai" },
          },
        },
      },
    };
    const pc = ensurePluginConfig(cfg);
    expect(cfg.plugins?.entries?.[PLUGIN_ID]?.enabled).toBe(true);
    expect(pc.llmProvider).toBe("openai");
  });
});

describe("applyCloudMode — direct token value", () => {
  it("stores the token inline when a literal value is provided", () => {
    const pc: Record<string, unknown> = {
      llmProvider: "openai",
      llmApiKey: { source: "env", provider: "default", id: "OPENAI_API_KEY" },
    };
    applyCloudMode(pc, { token: "hsk_literal_value" });
    expect(pc.hindsightApiUrl).toBe(HINDSIGHT_CLOUD_URL);
    expect(pc.hindsightApiToken).toBe("hsk_literal_value");
    expect(pc.llmProvider).toBeUndefined();
    expect(pc.llmApiKey).toBeUndefined();
  });

  it("trims whitespace around an inline token", () => {
    const pc: Record<string, unknown> = {};
    applyCloudMode(pc, { token: "  hsk_padded  " });
    expect(pc.hindsightApiToken).toBe("hsk_padded");
  });

  it("throws when neither token nor tokenEnvVar is provided", () => {
    const pc: Record<string, unknown> = {};
    expect(() => applyCloudMode(pc, {})).toThrow(/requires either/);
  });

  it("throws when both token and tokenEnvVar are provided", () => {
    const pc: Record<string, unknown> = {};
    expect(() => applyCloudMode(pc, { token: "x", tokenEnvVar: "Y" })).toThrow(
      /either a direct value or an env var name/
    );
  });
});

describe("applyCloudMode", () => {
  it("writes the default URL and a SecretRef, stripping local LLM state", () => {
    const pc: Record<string, unknown> = {
      llmProvider: "openai",
      llmApiKey: { source: "env", provider: "default", id: "OPENAI_API_KEY" },
      llmModel: "gpt-4o-mini",
      llmBaseUrl: "https://openrouter.ai/api/v1",
    };
    applyCloudMode(pc, { tokenEnvVar: "HINDSIGHT_CLOUD_TOKEN" });
    expect(pc.hindsightApiUrl).toBe(HINDSIGHT_CLOUD_URL);
    expect(pc.hindsightApiToken).toEqual({
      source: "env",
      provider: "default",
      id: "HINDSIGHT_CLOUD_TOKEN",
    });
    expect(pc.llmProvider).toBeUndefined();
    expect(pc.llmApiKey).toBeUndefined();
    expect(pc.llmModel).toBeUndefined();
    expect(pc.llmBaseUrl).toBeUndefined();
  });

  it("honours an overridden apiUrl", () => {
    const pc: Record<string, unknown> = {};
    applyCloudMode(pc, {
      apiUrl: "https://cloud.example.com",
      tokenEnvVar: "CLOUD_TOKEN",
    });
    expect(pc.hindsightApiUrl).toBe("https://cloud.example.com");
    expect((pc.hindsightApiToken as { id: string }).id).toBe("CLOUD_TOKEN");
  });
});

describe("applyApiMode — direct token value", () => {
  it("stores the token inline when a literal value is provided", () => {
    const pc: Record<string, unknown> = {};
    applyApiMode(pc, { apiUrl: "https://mcp.example.com", token: "api_literal" });
    expect(pc.hindsightApiUrl).toBe("https://mcp.example.com");
    expect(pc.hindsightApiToken).toBe("api_literal");
  });

  it("throws when both token and tokenEnvVar are provided", () => {
    const pc: Record<string, unknown> = {};
    expect(() =>
      applyApiMode(pc, { apiUrl: "https://mcp.example.com", token: "x", tokenEnvVar: "Y" })
    ).toThrow(/either a direct value or an env var name/);
  });
});

describe("applyApiMode", () => {
  it("writes the URL without a token when none is provided", () => {
    const pc: Record<string, unknown> = {
      llmProvider: "openai",
      hindsightApiToken: { source: "env", provider: "default", id: "STALE_TOKEN" },
    };
    applyApiMode(pc, { apiUrl: "https://mcp.example.com" });
    expect(pc.hindsightApiUrl).toBe("https://mcp.example.com");
    expect(pc.hindsightApiToken).toBeUndefined();
    expect(pc.llmProvider).toBeUndefined();
  });

  it("writes a SecretRef when a token env var is provided", () => {
    const pc: Record<string, unknown> = {};
    applyApiMode(pc, { apiUrl: "https://mcp.example.com", tokenEnvVar: "MY_TOKEN" });
    expect(pc.hindsightApiToken).toEqual({
      source: "env",
      provider: "default",
      id: "MY_TOKEN",
    });
  });

  it('treats an empty token env var as "no token"', () => {
    const pc: Record<string, unknown> = {};
    applyApiMode(pc, { apiUrl: "https://mcp.example.com", tokenEnvVar: "  " });
    expect(pc.hindsightApiToken).toBeUndefined();
  });
});

describe("applyEmbeddedMode — direct API key value", () => {
  it("stores the API key inline when a literal value is provided", () => {
    const pc: Record<string, unknown> = {};
    applyEmbeddedMode(pc, { llmProvider: "openai", apiKey: "sk-literal" });
    expect(pc.llmProvider).toBe("openai");
    expect(pc.llmApiKey).toBe("sk-literal");
  });

  it("throws when both apiKey and apiKeyEnvVar are provided", () => {
    const pc: Record<string, unknown> = {};
    expect(() =>
      applyEmbeddedMode(pc, {
        llmProvider: "openai",
        apiKey: "sk-x",
        apiKeyEnvVar: "OPENAI_API_KEY",
      })
    ).toThrow(/either a direct value or an env var name/);
  });
});

describe("applyEmbeddedMode", () => {
  it("writes llmProvider + SecretRef for providers that require a key", () => {
    const pc: Record<string, unknown> = {
      hindsightApiUrl: "https://stale.example.com",
      hindsightApiToken: { source: "env", provider: "default", id: "STALE" },
    };
    applyEmbeddedMode(pc, { llmProvider: "openai", apiKeyEnvVar: "OPENAI_API_KEY" });
    expect(pc.llmProvider).toBe("openai");
    expect(pc.llmApiKey).toEqual({
      source: "env",
      provider: "default",
      id: "OPENAI_API_KEY",
    });
    expect(pc.hindsightApiUrl).toBeUndefined();
    expect(pc.hindsightApiToken).toBeUndefined();
  });

  it("omits llmApiKey for no-key providers like claude-code", () => {
    const pc: Record<string, unknown> = {
      llmApiKey: { source: "env", provider: "default", id: "STALE" },
    };
    applyEmbeddedMode(pc, { llmProvider: "claude-code" });
    expect(pc.llmProvider).toBe("claude-code");
    expect(pc.llmApiKey).toBeUndefined();
  });

  it("throws when a key-requiring provider is given without a key", () => {
    const pc: Record<string, unknown> = {};
    expect(() => applyEmbeddedMode(pc, { llmProvider: "openai" })).toThrow(
      /requires either `apiKey` or `apiKeyEnvVar`/
    );
  });

  it("persists llmModel when provided and clears it when absent", () => {
    const pc: Record<string, unknown> = { llmModel: "legacy-model" };
    applyEmbeddedMode(pc, { llmProvider: "ollama", llmModel: "llama3" });
    expect(pc.llmModel).toBe("llama3");

    applyEmbeddedMode(pc, { llmProvider: "ollama" });
    expect(pc.llmModel).toBeUndefined();
  });
});

describe("summarize*", () => {
  it("produces human-readable mode summaries", () => {
    expect(summarizeCloud({ tokenEnvVar: "HINDSIGHT_CLOUD_TOKEN" })).toBe(
      "Cloud → https://api.hindsight.vectorize.io (token from ${HINDSIGHT_CLOUD_TOKEN})"
    );
    expect(summarizeApi({ apiUrl: "https://api.example.com", tokenEnvVar: "T" })).toBe(
      "External API → https://api.example.com (token from ${T})"
    );
    expect(summarizeApi({ apiUrl: "https://api.example.com", token: "literal" })).toBe(
      "External API → https://api.example.com (token stored inline)"
    );
    expect(summarizeApi({ apiUrl: "https://api.example.com" })).toBe(
      "External API → https://api.example.com (no auth)"
    );
    expect(summarizeEmbedded({ llmProvider: "openai", apiKeyEnvVar: "X" })).toBe(
      "Embedded daemon → openai (key from ${X})"
    );
    expect(summarizeEmbedded({ llmProvider: "openai", apiKey: "sk-test" })).toBe(
      "Embedded daemon → openai (key stored inline)"
    );
    expect(summarizeEmbedded({ llmProvider: "claude-code" })).toBe("Embedded daemon → claude-code");
  });
});

describe("loadConfig / saveConfig", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "hindsight-openclaw-setup-"));
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it("returns an empty object when the config file does not exist", async () => {
    const cfg = await loadConfig(join(tmpDir, "missing.json"));
    expect(cfg).toEqual({});
  });

  it("round-trips a config via atomic save and load", async () => {
    const path = join(tmpDir, "openclaw.json");
    const cfg: OpenClawConfigShape = {
      plugins: {
        entries: {
          [PLUGIN_ID]: {
            enabled: true,
            config: { llmProvider: "openai" },
          },
        },
      },
    };
    await saveConfig(path, cfg);
    const roundtrip = await loadConfig(path);
    expect(roundtrip).toEqual(cfg);
    // File should end in a newline (cosmetic — nice for diffs/editors).
    const raw = await readFile(path, "utf8");
    expect(raw.endsWith("\n")).toBe(true);
  });

  it("creates the parent directory if it does not exist", async () => {
    const path = join(tmpDir, "nested", "subdir", "openclaw.json");
    await saveConfig(path, { hello: "world" });
    const roundtrip = await loadConfig(path);
    expect(roundtrip).toEqual({ hello: "world" });
  });

  it("does not leave the .tmp file behind on success", async () => {
    const path = join(tmpDir, "openclaw.json");
    await saveConfig(path, {});
    const raw = await readFile(path, "utf8");
    expect(raw).toContain("{}");
    // Ensure the rename cleaned up the temp file.
    await expect(readFile(`${path}.tmp-1`, "utf8").catch(() => "missing")).resolves.toBe("missing");
  });

  it("throws a useful error when the config file is invalid JSON", async () => {
    const path = join(tmpDir, "bad.json");
    await writeFile(path, "{ not json", "utf8");
    await expect(loadConfig(path)).rejects.toThrow(/Failed to read/);
  });
});
