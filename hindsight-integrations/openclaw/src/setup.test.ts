import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, readFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { parseCliArgs, runNonInteractive } from "./setup.js";
import { PLUGIN_ID, type OpenClawConfigShape } from "./setup-lib.js";

describe("parseCliArgs", () => {
  it("returns defaults for no args", () => {
    const args = parseCliArgs([]);
    expect(args).toEqual({ help: false, noToken: false });
  });

  it("parses --help", () => {
    expect(parseCliArgs(["--help"]).help).toBe(true);
    expect(parseCliArgs(["-h"]).help).toBe(true);
  });

  it("parses --config-path and positional config path", () => {
    expect(parseCliArgs(["--config-path", "/tmp/a.json"]).configPath).toBe("/tmp/a.json");
    expect(parseCliArgs(["/tmp/b.json"]).positional).toBe("/tmp/b.json");
  });

  it("parses cloud-mode flags (direct token value)", () => {
    const args = parseCliArgs(["--mode", "cloud", "--token", "hsk_literal"]);
    expect(args).toMatchObject({ mode: "cloud", token: "hsk_literal" });
  });

  it("parses cloud-mode flags (token env var)", () => {
    const args = parseCliArgs([
      "--mode",
      "cloud",
      "--api-url",
      "https://cloud.example.com",
      "--token-env",
      "HINDSIGHT_CLOUD_TOKEN",
    ]);
    expect(args).toMatchObject({
      mode: "cloud",
      apiUrl: "https://cloud.example.com",
      tokenEnv: "HINDSIGHT_CLOUD_TOKEN",
    });
  });

  it("parses embedded-mode direct apiKey flag", () => {
    const args = parseCliArgs([
      "--mode",
      "embedded",
      "--provider",
      "openai",
      "--api-key",
      "sk-literal",
    ]);
    expect(args).toMatchObject({
      mode: "embedded",
      provider: "openai",
      apiKey: "sk-literal",
    });
  });

  it("parses api-mode flags with --no-token", () => {
    const args = parseCliArgs([
      "--mode",
      "api",
      "--api-url",
      "https://mcp.example.com",
      "--no-token",
    ]);
    expect(args).toMatchObject({
      mode: "api",
      apiUrl: "https://mcp.example.com",
      noToken: true,
    });
  });

  it("parses embedded-mode flags", () => {
    const args = parseCliArgs([
      "--mode",
      "embedded",
      "--provider",
      "openai",
      "--api-key-env",
      "OPENAI_API_KEY",
      "--model",
      "gpt-4o-mini",
    ]);
    expect(args).toMatchObject({
      mode: "embedded",
      provider: "openai",
      apiKeyEnv: "OPENAI_API_KEY",
      model: "gpt-4o-mini",
    });
  });

  it("rejects invalid --mode", () => {
    expect(() => parseCliArgs(["--mode", "bogus"])).toThrow(/invalid --mode/);
  });

  it("rejects unknown flags", () => {
    expect(() => parseCliArgs(["--what-is-this"])).toThrow(/unknown argument/);
  });

  it("rejects flags missing a value", () => {
    expect(() => parseCliArgs(["--mode"])).toThrow(/missing value for --mode/);
    expect(() => parseCliArgs(["--api-url"])).toThrow(/missing value for --api-url/);
  });

  it("rejects extra positional args", () => {
    expect(() => parseCliArgs(["/tmp/a.json", "/tmp/b.json"])).toThrow(/extra positional/);
  });
});

describe("runNonInteractive", () => {
  let tmpDir: string;
  let configPath: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "hindsight-openclaw-setup-cli-"));
    configPath = join(tmpDir, "openclaw.json");
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  async function readBack(): Promise<OpenClawConfigShape> {
    return JSON.parse(await readFile(configPath, "utf8")) as OpenClawConfigShape;
  }

  it("writes a cloud-mode config with the default URL", async () => {
    const args = parseCliArgs(["--mode", "cloud", "--token-env", "HINDSIGHT_CLOUD_TOKEN"]);
    const result = await runNonInteractive(args, configPath);
    expect(result.summary).toContain("Cloud");
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.hindsightApiUrl).toBe("https://api.hindsight.vectorize.io");
    expect(pc.hindsightApiToken).toEqual({
      source: "env",
      provider: "default",
      id: "HINDSIGHT_CLOUD_TOKEN",
    });
    expect(cfg.plugins?.entries?.[PLUGIN_ID]?.enabled).toBe(true);
  });

  it("writes a cloud-mode config with a custom URL", async () => {
    const args = parseCliArgs([
      "--mode",
      "cloud",
      "--api-url",
      "https://hindsight.custom.example.com",
      "--token-env",
      "MY_TOKEN",
    ]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.hindsightApiUrl).toBe("https://hindsight.custom.example.com");
    expect((pc.hindsightApiToken as { id: string }).id).toBe("MY_TOKEN");
  });

  it("writes a cloud-mode config with an inline token (--token)", async () => {
    const args = parseCliArgs(["--mode", "cloud", "--token", "hsk_direct_value"]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.hindsightApiUrl).toBe("https://api.hindsight.vectorize.io");
    expect(pc.hindsightApiToken).toBe("hsk_direct_value");
  });

  it("rejects cloud mode without --token or --token-env", async () => {
    const args = parseCliArgs(["--mode", "cloud"]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/--token .*--token-env/);
  });

  it("rejects cloud mode with both --token and --token-env", async () => {
    const args = parseCliArgs([
      "--mode",
      "cloud",
      "--token",
      "hsk_x",
      "--token-env",
      "HINDSIGHT_CLOUD_TOKEN",
    ]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/mutually exclusive/);
  });

  it("rejects cloud mode with a bad token env var name", async () => {
    const args = parseCliArgs(["--mode", "cloud", "--token-env", "bad-name"]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/UPPER_SNAKE_CASE/);
  });

  it("writes an api-mode config without token", async () => {
    const args = parseCliArgs([
      "--mode",
      "api",
      "--api-url",
      "https://mcp.example.com",
      "--no-token",
    ]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.hindsightApiUrl).toBe("https://mcp.example.com");
    expect(pc.hindsightApiToken).toBeUndefined();
  });

  it("writes an api-mode config with token", async () => {
    const args = parseCliArgs([
      "--mode",
      "api",
      "--api-url",
      "https://mcp.example.com",
      "--token-env",
      "HINDSIGHT_API_TOKEN",
    ]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect((pc.hindsightApiToken as { id: string }).id).toBe("HINDSIGHT_API_TOKEN");
  });

  it("rejects api mode without --api-url", async () => {
    const args = parseCliArgs(["--mode", "api"]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/--api-url/);
  });

  it("rejects api mode with conflicting --token-env and --no-token", async () => {
    const args = parseCliArgs([
      "--mode",
      "api",
      "--api-url",
      "https://mcp.example.com",
      "--token-env",
      "FOO",
      "--no-token",
    ]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/mutually exclusive/);
  });

  it("writes an embedded-mode config for openai", async () => {
    const args = parseCliArgs([
      "--mode",
      "embedded",
      "--provider",
      "openai",
      "--api-key-env",
      "OPENAI_API_KEY",
      "--model",
      "gpt-4o-mini",
    ]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.llmProvider).toBe("openai");
    expect((pc.llmApiKey as { id: string }).id).toBe("OPENAI_API_KEY");
    expect(pc.llmModel).toBe("gpt-4o-mini");
    expect(pc.hindsightApiUrl).toBeUndefined();
  });

  it("writes an embedded-mode config for a no-key provider", async () => {
    const args = parseCliArgs(["--mode", "embedded", "--provider", "claude-code"]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.llmProvider).toBe("claude-code");
    expect(pc.llmApiKey).toBeUndefined();
  });

  it("rejects embedded mode without --provider", async () => {
    const args = parseCliArgs(["--mode", "embedded"]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/--provider/);
  });

  it("rejects embedded mode with a key-requiring provider but no --api-key or --api-key-env", async () => {
    const args = parseCliArgs(["--mode", "embedded", "--provider", "openai"]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/--api-key .*--api-key-env/);
  });

  it("rejects embedded mode with both --api-key and --api-key-env", async () => {
    const args = parseCliArgs([
      "--mode",
      "embedded",
      "--provider",
      "openai",
      "--api-key",
      "sk-x",
      "--api-key-env",
      "OPENAI_API_KEY",
    ]);
    await expect(runNonInteractive(args, configPath)).rejects.toThrow(/mutually exclusive/);
  });

  it("writes an embedded-mode config with an inline API key (--api-key)", async () => {
    const args = parseCliArgs([
      "--mode",
      "embedded",
      "--provider",
      "openai",
      "--api-key",
      "sk-inline",
    ]);
    await runNonInteractive(args, configPath);
    const cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.llmProvider).toBe("openai");
    expect(pc.llmApiKey).toBe("sk-inline");
    expect(typeof pc.llmApiKey).toBe("string");
  });

  it("clears stale fields when switching between modes", async () => {
    // First write an embedded-mode config
    await runNonInteractive(
      parseCliArgs([
        "--mode",
        "embedded",
        "--provider",
        "openai",
        "--api-key-env",
        "OPENAI_API_KEY",
      ]),
      configPath
    );
    let cfg = await readBack();
    expect(cfg.plugins?.entries?.[PLUGIN_ID]?.config?.llmProvider).toBe("openai");

    // Now switch to cloud mode — local LLM fields should be gone
    await runNonInteractive(
      parseCliArgs(["--mode", "cloud", "--token-env", "HINDSIGHT_CLOUD_TOKEN"]),
      configPath
    );
    cfg = await readBack();
    const pc = cfg.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
    expect(pc.llmProvider).toBeUndefined();
    expect(pc.llmApiKey).toBeUndefined();
    expect(pc.hindsightApiUrl).toBe("https://api.hindsight.vectorize.io");
  });
});
