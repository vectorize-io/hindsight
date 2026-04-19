/**
 * Integration tests for the Hindsight OpenClaw integration.
 *
 * Exercises both HTTP mode (direct API calls) and Embed mode (local daemon
 * spawned via HindsightServer), talking to Hindsight through
 * `@vectorize-io/hindsight-client`.
 *
 * Requirements:
 *   HTTP mode:  Running Hindsight API at HINDSIGHT_API_URL (default: http://localhost:8888)
 *   Embed mode: hindsight-embed package at HINDSIGHT_EMBED_PACKAGE_PATH
 *               + LLM credentials (HINDSIGHT_API_LLM_PROVIDER / HINDSIGHT_API_LLM_API_KEY)
 *
 * Run:
 *   npm run test:integration
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { HindsightServer } from "@vectorize-io/hindsight-all";
import { HindsightClient } from "@vectorize-io/hindsight-client";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ---------------------------------------------------------------------------
// Test configuration (driven by environment variables)
// ---------------------------------------------------------------------------

const HINDSIGHT_API_URL = process.env.HINDSIGHT_API_URL || "http://localhost:8888";
const LLM_PROVIDER = process.env.HINDSIGHT_API_LLM_PROVIDER || "";
const LLM_API_KEY = process.env.HINDSIGHT_API_LLM_API_KEY || "";
const LLM_MODEL = process.env.HINDSIGHT_API_LLM_MODEL || "";

// Embed package path – defaults to the sibling hindsight-embed directory in the repo
const EMBED_PACKAGE_PATH =
  process.env.HINDSIGHT_EMBED_PACKAGE_PATH || join(__dirname, "..", "..", "..", "hindsight-embed");

// Port for the test embed daemon (different from production default 9077 to avoid conflicts)
const EMBED_TEST_PORT = 19077;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randomBankId(): string {
  return `openclaw_test_${Math.random().toString(36).slice(2, 14)}`;
}

async function waitForApi(url: string, maxMs = 5000): Promise<boolean> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1000) });
      if (res.ok) return true;
    } catch {
      // not ready yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

// ---------------------------------------------------------------------------
// HTTP Mode Tests
// ---------------------------------------------------------------------------

describe("openclaw integration — HTTP mode", () => {
  let client: HindsightClient;

  beforeAll(async () => {
    const reachable = await waitForApi(HINDSIGHT_API_URL);
    if (!reachable) {
      throw new Error(
        `Hindsight API not reachable at ${HINDSIGHT_API_URL}. ` +
          "Start the server before running integration tests."
      );
    }

    client = new HindsightClient({ baseUrl: HINDSIGHT_API_URL });
  });

  it("should retain a conversation", async () => {
    const bankId = randomBankId();

    const response = await client.retain(
      bankId,
      "[role: user]\nMy name is Alice and I love hiking.\n[user:end]\n\n" +
        "[role: assistant]\nNice to meet you, Alice!\n[assistant:end]",
      {
        documentId: "http-retain-test-1",
        metadata: { channel_type: "slack", sender_id: "U001" },
        async: true,
      }
    );

    expect(response).toBeDefined();
  });

  it("should retain with auto-generated document id", async () => {
    const bankId = randomBankId();

    const response = await client.retain(
      bankId,
      "[role: user]\nI work at TechCorp as a software engineer.\n[user:end]",
      { async: true }
    );

    expect(response).toBeDefined();
  });

  it("should recall from an empty bank without error", async () => {
    const bankId = randomBankId();
    const response = await client.recall(bankId, "What do I like?", { maxTokens: 512 });
    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it("should set bank mission via createBank after retain creates the bank", async () => {
    const bankId = randomBankId();
    await client.retain(bankId, "[role: user]\nHello\n[user:end]", { async: true });
    await expect(
      client.createBank(bankId, { reflectMission: "You are a helpful AI assistant." })
    ).resolves.toBeDefined();
  });

  it("should retain and then recall relevant memories", async () => {
    const bankId = randomBankId();

    await client.retain(
      bankId,
      "[role: user]\nMy favorite programming language is Python.\n[user:end]\n\n" +
        "[role: assistant]\nPython is a great choice!\n[assistant:end]",
      { documentId: `session-${Date.now()}`, async: true }
    );

    const response = await client.recall(bankId, "What programming language do I like?", {
      maxTokens: 1024,
    });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it("should use custom maxTokens in recall request", async () => {
    const bankId = randomBankId();
    const response = await client.recall(bankId, "anything", { maxTokens: 256 });
    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it("should map recall results to the RecallResult shape", async () => {
    const bankId = randomBankId();

    await client.retain(
      bankId,
      "[role: user]\nI enjoy reading science fiction books.\n[user:end]\n\n" +
        "[role: assistant]\nSounds like a great hobby!\n[assistant:end]",
      { documentId: "mapping-test", async: true }
    );

    const response = await client.recall(bankId, "What are my hobbies?", { maxTokens: 1024 });

    for (const result of response.results) {
      expect(typeof result.id).toBe("string");
      expect(typeof result.text).toBe("string");
    }
  });
});

// ---------------------------------------------------------------------------
// Embed Mode Tests (local daemon spawned by HindsightServer)
// ---------------------------------------------------------------------------

describe("openclaw integration — embed mode", () => {
  let client: HindsightClient;
  let server: HindsightServer;

  const hasEmbedCredentials = Boolean(LLM_PROVIDER && LLM_API_KEY);

  beforeAll(async () => {
    if (!hasEmbedCredentials) {
      console.warn(
        "[Integration] Skipping embed mode tests: " +
          "HINDSIGHT_API_LLM_PROVIDER and HINDSIGHT_API_LLM_API_KEY must both be set."
      );
      return;
    }

    server = new HindsightServer({
      profile: "openclaw-test",
      port: EMBED_TEST_PORT,
      embedVersion: "latest",
      embedPackagePath: EMBED_PACKAGE_PATH,
      env: {
        HINDSIGHT_API_LLM_PROVIDER: LLM_PROVIDER,
        HINDSIGHT_API_LLM_API_KEY: LLM_API_KEY,
        HINDSIGHT_API_LLM_MODEL: LLM_MODEL || undefined,
        HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: "0",
      },
    });

    await server.start();

    client = new HindsightClient({ baseUrl: server.getBaseUrl() });
  }, 120_000); // daemon startup can take up to 2 minutes

  afterAll(async () => {
    if (server) {
      await server.stop();
    }
  }, 30_000);

  it("should retain a conversation against the local daemon", async () => {
    if (!hasEmbedCredentials) return;
    const bankId = randomBankId();
    const response = await client.retain(
      bankId,
      "[role: user]\nI love hiking in the mountains.\n[user:end]\n\n" +
        "[role: assistant]\nSounds adventurous!\n[assistant:end]",
      { documentId: "embed-retain-test-1", async: true }
    );
    expect(response).toBeDefined();
  }, 60_000);

  it("should recall from an empty bank against the local daemon", async () => {
    if (!hasEmbedCredentials) return;
    const bankId = randomBankId();
    const response = await client.recall(bankId, "What do I like?", { maxTokens: 512 });
    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  }, 60_000);

  it("should set bank mission against the local daemon", async () => {
    if (!hasEmbedCredentials) return;
    const bankId = randomBankId();
    // Create bank by retaining first, then set mission
    await client.retain(bankId, "[role: user]\nHello\n[user:end]", { async: true });
    await expect(
      client.createBank(bankId, { reflectMission: "Test mission for embed integration tests." })
    ).resolves.toBeDefined();
  }, 60_000);

  it("should retain and recall against the local daemon", async () => {
    if (!hasEmbedCredentials) return;
    const bankId = randomBankId();
    await client.retain(
      bankId,
      "[role: user]\nMy cat is named Whiskers and she is 3 years old.\n[user:end]\n\n" +
        "[role: assistant]\nWhat a lovely name!\n[assistant:end]",
      { documentId: `embed-e2e-${Date.now()}`, async: true }
    );
    const response = await client.recall(bankId, "What is my cat's name?", { maxTokens: 1024 });
    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  }, 60_000);
});
