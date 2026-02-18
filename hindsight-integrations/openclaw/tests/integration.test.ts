/**
 * Integration tests for the Hindsight OpenClaw integration.
 *
 * Tests both HTTP mode (direct API calls) and Embed mode (subprocess/daemon).
 *
 * Requirements:
 *   HTTP mode:  Running Hindsight API at HINDSIGHT_API_URL (default: http://localhost:8888)
 *   Embed mode: hindsight-embed package at HINDSIGHT_EMBED_PACKAGE_PATH
 *               + LLM credentials (HINDSIGHT_API_LLM_PROVIDER / HINDSIGHT_API_LLM_API_KEY)
 *
 * Run:
 *   npm run test:integration
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { HindsightClient } from '../src/client.js';
import { HindsightEmbedManager } from '../src/embed-manager.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ---------------------------------------------------------------------------
// Test configuration (driven by environment variables)
// ---------------------------------------------------------------------------

const HINDSIGHT_API_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';
const LLM_PROVIDER = process.env.HINDSIGHT_API_LLM_PROVIDER || '';
const LLM_API_KEY = process.env.HINDSIGHT_API_LLM_API_KEY || '';
const LLM_MODEL = process.env.HINDSIGHT_API_LLM_MODEL || '';

// Embed package path – defaults to the sibling hindsight-embed directory in the repo
const EMBED_PACKAGE_PATH =
  process.env.HINDSIGHT_EMBED_PACKAGE_PATH ||
  join(__dirname, '..', '..', '..', 'hindsight-embed');

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

describe('HindsightClient – HTTP Mode', () => {
  let client: HindsightClient;

  beforeAll(async () => {
    const reachable = await waitForApi(HINDSIGHT_API_URL);
    if (!reachable) {
      throw new Error(
        `Hindsight API not reachable at ${HINDSIGHT_API_URL}. ` +
          'Start the server before running integration tests.',
      );
    }

    client = new HindsightClient({
      llmProvider: LLM_PROVIDER || 'openai',
      llmApiKey: LLM_API_KEY || 'test-key',
      llmModel: LLM_MODEL || undefined,
      apiUrl: HINDSIGHT_API_URL,
    });
  });

  it('should retain a conversation', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.retain({
      content:
        '[role: user]\nMy name is Alice and I love hiking.\n[user:end]\n\n' +
        '[role: assistant]\nNice to meet you, Alice!\n[assistant:end]',
      document_id: 'http-retain-test-1',
      metadata: { channel_type: 'slack', sender_id: 'U001' },
    });

    expect(response).toBeDefined();
    expect(response.message).toBeDefined();
    expect(response.document_id).toBe('http-retain-test-1');
  });

  it('should retain with auto-generated document id', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.retain({
      content: '[role: user]\nI work at TechCorp as a software engineer.\n[user:end]',
    });

    expect(response).toBeDefined();
    expect(response.document_id).toBe('conversation');
  });

  it('should recall from an empty bank without error', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.recall({ query: 'What do I like?', max_tokens: 512 });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it('should set bank mission without throwing', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    // setBankMission on a non-existent bank logs a warning but does not throw
    await expect(
      client.setBankMission('You are an assistant helping users via Slack.'),
    ).resolves.not.toThrow();
  });

  it('should set bank mission after retain creates the bank', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    // Create the bank by retaining something first
    await client.retain({ content: '[role: user]\nHello\n[user:end]' });

    // Now set the mission – bank exists so this should succeed
    await expect(
      client.setBankMission('You are a helpful AI assistant.'),
    ).resolves.not.toThrow();
  });

  it('should retain and then recall relevant memories', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    await client.retain({
      content:
        '[role: user]\nMy favorite programming language is Python.\n[user:end]\n\n' +
        '[role: assistant]\nPython is a great choice!\n[assistant:end]',
      document_id: `session-${Date.now()}`,
    });

    const response = await client.recall({
      query: 'What programming language do I like?',
      max_tokens: 1024,
    });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it('should silently truncate recall queries over 800 chars', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    const longQuery = 'Tell me about my interests. '.repeat(50); // > 800 chars
    const response = await client.recall({ query: longQuery, max_tokens: 512 });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it('should use custom max_tokens in recall request', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.recall({ query: 'anything', max_tokens: 256 });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  });

  it('should map recall results to MemoryResult shape', async () => {
    const bankId = randomBankId();
    client.setBankId(bankId);

    await client.retain({
      content:
        '[role: user]\nI enjoy reading science fiction books.\n[user:end]\n\n' +
        '[role: assistant]\nSounds like a great hobby!\n[assistant:end]',
      document_id: 'mapping-test',
    });

    const response = await client.recall({ query: 'What are my hobbies?', max_tokens: 1024 });

    for (const result of response.results) {
      expect(typeof result.id).toBe('string');
      expect(typeof result.text).toBe('string');
      expect(typeof result.type).toBe('string');
      expect(Array.isArray(result.entities)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// Embed Mode Tests (subprocess / daemon)
// ---------------------------------------------------------------------------

describe('HindsightClient – Embed Mode (Subprocess)', () => {
  let client: HindsightClient;
  let embedManager: HindsightEmbedManager;

  const hasEmbedCredentials = Boolean(LLM_PROVIDER && LLM_API_KEY);

  beforeAll(async () => {
    if (!hasEmbedCredentials) {
      console.warn(
        '[Integration] Skipping embed mode tests: ' +
          'HINDSIGHT_API_LLM_PROVIDER and HINDSIGHT_API_LLM_API_KEY must both be set.',
      );
      return;
    }

    embedManager = new HindsightEmbedManager(
      EMBED_TEST_PORT,
      LLM_PROVIDER,
      LLM_API_KEY,
      LLM_MODEL || undefined,
      undefined, // no custom base URL
      0, // never idle-timeout
      'latest',
      EMBED_PACKAGE_PATH,
    );

    await embedManager.start();

    client = new HindsightClient({
      llmProvider: LLM_PROVIDER,
      llmApiKey: LLM_API_KEY,
      llmModel: LLM_MODEL || undefined,
      embedPackagePath: EMBED_PACKAGE_PATH,
    });
  }, 120_000); // daemon startup can take up to 2 minutes

  afterAll(async () => {
    if (embedManager) {
      await embedManager.stop();
    }
  }, 30_000);

  it('should retain a conversation via subprocess', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.retain({
      content:
        '[role: user]\nI love hiking in the mountains.\n[user:end]\n\n' +
        '[role: assistant]\nSounds adventurous!\n[assistant:end]',
      document_id: 'embed-retain-test-1',
    });

    expect(response).toBeDefined();
    expect(response.message).toBeDefined();
    expect(response.document_id).toBe('embed-retain-test-1');
  }, 60_000);

  it('should retain with auto-generated document id via subprocess', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.retain({
      content: '[role: user]\nI am a TypeScript developer.\n[user:end]',
    });

    expect(response).toBeDefined();
    expect(response.document_id).toBe('conversation');
  }, 60_000);

  it('should recall from an empty bank without error via subprocess', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    const response = await client.recall({ query: 'What do I like?', max_tokens: 512 });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  }, 60_000);

  it('should set bank mission via subprocess without throwing', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    // Create bank by retaining first, then set mission
    await client.retain({ content: '[role: user]\nHello\n[user:end]' });

    await expect(
      client.setBankMission('Test mission for embed integration tests.'),
    ).resolves.not.toThrow();
  }, 60_000);

  it('should retain and then recall relevant memories via subprocess', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    await client.retain({
      content:
        '[role: user]\nMy cat is named Whiskers and she is 3 years old.\n[user:end]\n\n' +
        '[role: assistant]\nWhat a lovely name!\n[assistant:end]',
      document_id: `embed-e2e-${Date.now()}`,
    });

    const response = await client.recall({
      query: "What is my cat's name?",
      max_tokens: 1024,
    });

    expect(response).toBeDefined();
    expect(Array.isArray(response.results)).toBe(true);
  }, 60_000);

  it('should map recall results to MemoryResult shape via subprocess', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    await client.retain({
      content:
        '[role: user]\nI enjoy cooking Italian food.\n[user:end]\n\n' +
        '[role: assistant]\nItalian cuisine is delicious!\n[assistant:end]',
      document_id: 'embed-shape-test',
    });

    const response = await client.recall({ query: 'What food do I like?', max_tokens: 1024 });

    for (const result of response.results) {
      expect(typeof result.id).toBe('string');
      expect(typeof result.text).toBe('string');
      expect(typeof result.type).toBe('string');
      expect(Array.isArray(result.entities)).toBe(true);
    }
  }, 60_000);

  it('should handle full end-to-end workflow via subprocess', async () => {
    if (!hasEmbedCredentials) return;

    const bankId = randomBankId();
    client.setBankId(bankId);

    // Step 1: Retain
    const retainResp = await client.retain({
      content:
        '[role: user]\nI am learning Rust programming.\n[user:end]\n\n' +
        '[role: assistant]\nRust is a powerful systems language!\n[assistant:end]',
      document_id: `embed-workflow-${Date.now()}`,
      metadata: { channel_type: 'telegram', sender_id: '999' },
    });
    expect(retainResp).toBeDefined();

    // Step 2: Recall
    const recallResp = await client.recall({
      query: 'What am I learning?',
      max_tokens: 1024,
    });
    expect(recallResp).toBeDefined();
    expect(Array.isArray(recallResp.results)).toBe(true);
  }, 60_000);
});
