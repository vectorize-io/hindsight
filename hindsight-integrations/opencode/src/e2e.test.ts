/**
 * Gated live end-to-end test for the OpenCode Hindsight plugin.
 *
 * Drives the plugin's exported tools + hooks against a live Hindsight server
 * the way the OpenCode runtime would. Skipped by default; runs only when
 * HINDSIGHT_LIVE_E2E=1 (analogous to the `requires_real_llm` pytest marker
 * used by the Python integrations).
 *
 * Run with:
 *   HINDSIGHT_API_URL=http://127.0.0.1:8888 npm run test:e2e
 *
 * Requires:
 *   - A reachable Hindsight server (defaults to http://127.0.0.1:8888).
 *   - When pointing at the hosted backend: HINDSIGHT_API_TOKEN with a valid key.
 */
import { afterAll, beforeAll, describe, it, expect } from "vitest";
import { randomBytes } from "node:crypto";

import { HindsightPlugin } from "./index.js";
import { HindsightClient } from "@vectorize-io/hindsight-client";

const LIVE = process.env.HINDSIGHT_LIVE_E2E === "1";
const URL = process.env.HINDSIGHT_API_URL || "http://127.0.0.1:8888";
const BANK = `e2e-opencode-${randomBytes(4).toString("hex")}`;

function mockOpencodeSessionMessages(messages: Array<{ role: string; content: string }>) {
  return {
    session: {
      async messages() {
        return {
          data: messages.map((m) => ({
            info: { role: m.role },
            parts: [{ type: "text", text: m.content }],
          })),
        };
      },
    },
  };
}

const describeLive = LIVE ? describe : describe.skip;

describeLive("live: OpenCode plugin against Hindsight", () => {
  // When pointing at the hosted backend, the suite's direct (non-plugin)
  // retain/recall/deleteBank calls need the same token the plugin reads from
  // HINDSIGHT_API_TOKEN — otherwise they 401 even when the plugin path is fine.
  const TOKEN = process.env.HINDSIGHT_API_TOKEN;
  const client = new HindsightClient(TOKEN ? { baseUrl: URL, apiKey: TOKEN } : { baseUrl: URL });

  afterAll(async () => {
    try {
      await client.deleteBank(BANK);
    } catch {
      // bank may not exist if a test bailed early; harmless
    }
  });

  it("retain → server-side extraction → recall via tools surfaces the stored fact", async () => {
    const plugin = await HindsightPlugin(
      {
        client: mockOpencodeSessionMessages([]) as any,
        directory: "/tmp/fake-project",
      } as any,
      { hindsightApiUrl: URL, bankId: BANK, debug: false }
    );

    const retainOut = await plugin.tool!.hindsight_retain.execute(
      { content: "User's favourite programming language is Haskell." } as any,
      {} as any
    );
    expect(String(retainOut)).toMatch(/stored/i);

    // Server-side fact extraction is asynchronous; give it time before recall.
    await new Promise((r) => setTimeout(r, 6000));

    const recallOut = await plugin.tool!.hindsight_recall.execute(
      { query: "favourite programming language" } as any,
      {} as any
    );
    expect(String(recallOut).toLowerCase()).toContain("haskell");
  }, 30_000);

  it("session.idle → auto-retain captures the transcript", async () => {
    const sessionId = "idle-test-session";
    const fakeMessages = [
      { role: "user", content: "I prefer dark mode and use VS Code." },
      { role: "assistant", content: "Noted, dark mode in VS Code." },
    ];

    const plugin = await HindsightPlugin(
      {
        client: mockOpencodeSessionMessages(fakeMessages) as any,
        directory: "/tmp/fake-project",
      } as any,
      {
        hindsightApiUrl: URL,
        bankId: BANK,
        retainEveryNTurns: 1,
        debug: false,
      }
    );

    await plugin.event!({
      event: { type: "session.idle", properties: { sessionID: sessionId } },
    } as any);

    // Give the auto-retain RPC and the server-side extraction time to land.
    await new Promise((r) => setTimeout(r, 6000));

    const direct = await client.recall(BANK, "IDE preferences");
    const texts = (direct.results || []).map((r) => r.text.toLowerCase()).join(" | ");
    expect(texts).toMatch(/vs code|dark mode/);
  }, 30_000);

  it("session.created + system transform injects recalled context on first prompt", async () => {
    const sessionId = "first-prompt-session";

    // Seed with content that matches the hardcoded system-transform query
    // ("project context and recent work").
    await client.retain(
      BANK,
      "Project context: TypeScript monorepo; recent work was on the hindsight-opencode plugin's Cloud-default config."
    );
    // Give the server-side extraction time to index the new content.
    await new Promise((r) => setTimeout(r, 6000));

    const plugin = await HindsightPlugin(
      {
        client: mockOpencodeSessionMessages([]) as any,
        directory: "/tmp/fake-project",
      } as any,
      { hindsightApiUrl: URL, bankId: BANK, debug: false }
    );

    // First, session.created → marks the session as awaiting first-prompt recall
    await plugin.event!({
      event: {
        type: "session.created",
        properties: { info: { id: sessionId, title: "t" } },
      },
    } as any);

    // Then system.transform should inject (bank has matching project context).
    const sysOut = { system: [] as string[] };
    await plugin["experimental.chat.system.transform"]!(
      { sessionID: sessionId, model: {} } as any,
      sysOut
    );

    expect(sysOut.system.length).toBeGreaterThan(0);
    expect(sysOut.system.join("\n")).toMatch(/hindsight_memories/);
  }, 30_000);
});
