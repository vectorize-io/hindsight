import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import registerPlugin, {
  prepareRetentionTranscript,
  retainStartIndex,
  runStartMarker,
} from "./index.js";
import type { MoltbotPluginAPI, PluginHookAgentContext, ServiceConfig } from "./types.js";

// #5569: a user writing again while the agent works puts two user messages in one
// run. Saving from the last one dropped the first request and the work on it.

const dirs: string[] = [];
let service: ServiceConfig | undefined;

afterEach(async () => {
  await service?.stop();
  service = undefined;
  vi.unstubAllGlobals();
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

type Handler = (event: unknown, ctx?: PluginHookAgentContext) => unknown;

async function loadPlugin(extraConfig: Record<string, unknown> = {}) {
  const dir = mkdtempSync(join(tmpdir(), "hindsight-run-start-"));
  dirs.push(dir);
  const retained: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      if (request.url.endsWith("/version")) {
        return Response.json({ api_version: "1.1.0", features: { store_document_text: true } });
      }
      if (request.method === "POST" && request.url.includes("/memories")) {
        const body = (await request.json()) as { items: Array<{ content: string }> };
        retained.push(...body.items.map((item) => item.content));
        return Response.json({ success: true, bank_id: "b", items_count: 1, async: true });
      }
      return Response.json({});
    })
  );
  const hooks = new Map<string, Handler>();
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "https://hindsight.test",
              retainQueuePath: join(dir, "queue.jsonl"),
              dynamicBankId: false,
              bankId: "b",
              autoRecall: false,
              logLevel: "off",
              ...extraConfig,
            },
          },
        },
      },
    },
    registerService: (registered) => {
      service = registered;
    },
    on: (name, handler) => {
      hooks.set(name, handler as Handler);
    },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  };
  registerPlugin(api);
  await service!.start();
  return { hooks, retained };
}

const ctx = {
  agentId: "main",
  sessionKey: "agent:main:discord:direct:u1",
  messageProvider: "discord",
  channelId: "direct:u1",
  senderId: "u1",
} as PluginHookAgentContext;

const earlier = [
  { role: "user", content: "Old question" },
  { role: "assistant", content: "Old answer" },
];
const first = { role: "user", content: "Research flights to Lisbon" };
const progress = { role: "assistant", content: "Searching the airline sites now." };
const steer = { role: "user", content: "Only direct flights please" };
const answer = { role: "assistant", content: "The direct TAP flight at 9:05 fits." };

describe("retain saves the whole run (#5569)", () => {
  it("keeps the first request when the user writes again mid-run", async () => {
    const { hooks, retained } = await loadPlugin();
    await hooks.get("before_prompt_build")!(
      { prompt: first.content, currentUserMessage: first.content, messages: [...earlier, first] },
      ctx
    );
    // A second build inside the same run must not move the start.
    await hooks.get("before_prompt_build")!(
      {
        prompt: steer.content,
        currentUserMessage: steer.content,
        messages: [...earlier, first, progress, steer],
      },
      ctx
    );
    await hooks.get("agent_end")!(
      { success: true, messages: [...earlier, first, progress, steer, answer] },
      ctx
    );

    expect(retained).toHaveLength(1);
    expect(retained[0]).toContain("Research flights to Lisbon");
    expect(retained[0]).toContain("Searching the airline sites now.");
    expect(retained[0]).toContain("Only direct flights please");
    expect(retained[0]).not.toContain("Old question");
  });

  it("keeps the request when the run used tools (tool results are user-role messages)", async () => {
    const { hooks, retained } = await loadPlugin();
    const toolCall = {
      role: "assistant",
      content: [{ type: "tool_use", id: "t1", name: "exec", input: { cmd: "ls" } }],
    };
    const toolResult = {
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "t1", content: "flights.txt" }],
    };
    await hooks.get("before_prompt_build")!(
      { messages: [first], currentUserMessage: first.content },
      ctx
    );
    await hooks.get("agent_end")!(
      { success: true, messages: [first, toolCall, toolResult, answer] },
      ctx
    );

    expect(retained).toHaveLength(1);
    expect(retained[0]).toContain("Research flights to Lisbon");
  });

  it("starts the next run where the previous one ended", async () => {
    const { hooks, retained } = await loadPlugin();
    const run1 = [first, progress];
    await hooks.get("before_prompt_build")!(
      { messages: [first], currentUserMessage: first.content },
      ctx
    );
    await hooks.get("agent_end")!({ success: true, messages: run1 }, ctx);
    await hooks.get("before_prompt_build")!(
      { messages: run1, currentUserMessage: steer.content },
      ctx
    );
    await hooks.get("agent_end")!({ success: true, messages: [...run1, steer, answer] }, ctx);

    expect(retained).toHaveLength(2);
    expect(retained[1]).toContain("Only direct flights please");
    expect(retained[1]).not.toContain("Research flights to Lisbon");
  });
});

describe("runStartMarker / retainStartIndex", () => {
  it("treats the current request as part of the run, whether or not it is in the history", () => {
    const withCurrent = runStartMarker([...earlier, first], first.content);
    const withoutCurrent = runStartMarker(earlier, first.content);
    expect(withCurrent).toBe(withoutCurrent);
    expect(retainStartIndex([...earlier, first, answer], withCurrent)).toBe(2);
  });

  it("starts at 0 for a run on an empty session", () => {
    expect(runStartMarker([first], first.content)).toBeNull();
    expect(retainStartIndex([first, answer], null)).toBe(0);
  });

  it("is unknown when the run start was rewritten away (compaction)", () => {
    const marker = runStartMarker(earlier, first.content);
    expect(
      retainStartIndex([{ role: "system", content: "Summary" }, first], marker)
    ).toBeUndefined();
    expect(retainStartIndex([first], undefined)).toBeUndefined();
  });
});

describe("prepareRetentionTranscript with a run start", () => {
  const messages = [...earlier, first, progress, steer, answer];

  it("saves from the run start", () => {
    const result = prepareRetentionTranscript(messages, { retainFormat: "text" }, false, 2);
    expect(result?.transcript).toContain("Research flights to Lisbon");
  });

  it("never saves less than from the last user message", () => {
    const result = prepareRetentionTranscript(messages, { retainFormat: "text" }, false, 6);
    expect(result?.transcript).toContain("Only direct flights please");
    expect(result?.transcript).not.toContain("Research flights to Lisbon");
  });
});
