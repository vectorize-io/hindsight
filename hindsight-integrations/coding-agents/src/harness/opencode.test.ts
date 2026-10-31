/**
 * The opencode adapter's `config` hook: it teaches the host about the survey agent, so the recipe
 * in core/survey.ts (`opencode run --agent hindsight-survey`) needs nothing in the user's
 * opencode.json. See SURVEY_AGENT in core/survey.ts for why the built-in `plan` agent can't do it.
 *
 * Also its `chat.message` hook, which puts the per-turn memory block into the user message.
 */
import { describe, expect, it, vi } from "vitest";
import { opencodeAdapter } from "./opencode";
import type { RuntimeCore } from "../core/runtime";
import { SURVEY_AGENT, SURVEY_AGENT_CONFIG } from "../core/survey";

/** Only the members createRuntime touches when it builds the hook object. */
function fakeCore() {
  return { harness: "opencode", toolSpecs: () => [] } as unknown as RuntimeCore;
}

type ConfigHook = (cfg: { agent?: Record<string, unknown> }) => Promise<void>;

function configHook(): ConfigHook {
  const runtime = opencodeAdapter.createRuntime(fakeCore()) as { config?: ConfigHook };
  if (!runtime.config) throw new Error("adapter exposes no config hook");
  return runtime.config;
}

describe("opencode config hook", () => {
  it("defines the survey agent on a config that has no agents at all", async () => {
    const cfg: { agent?: Record<string, unknown> } = {};
    await configHook()(cfg);
    expect(cfg.agent?.[SURVEY_AGENT]).toEqual(SURVEY_AGENT_CONFIG);
  });

  it("leaves the user's other agents alone", async () => {
    const mine = { description: "mine" };
    const cfg = { agent: { reviewer: mine } as Record<string, unknown> };
    await configHook()(cfg);
    expect(cfg.agent.reviewer).toBe(mine);
    expect(cfg.agent[SURVEY_AGENT]).toEqual(SURVEY_AGENT_CONFIG);
  });

  it("does NOT overwrite a hindsight-survey agent the user defined themselves", async () => {
    // Their file is the last word: someone who redefines this name has a reason, and silently
    // replacing it would be a plugin overriding local config.
    const theirs = { description: "customised", mode: "primary" };
    const cfg = { agent: { [SURVEY_AGENT]: theirs } as Record<string, unknown> };
    await configHook()(cfg);
    expect(cfg.agent[SURVEY_AGENT]).toBe(theirs);
  });
});

describe("opencode per-turn injection", () => {
  type Part = Record<string, unknown>;
  type ChatMessage = (
    input: { sessionID: string },
    output: { message: { id: string }; parts: Part[] }
  ) => Promise<void>;

  function runtimeWith(injection: string | undefined) {
    const core = {
      harness: "opencode",
      toolSpecs: () => [],
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => injection),
    };
    const runtime = opencodeAdapter.createRuntime(core as unknown as RuntimeCore) as Record<
      string,
      unknown
    >;
    return { core, runtime, chatMessage: runtime["chat.message"] as ChatMessage };
  }

  it("adds this turn's injection to the user message, never to the system prompt", async () => {
    // The block changes between turns, so a system prompt carrying it broke the provider's prefix
    // cache (#5914). No system.transform hook means nothing of ours can reach the system prompt.
    const { core, runtime, chatMessage } = runtimeWith("<hindsight_memory>m</hindsight_memory>");
    const prompt = { id: "prt_1", type: "text", text: "why do we retry?" };
    const output = { message: { id: "msg_1" }, parts: [prompt] as Part[] };
    await chatMessage({ sessionID: "ses_1" }, output);
    expect(core.onPrompt).toHaveBeenCalledWith("ses_1", "why do we retry?");
    expect(output.parts).toEqual([
      {
        id: "prt-hindsight-msg_1",
        sessionID: "ses_1",
        messageID: "msg_1",
        type: "text",
        text: "<hindsight_memory>m</hindsight_memory>",
        synthetic: true,
      },
      prompt,
    ]);
    expect(runtime["experimental.chat.system.transform"]).toBeUndefined();
  });

  it("leaves the user message alone when there is nothing to inject", async () => {
    const { chatMessage } = runtimeWith(undefined);
    const output = { message: { id: "msg_1" }, parts: [{ type: "text", text: "hi" }] as Part[] };
    await chatMessage({ sessionID: "ses_1" }, output);
    expect(output.parts).toEqual([{ type: "text", text: "hi" }]);
  });
});
