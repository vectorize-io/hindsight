/**
 * The opencode adapter's `config` hook: it teaches the host about the survey agent, so the recipe
 * in core/survey.ts (`opencode run --agent hindsight-survey`) needs nothing in the user's
 * opencode.json. See SURVEY_AGENT in core/survey.ts for why the built-in `plan` agent can't do it.
 */
import { describe, expect, it } from "vitest";
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
