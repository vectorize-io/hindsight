import { expect, it } from "vitest";
import plugin from "./index.js";
import type { MoltbotPluginAPI } from "./types.js";

// OpenClaw refuses to start or resume a Goal while any plugin registers one of
// these hooks, because it can't replay them after a restart
// (src/plugins/restart-recovery-hook-safety.ts). Holding one blocked Goals. (#5177)
const GOAL_UNSAFE_HOOKS = [
  "before_dispatch",
  "before_agent_run",
  "before_message_write",
  "reply_dispatch",
];

it("registers no hook that blocks OpenClaw Goals", () => {
  const registered: string[] = [];
  plugin({
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: { hindsightApiUrl: "http://localhost:8888", logLevel: "error" },
          },
        },
      },
    },
    registerService: () => undefined,
    on: (name: string) => {
      registered.push(name);
    },
    logger: { info: () => undefined, warn: () => undefined, error: () => undefined },
  } as unknown as MoltbotPluginAPI);

  expect(registered).toContain("message_received");
  expect(registered.filter((name) => GOAL_UNSAFE_HOOKS.includes(name))).toEqual([]);
});
