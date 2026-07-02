import type { HindsightConfig } from "./config.js";

export function makeConfig(overrides: Partial<HindsightConfig> = {}): HindsightConfig {
  return {
    autoRecall: true,
    recallBudget: "mid",
    recallMaxTokens: 1024,
    recallTypes: ["observation", "world", "experience"],
    recallContextTurns: 1,
    recallMaxQueryChars: 800,
    recallTags: [],
    recallTagsMatch: "any",
    autoRetain: true,
    retainContext: "conversation between OpenCode Agent and the User",
    retainTags: [],
    retainMetadata: {},
    hindsightApiUrl: "https://api.hindsight.vectorize.io",
    hindsightApiToken: null,
    bankId: null,
    bankIdPrefix: "",
    dynamicBankId: false,
    dynamicBankGranularity: ["agent", "project"],
    bankMission: "",
    retainMission: null,
    agentName: "opencode",
    debug: false,
    ...overrides,
  };
}
