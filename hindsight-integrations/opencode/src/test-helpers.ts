import type { HindsightConfig } from "./config.js";

export function makeConfig(overrides: Partial<HindsightConfig> = {}): HindsightConfig {
  return {
    autoRecall: true,
    recallBudget: "mid",
    recallMaxTokens: 1024,
    recallTypes: ["world", "experience"],
    recallContextTurns: 1,
    recallMaxQueryChars: 800,
    recallPromptPreamble: "",
    recallTags: [],
    recallTagsMatch: "any",
    autoRetain: true,
    retainMode: "full-session",
    retainEveryNTurns: 10,
    retainOverlapTurns: 2,
    retainContext: "opencode",
    retainTags: [],
    retainMetadata: {},
    hindsightApiUrl: null,
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
