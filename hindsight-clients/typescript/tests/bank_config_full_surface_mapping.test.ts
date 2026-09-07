/**
 * Unit tests for the full bank-config request mapping (#4029).
 *
 * `updateBankConfig` enumerates config fields rather than passing a dict through,
 * so a field accepted as an option but never written into `updates` is silently
 * dropped: the caller gets a 200 and no change. Before #4029 this wrapper reached
 * only 17 of the server's 47 configurable fields — the whole recall-budget group,
 * memory defense and the consolidation knobs were unreachable from TypeScript.
 *
 * The Python wrapper has the mirror of this file in
 * tests/test_bank_config_update_payload.py; the two must stay in step.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedUpdateConfig = sdk.updateBankConfig as jest.MockedFunction<typeof sdk.updateBankConfig>;
const mockedReflect = sdk.reflect as jest.MockedFunction<typeof sdk.reflect>;

describe("full bank config surface mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedUpdateConfig.mockReset();
    mockedUpdateConfig.mockResolvedValue({
      data: { bank_id: "bank", config: {}, overrides: {} },
    } as any);
    mockedReflect.mockReset();
    mockedReflect.mockResolvedValue({ data: { text: "ok" } } as any);
  });

  test("maps the recall budget group onto the updates map", async () => {
    await client.updateBankConfig("bank", {
      recallMaxTokens: 4096,
      recallIncludeChunks: false,
      recallChunksMaxTokens: 500,
      recallBudgetFunction: "adaptive",
      recallBudgetFixedLow: 50,
      recallBudgetFixedMid: 150,
      recallBudgetFixedHigh: 500,
      recallBudgetAdaptiveLow: 0.01,
      recallBudgetAdaptiveMid: 0.05,
      recallBudgetAdaptiveHigh: 0.2,
      recallBudgetMin: 10,
      recallBudgetMax: 1000,
    });

    const body = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(body.updates).toEqual({
      recall_max_tokens: 4096,
      recall_include_chunks: false,
      recall_chunks_max_tokens: 500,
      recall_budget_function: "adaptive",
      recall_budget_fixed_low: 50,
      recall_budget_fixed_mid: 150,
      recall_budget_fixed_high: 500,
      recall_budget_adaptive_low: 0.01,
      recall_budget_adaptive_mid: 0.05,
      recall_budget_adaptive_high: 0.2,
      recall_budget_min: 10,
      recall_budget_max: 1000,
    });
  });

  test("maps the consolidation, retain and security groups onto the updates map", async () => {
    await client.updateBankConfig("bank", {
      enableAutoConsolidation: false,
      consolidationLlmBatchSize: 4,
      consolidationLlmParallelism: 2,
      consolidationMaxMemoriesPerRound: 50,
      consolidationSourceFactsMaxTokens: 2048,
      consolidationSourceFactsMaxTokensPerObservation: 128,
      mentalModelMinRefreshIntervalSeconds: 3600,
      reflectSourceFactsMaxTokens: 1024,
      maxObservationsPerScope: 25,
      observationScopeLimits: [{ scope: "team", limit: 10 }],
      retainDefaultStrategy: "fast",
      retainStrategies: { fast: { retain_extraction_mode: "concise" } },
      retainChunkBatchSize: 64,
      storeDocumentText: false,
      mcpEnabledTools: ["recall", "reflect"],
      llmGeminiSafetySettings: [{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" }],
      memoryDefense: { redact_secrets: true },
      auditLogEnabled: true,
    });

    const body = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(body.updates).toEqual({
      enable_auto_consolidation: false,
      consolidation_llm_batch_size: 4,
      consolidation_llm_parallelism: 2,
      consolidation_max_memories_per_round: 50,
      consolidation_source_facts_max_tokens: 2048,
      consolidation_source_facts_max_tokens_per_observation: 128,
      mental_model_min_refresh_interval_seconds: 3600,
      reflect_source_facts_max_tokens: 1024,
      max_observations_per_scope: 25,
      observation_scope_limits: [{ scope: "team", limit: 10 }],
      retain_default_strategy: "fast",
      retain_strategies: { fast: { retain_extraction_mode: "concise" } },
      retain_chunk_batch_size: 64,
      store_document_text: false,
      mcp_enabled_tools: ["recall", "reflect"],
      llm_gemini_safety_settings: [
        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
      ],
      memory_defense: { redact_secrets: true },
      audit_log_enabled: true,
    });
  });

  test("sends only the fields that were set", async () => {
    await client.updateBankConfig("bank", { recallMaxTokens: 100 });

    const body = mockedUpdateConfig.mock.calls[0][0].body as any;
    expect(body.updates).toEqual({ recall_max_tokens: 100 });
  });

  test("reflect forwards applyAllDirectives", async () => {
    await client.reflect("bank", "why?", { applyAllDirectives: true });

    const body = mockedReflect.mock.calls[0][0].body as any;
    expect(body.apply_all_directives).toBe(true);
  });

  test("reflect omits applyAllDirectives when unset", async () => {
    await client.reflect("bank", "why?", {});

    const body = mockedReflect.mock.calls[0][0].body as any;
    expect(body.apply_all_directives).toBeUndefined();
  });
});
