/**
 * Unit tests for the updateMentalModel wrapper's trigger mapping.
 *
 * The generated sdk layer is mocked, so no server is needed: these assert that
 * the camelCase options reach the snake_case body, and — the point of the
 * suite — that a partial trigger stays partial. The server patches a supplied
 * trigger over the model's stored one and reads "named" from the fields the
 * request carried (#3506/#3549), so any field the caller did not set must be
 * absent from the body rather than filled in with a default.
 *
 * Mirrors tests/test_mental_model_trigger_patch.py on the Python side, where
 * the generated model's own defaults did ride along on every request.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedUpdate = sdk.updateMentalModel as jest.MockedFunction<typeof sdk.updateMentalModel>;

function lastBody(): any {
  return mockedUpdate.mock.calls[0][0].body;
}

/** The keys a JSON body actually carries — `undefined` values are dropped in transit. */
function sentTriggerKeys(): string[] {
  return Object.entries(lastBody().trigger ?? {})
    .filter(([, value]) => value !== undefined)
    .map(([key]) => key)
    .sort();
}

describe("updateMentalModel trigger mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedUpdate.mockReset();
    mockedUpdate.mockResolvedValue({
      data: { id: "mm-1", bank_id: "bank", name: "Model", tags: [] },
    } as any);
  });

  test("sends only the setting the caller named", async () => {
    await client.updateMentalModel("bank", "mm-1", { trigger: { mode: "delta" } });

    expect(lastBody().trigger.mode).toBe("delta");
    expect(sentTriggerKeys()).toEqual(["mode"]);
  });

  test("keeps an explicit false rather than treating it as an omission", async () => {
    await client.updateMentalModel("bank", "mm-1", {
      trigger: { refreshAfterConsolidation: false },
    });

    expect(sentTriggerKeys()).toEqual(["refresh_after_consolidation"]);
    expect(lastBody().trigger.refresh_after_consolidation).toBe(false);
  });

  test("maps an explicit 0 rather than dropping it as falsy", async () => {
    // 0 is a meaningful value here — it exempts one model from a bank-wide floor.
    await client.updateMentalModel("bank", "mm-1", { trigger: { minRefreshIntervalSeconds: 0 } });

    expect(sentTriggerKeys()).toEqual(["min_refresh_interval_seconds"]);
    expect(lastBody().trigger.min_refresh_interval_seconds).toBe(0);
  });

  test("sends an explicit null cron, which is how a schedule is removed", async () => {
    await client.updateMentalModel("bank", "mm-1", { trigger: { refreshCron: null } });

    expect(sentTriggerKeys()).toEqual(["refresh_cron"]);
    expect(lastBody().trigger.refresh_cron).toBeNull();
  });

  test("exposes the settings the update used to have no way to reach", async () => {
    await client.updateMentalModel("bank", "mm-1", {
      trigger: {
        refreshCron: "0 3 * * *",
        tagsMatch: "any_strict",
        keepTrace: true,
        excludeMentalModels: true,
        factTypes: ["observation"],
        includeChunks: true,
        recallMaxTokens: 4096,
        recallChunksMaxTokens: 2048,
        responseSchema: { type: "object" },
        excludeMentalModelIds: ["mm-other"],
        tagGroups: [{ tags: ["user:alice"], match: "all_strict" }],
      },
    });

    const trigger = lastBody().trigger;
    expect(trigger.refresh_cron).toBe("0 3 * * *");
    expect(trigger.tags_match).toBe("any_strict");
    expect(trigger.keep_trace).toBe(true);
    expect(trigger.exclude_mental_models).toBe(true);
    expect(trigger.fact_types).toEqual(["observation"]);
    expect(trigger.include_chunks).toBe(true);
    expect(trigger.recall_max_tokens).toBe(4096);
    expect(trigger.recall_chunks_max_tokens).toBe(2048);
    expect(trigger.response_schema).toEqual({ type: "object" });
    expect(trigger.exclude_mental_model_ids).toEqual(["mm-other"]);
    expect(trigger.tag_groups).toEqual([{ tags: ["user:alice"], match: "all_strict" }]);
    // Still no mode: it was not asked for, and the stored one must survive.
    expect(sentTriggerKeys()).not.toContain("mode");
  });

  test("omitting trigger sends no trigger at all", async () => {
    await client.updateMentalModel("bank", "mm-1", { name: "Renamed" });

    expect(lastBody().trigger).toBeUndefined();
  });
});
