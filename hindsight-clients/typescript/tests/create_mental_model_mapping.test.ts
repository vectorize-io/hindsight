/**
 * Unit tests for the createMentalModel wrapper's option mapping.
 *
 * Unlike the other suites, these do NOT require a running server: the generated
 * sdk layer is mocked so we can assert the ergonomic camelCase options are
 * mapped onto the snake_case request body. Regression cover for #2808, where
 * the wrapper dropped every trigger field except refreshAfterConsolidation and
 * so a caller could not set tags_match.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedCreate = sdk.createMentalModel as jest.MockedFunction<typeof sdk.createMentalModel>;

function lastBody(): any {
  return mockedCreate.mock.calls[0][0].body;
}

describe("createMentalModel trigger mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedCreate.mockReset();
    mockedCreate.mockResolvedValue({
      data: { mental_model_id: "mm-1", operation_id: "op-1" },
    } as any);
  });

  test("threads tagsMatch into trigger.tags_match", async () => {
    await client.createMentalModel("bank", "Projects", "Which projects?", {
      tags: ["projects", "mental-model"],
      trigger: { tagsMatch: "any" },
    });

    expect(mockedCreate).toHaveBeenCalledTimes(1);
    const body = lastBody();
    expect(body.tags).toEqual(["projects", "mental-model"]);
    expect(body.trigger.tags_match).toBe("any");
  });

  test("threads tagGroups into trigger.tag_groups", async () => {
    await client.createMentalModel("bank", "Scoped", "q", {
      trigger: { tagGroups: [{ tags: ["user:alice"], match: "all_strict" }] },
    });

    expect(lastBody().trigger.tag_groups).toEqual([{ tags: ["user:alice"], match: "all_strict" }]);
  });

  test("still maps refreshAfterConsolidation", async () => {
    await client.createMentalModel("bank", "Auto", "q", {
      trigger: { refreshAfterConsolidation: true },
    });

    expect(lastBody().trigger.refresh_after_consolidation).toBe(true);
  });

  test("threads minRefreshIntervalSeconds into trigger.min_refresh_interval_seconds", async () => {
    await client.createMentalModel("bank", "Paced", "q", {
      trigger: { refreshAfterConsolidation: true, minRefreshIntervalSeconds: 1800 },
    });

    expect(lastBody().trigger.min_refresh_interval_seconds).toBe(1800);
  });

  test("maps an explicit 0 rather than dropping it as falsy", async () => {
    // 0 is a meaningful value here — it exempts one model from a bank-wide floor —
    // so the mapping must not treat it like an omitted field.
    await client.createMentalModel("bank", "Hot", "q", {
      trigger: { minRefreshIntervalSeconds: 0 },
    });

    expect(lastBody().trigger.min_refresh_interval_seconds).toBe(0);
  });

  test("omitting trigger sends no trigger (preserves the all_strict default)", async () => {
    await client.createMentalModel("bank", "Plain", "q");

    expect(lastBody().trigger).toBeUndefined();
  });

  test("threads authored content into the request body", async () => {
    // The point of authored content is that the exact text travels; a wrapper that
    // forgot the field would still create a model — just a generated one.
    await client.createMentalModel("bank", "Deployment", "What are the conventions?", {
      content: "## Conventions\n\n- Ship small\n",
      trigger: { mode: "delta" },
    });

    expect(lastBody().content).toBe("## Conventions\n\n- Ship small\n");
    expect(lastBody().trigger.mode).toBe("delta");
  });

  test("omitting content sends no content (the server generates it)", async () => {
    await client.createMentalModel("bank", "Plain", "q");

    expect(lastBody().content).toBeUndefined();
  });
});
