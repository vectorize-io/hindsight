/**
 * Unit tests for the createMentalModel wrapper's trigger mapping.
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

  test("omitting trigger sends no trigger (preserves the all_strict default)", async () => {
    await client.createMentalModel("bank", "Plain", "q");

    expect(lastBody().trigger).toBeUndefined();
  });
});
