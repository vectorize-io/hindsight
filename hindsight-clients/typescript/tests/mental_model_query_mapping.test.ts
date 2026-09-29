/**
 * Unit tests for mental-model query forwarding in the hand-written wrapper.
 *
 * These mock the generated SDK so a regression cannot silently restore its
 * `detail=full` default or drop list pagination and tag-match controls.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedList = sdk.listMentalModels as jest.MockedFunction<typeof sdk.listMentalModels>;
const mockedGet = sdk.getMentalModel as jest.MockedFunction<typeof sdk.getMentalModel>;

describe("mental-model query mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedList.mockReset();
    mockedGet.mockReset();
    mockedList.mockResolvedValue({ data: { items: [], total: 0 } } as any);
    mockedGet.mockResolvedValue({ data: { id: "model-1" } } as any);
  });

  test("forwards every supported list query option", async () => {
    const signal = new AbortController().signal;

    await client.listMentalModels("bank-1", {
      tags: ["project"],
      tagsMatch: "exact",
      detail: "metadata",
      limit: 25,
      offset: 50,
      signal,
    });

    expect(mockedList).toHaveBeenCalledWith(
      expect.objectContaining({
        path: { bank_id: "bank-1" },
        query: {
          tags: ["project"],
          tags_match: "exact",
          detail: "metadata",
          limit: 25,
          offset: 50,
        },
        signal,
      })
    );
  });

  test("preserves optionless and tags-only list request shapes", async () => {
    await client.listMentalModels("bank-1");
    await client.listMentalModels("bank-1", { tags: ["project"] });

    expect(mockedList.mock.calls[0][0].query).toEqual({ tags: undefined });
    expect(mockedList.mock.calls[1][0].query).toEqual({ tags: ["project"] });
  });

  test("forwards detail when fetching one mental model", async () => {
    await client.getMentalModel("bank-1", "model-1", { detail: "content" });

    expect(mockedGet).toHaveBeenCalledWith(
      expect.objectContaining({
        path: { bank_id: "bank-1", mental_model_id: "model-1" },
        query: { detail: "content" },
      })
    );
  });

  test("preserves the optionless get request shape", async () => {
    await client.getMentalModel("bank-1", "model-1");

    expect(mockedGet).toHaveBeenCalledTimes(1);
    expect(mockedGet.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        path: { bank_id: "bank-1", mental_model_id: "model-1" },
      })
    );
    expect(mockedGet.mock.calls[0][0]).not.toHaveProperty("query");
  });
});
