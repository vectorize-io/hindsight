/**
 * Unit tests for reflect's abstain and search_queries options in the hand-written wrapper.
 *
 * Mirrors the Python wrapper's `test_reflect_abstain_options.py`. A wrapper that dropped
 * `abstain` would hand a memory watcher a written answer for every moment it asks about.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedReflect = sdk.reflect as jest.MockedFunction<typeof sdk.reflect>;

describe("reflect abstain options mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedReflect.mockReset();
    mockedReflect.mockResolvedValue({ data: { text: "", abstained: true } } as any);
  });

  test("omitting the options leaves them unset", async () => {
    await client.reflect("bank-1", "does my change go against a decision?");

    const body = mockedReflect.mock.calls[0][0].body as Record<string, unknown>;
    expect(body.abstain).toBeUndefined();
    expect(body.search_queries).toBeUndefined();
  });

  test("both options reach the request body", async () => {
    await client.reflect("bank-1", "does my change go against a decision?", {
      abstain: true,
      searchQueries: ["parse_quantity units", "disk quotas"],
    });

    expect(mockedReflect).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({
          abstain: true,
          search_queries: ["parse_quantity units", "disk quotas"],
        }),
      })
    );
  });
});
