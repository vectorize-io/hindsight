/**
 * Unit tests for the search_observations budget options in the hand-written wrapper.
 *
 * Mirrors the Python wrapper's `test_reflect_search_observations_options.py`. These are
 * per-call overrides of the bank's `reflect_default_options` (#4483); a wrapper that
 * dropped them would leave every TypeScript consumer stuck on the bank default with
 * nothing in the response to say the argument was ignored.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedReflect = sdk.reflect as jest.MockedFunction<typeof sdk.reflect>;

describe("reflect search_observations options mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedReflect.mockReset();
    mockedReflect.mockResolvedValue({ data: { text: "" } } as any);
  });

  test("omitting the options leaves them unset so the bank decides", async () => {
    await client.reflect("bank-1", "what did we decide");

    const body = mockedReflect.mock.calls[0][0].body as Record<string, unknown>;
    expect(body.reflect_search_observations_max_tokens).toBeUndefined();
    expect(body.reflect_search_observations_include_entities).toBeUndefined();
  });

  test("both options reach the request body", async () => {
    await client.reflect("bank-1", "what did we decide", {
      reflectSearchObservationsMaxTokens: 3000,
      reflectSearchObservationsIncludeEntities: false,
    });

    expect(mockedReflect).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({
          reflect_search_observations_max_tokens: 3000,
          reflect_search_observations_include_entities: false,
        }),
      })
    );
  });
});
