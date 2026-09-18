/**
 * The list time window's request mapping (#4349).
 *
 * Like the other *_mapping tests these do NOT require a running server: the
 * generated sdk layer is mocked so we can assert the camelCase options land on
 * the snake_case query string.
 *
 * Both wrappers enumerate query parameters rather than passing a dict through,
 * so one added to the API but not to a wrapper is silently dropped for every
 * consumer of that language — and `client-coverage-check` only validates request
 * bodies, so a GET query parameter has nothing but this test behind it. The
 * Python wrapper has the mirror of this file in
 * tests/test_list_time_window_mapping.py; the two must stay in step.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedListMemories = sdk.listMemories as jest.MockedFunction<typeof sdk.listMemories>;
const mockedListDocuments = sdk.listDocuments as jest.MockedFunction<typeof sdk.listDocuments>;

describe("list time window mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedListMemories.mockReset();
    mockedListMemories.mockResolvedValue({
      data: { items: [], total: 0, limit: 100, offset: 0 },
    } as any);
    mockedListDocuments.mockReset();
    mockedListDocuments.mockResolvedValue({
      data: { items: [], total: 0, limit: 100, offset: 0 },
    } as any);
  });

  test("listMemories maps the window onto the query", async () => {
    await client.listMemories("bank", {
      timeField: "mentioned_at",
      startDate: "2024-01-01T00:00:00Z",
      endDate: "2024-02-01T00:00:00Z",
    });

    const query = mockedListMemories.mock.calls[0][0].query as any;
    expect(query.time_field).toBe("mentioned_at");
    expect(query.start_date).toBe("2024-01-01T00:00:00Z");
    expect(query.end_date).toBe("2024-02-01T00:00:00Z");
  });

  test("listDocuments maps the window onto the query", async () => {
    await client.listDocuments("bank", {
      timeField: "created_at",
      startDate: "2024-01-01T00:00:00Z",
    });

    const query = mockedListDocuments.mock.calls[0][0].query as any;
    expect(query.time_field).toBe("created_at");
    expect(query.start_date).toBe("2024-01-01T00:00:00Z");
  });

  test("the window is absent unless asked for", async () => {
    await client.listMemories("bank");

    const query = mockedListMemories.mock.calls[0][0].query as any;
    // undefined rather than a default axis: the endpoint keeps its own ordering
    // and drops nothing unless a caller asks for a window.
    expect(query.time_field).toBeUndefined();
    expect(query.start_date).toBeUndefined();
    expect(query.end_date).toBeUndefined();
  });
});
