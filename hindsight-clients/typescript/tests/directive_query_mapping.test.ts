/**
 * Unit tests for directive query forwarding in the hand-written wrapper.
 *
 * The directive list endpoint reports a `total`, so a regression that dropped
 * `limit`/`offset` here would strand SDK callers on the server's first page.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedList = sdk.listDirectives as jest.MockedFunction<typeof sdk.listDirectives>;

describe("directive query mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedList.mockReset();
    mockedList.mockResolvedValue({ data: { items: [], total: 0 } } as any);
  });

  test("forwards every supported list query option", async () => {
    const signal = new AbortController().signal;

    await client.listDirectives("bank-1", { tags: ["project"], limit: 25, offset: 50, signal });

    expect(mockedList).toHaveBeenCalledWith(
      expect.objectContaining({
        path: { bank_id: "bank-1" },
        query: { tags: ["project"], limit: 25, offset: 50 },
        signal,
      })
    );
  });

  test("preserves optionless and tags-only list request shapes", async () => {
    await client.listDirectives("bank-1");
    await client.listDirectives("bank-1", { tags: ["project"] });

    expect(mockedList.mock.calls[0][0].query).toEqual({ tags: undefined });
    expect(mockedList.mock.calls[1][0].query).toEqual({ tags: ["project"] });
  });
});
