/**
 * Unit tests for `rerank` forwarding in the hand-written wrapper.
 *
 * Mirrors the Python wrapper's `test_recall_rerank.py`: the decision-model options reach the
 * request body as one nested `rerank` object, and are left out when the caller omits them so
 * the bank's defaults decide.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedRecall = sdk.recallMemories as jest.MockedFunction<typeof sdk.recallMemories>;

describe("recall rerank mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedRecall.mockReset();
    mockedRecall.mockResolvedValue({ data: { results: [] } } as any);
  });

  test("recall forwards prune and dedup inside rerank", async () => {
    await client.recall("bank-1", "what tea does Marta like", {
      rerank: { prune: true, dedup: false },
    });

    expect(mockedRecall).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({ rerank: { prune: true, dedup: false } }),
      })
    );
  });

  test("recall leaves rerank out when omitted", async () => {
    await client.recall("bank-1", "what tea does Marta like");

    expect(mockedRecall.mock.calls[0][0].body.rerank).toBeUndefined();
  });
});
