/**
 * Unit tests for tag_groups forwarding in the hand-written wrapper.
 *
 * Mirrors the Python wrapper's `test_fuzzy_tag_groups_mapping.py`. Pins that a
 * `tag_groups` filter reaches the request body with the leaf's `resolve` field intact, so
 * fuzzy tag matching (#4026) is usable through the wrapper and not only over raw HTTP.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedRecall = sdk.recallMemories as jest.MockedFunction<typeof sdk.recallMemories>;
const mockedReflect = sdk.reflect as jest.MockedFunction<typeof sdk.reflect>;

const FUZZY_GROUP = {
  tags: ["typsecript"],
  match: "any_strict" as const,
  resolve: "fuzzy" as const,
};

describe("fuzzy tag_groups mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedRecall.mockReset();
    mockedReflect.mockReset();
    mockedRecall.mockResolvedValue({ data: { results: [] } } as any);
    mockedReflect.mockResolvedValue({ data: { text: "" } } as any);
  });

  test("recall forwards a fuzzy tag group with resolve intact", async () => {
    await client.recall("bank-1", "what language is the parser in", {
      tagGroups: [{ ...FUZZY_GROUP }],
    });

    expect(mockedRecall).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({ tag_groups: [FUZZY_GROUP] }),
      })
    );
  });

  test("reflect forwards a fuzzy tag group with resolve intact", async () => {
    await client.reflect("bank-1", "what language is the parser in", {
      tagGroups: [{ ...FUZZY_GROUP }],
    });

    expect(mockedReflect).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({ tag_groups: [FUZZY_GROUP] }),
      })
    );
  });
});
