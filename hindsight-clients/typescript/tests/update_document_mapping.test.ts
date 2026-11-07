/**
 * Unit tests for the updateDocument wrapper (#6339): the camelCase options must reach the
 * PATCH body under their API names, or a partial tag change silently does nothing.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedUpdate = sdk.updateDocument as jest.MockedFunction<typeof sdk.updateDocument>;

describe("updateDocument mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedUpdate.mockReset();
    mockedUpdate.mockResolvedValue({ data: { success: true } } as any);
  });

  test("forwards add/remove tags and observation scopes", async () => {
    await client.updateDocument("bank-1", "doc-1", {
      addTags: ["team-c"],
      removeTags: ["team-a"],
      observationScopes: "shared",
    });

    expect(mockedUpdate).toHaveBeenCalledWith(
      expect.objectContaining({
        path: { bank_id: "bank-1", document_id: "doc-1" },
        body: {
          tags: undefined,
          add_tags: ["team-c"],
          remove_tags: ["team-a"],
          observation_scopes: "shared",
        },
      })
    );
  });

  test("forwards a full tag replacement", async () => {
    await client.updateDocument("bank-1", "doc-1", { tags: [] });

    expect(mockedUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ body: expect.objectContaining({ tags: [] }) })
    );
  });
});
