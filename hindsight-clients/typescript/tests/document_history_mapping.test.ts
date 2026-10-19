/**
 * Unit tests for the document history wrapper method.
 *
 * The history endpoint pages by operation, so a regression that dropped `limit`/`offset`
 * here would leave SDK callers on the first page of a long-lived document.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

jest.mock("../generated/sdk.gen");

const mockedHistory = sdk.getDocumentHistory as jest.MockedFunction<typeof sdk.getDocumentHistory>;

describe("document history mapping", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = new HindsightClient({ baseUrl: "http://localhost:8888" });
    mockedHistory.mockReset();
  });

  test("forwards the document, paging and signal", async () => {
    mockedHistory.mockResolvedValue({
      data: { document_id: "doc-1", total: 0, limit: 25, offset: 50, items: [] },
    } as any);
    const signal = new AbortController().signal;

    const history = await client.getDocumentHistory("bank-1", "doc-1", {
      limit: 25,
      offset: 50,
      signal,
    });

    expect(mockedHistory).toHaveBeenCalledWith(
      expect.objectContaining({
        path: { bank_id: "bank-1", document_id: "doc-1" },
        query: { limit: 25, offset: 50 },
        signal,
      })
    );
    expect(history?.document_id).toBe("doc-1");
  });

  test("returns null for a document outside the caller's scope", async () => {
    mockedHistory.mockResolvedValue({
      error: { detail: "Document not found" },
      response: { status: 404 },
    } as any);

    expect(await client.getDocumentHistory("bank-1", "doc-1")).toBeNull();
  });
});
