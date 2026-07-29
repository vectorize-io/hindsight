import type { NextRequest } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

type ListDocumentsArg = { query: Record<string, unknown> };

const { listDocuments } = vi.hoisted(() => ({
  listDocuments: vi.fn<(arg: ListDocumentsArg) => Promise<unknown>>(),
}));

vi.mock("@/lib/hindsight-client", () => ({
  sdk: { listDocuments },
  lowLevelClient: {},
}));

vi.mock("@/lib/sdk-response", () => ({
  respondWithSdk: vi.fn(() => new Response(null, { status: 200 })),
}));

import { GET } from "@/app/api/documents/route";

function makeRequest(url: string): NextRequest {
  // The route only reads `request.nextUrl.searchParams`.
  return { nextUrl: new URL(url) } as unknown as NextRequest;
}

describe("GET /api/documents", () => {
  beforeEach(() => {
    listDocuments.mockReset();
    listDocuments.mockResolvedValue({ data: { items: [], total: 0 }, error: undefined });
  });

  it("forwards the `q` search term to the dataplane (search by document ID)", async () => {
    await GET(makeRequest("http://localhost/api/documents?bank_id=b1&q=my-doc-id&limit=25&offset=0"));

    expect(listDocuments).toHaveBeenCalledTimes(1);
    expect(listDocuments.mock.calls[0][0].query).toMatchObject({ q: "my-doc-id" });
  });

  it("omits `q` when no search term is provided", async () => {
    await GET(makeRequest("http://localhost/api/documents?bank_id=b1&limit=25&offset=0"));

    expect(listDocuments.mock.calls[0][0].query.q).toBeUndefined();
  });

  it("forwards repeated `tags` params and `tags_match` to the dataplane", async () => {
    await GET(
      makeRequest("http://localhost/api/documents?bank_id=b1&tags=alpha&tags=beta&tags_match=all_strict")
    );

    expect(listDocuments.mock.calls[0][0].query).toMatchObject({
      tags: ["alpha", "beta"],
      tags_match: "all_strict",
    });
  });

  it("omits `tags` when none are provided", async () => {
    await GET(makeRequest("http://localhost/api/documents?bank_id=b1"));

    expect(listDocuments.mock.calls[0][0].query.tags).toBeUndefined();
  });

  it("drops `tags_match` when no tags are filtered, so the dataplane default stands", async () => {
    await GET(makeRequest("http://localhost/api/documents?bank_id=b1&tags_match=all_strict"));

    expect(listDocuments.mock.calls[0][0].query.tags_match).toBeUndefined();
  });

  it("rejects an unknown `tags_match` value rather than passing it through", async () => {
    await GET(makeRequest("http://localhost/api/documents?bank_id=b1&tags=alpha&tags_match=bogus"));

    expect(listDocuments.mock.calls[0][0].query).toMatchObject({ tags: ["alpha"] });
    expect(listDocuments.mock.calls[0][0].query.tags_match).toBeUndefined();
  });
});
