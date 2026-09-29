import type { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", () => ({
  dataplaneBankUrl: (bankId: string, suffix: string) =>
    `http://dataplane/v1/default/banks/${encodeURIComponent(bankId)}${suffix}`,
  getDataplaneHeaders: () => ({}),
}));

import { GET } from "@/app/api/documents/chunks/route";

function makeRequest(url: string): NextRequest {
  // The route only reads `request.nextUrl.searchParams`.
  return { nextUrl: new URL(url) } as unknown as NextRequest;
}

describe("GET /api/documents/chunks", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })
    );
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // Regression for #4587: the id was interpolated raw, so `folder/example` became
  // two extra path segments on the dataplane URL and the Chunks tab 404'd — with no
  // proxy involved at all.
  it("encodes a slash-bearing document id into the dataplane URL", async () => {
    await GET(
      makeRequest("http://localhost/api/documents/chunks?bank_id=b1&document_id=folder%2Fexample")
    );

    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://dataplane/v1/default/banks/b1/documents/folder%2Fexample/chunks?limit=100&offset=0"
    );
  });

  it("rejects a request with no document_id", async () => {
    const response = await GET(makeRequest("http://localhost/api/documents/chunks?bank_id=b1"));

    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
