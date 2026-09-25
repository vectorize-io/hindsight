import type { NextRequest } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { getDataplaneHeaders } = vi.hoisted(() => ({
  getDataplaneHeaders: vi.fn(() => ({ Authorization: "Bearer test" })),
}));

vi.mock("@/lib/hindsight-client", () => ({
  getDataplaneHeaders,
  dataplaneBankUrl: (bankId: string, suffix = "") =>
    `http://dataplane.test/v1/default/banks/${encodeURIComponent(bankId)}${suffix}`,
}));

import { GET } from "@/app/api/documents/[documentId]/chunks/route";

function makeRequest(url: string): NextRequest {
  return { nextUrl: new URL(url) } as unknown as NextRequest;
}

describe("GET /api/documents/[documentId]/chunks", () => {
  beforeEach(() => {
    getDataplaneHeaders.mockClear();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 }))
    );
  });

  it("re-encodes a slashed document id before forwarding to the dataplane", async () => {
    await GET(
      makeRequest("http://localhost/api/documents/x/chunks?bank_id=bank%2Fa&limit=10&offset=0"),
      {
        params: Promise.resolve({ documentId: "folder/example" }),
      }
    );

    const url = vi.mocked(fetch).mock.calls[0][0] as string;
    expect(url).toBe(
      "http://dataplane.test/v1/default/banks/bank%2Fa/documents/folder%2Fexample/chunks?limit=10&offset=0"
    );
    expect(url).not.toContain("/documents/folder/example/");
  });
});
