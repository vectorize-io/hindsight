import type { NextRequest } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", () => ({
  dataplaneBankUrl: vi.fn(() => "http://localhost/v1/default/banks/test-bank/files/retain"),
  getDataplaneHeaders: vi.fn(() => ({})),
}));

import { POST } from "@/app/api/files/retain/route";

describe("POST /api/files/retain", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns a localized 400 when the multipart request payload is not valid JSON", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const formData = new FormData();
    formData.set("request", "{not-json");

    const response = await POST(
      new Request("http://localhost/api/files/retain", {
        method: "POST",
        body: formData,
      }) as unknown as NextRequest
    );

    await expect(response.json()).resolves.toEqual({ error: "Invalid request body" });
    expect(response.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
