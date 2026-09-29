import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ControlPlaneClient } from "@/lib/api";

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    warning: vi.fn(),
  },
}));

/** A list response page: `total` counts every match, `items` only this page. */
function page(items: unknown[], total: number) {
  return new Response(JSON.stringify({ items, total, limit: 1000, offset: 0 }), { status: 200 });
}

function itemsOf(count: number, offset = 0) {
  return Array.from({ length: count }, (_, i) => ({ id: `item-${offset + i}` }));
}

describe("ControlPlaneClient paging helpers", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let client: ControlPlaneClient;

  beforeEach(() => {
    client = new ControlPlaneClient();
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("stops after one request when the first page holds every mental model", async () => {
    fetchSpy.mockResolvedValueOnce(page(itemsOf(3), 3));

    const all = await client.listAllMentalModels("bank-a");

    expect(all).toHaveLength(3);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("keeps requesting mental models until it has collected the reported total", async () => {
    // A bank past the endpoint's 1000 cap: the total is what tells the client to
    // ask again, which is exactly what a bare `items` response could not express.
    fetchSpy
      .mockResolvedValueOnce(page(itemsOf(1000), 1500))
      .mockResolvedValueOnce(page(itemsOf(500, 1000), 1500));

    const all = await client.listAllMentalModels("bank-a");

    expect(all).toHaveLength(1500);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    const secondUrl = String(fetchSpy.mock.calls[1][0]);
    expect(secondUrl).toContain("offset=1000");
  });

  it("forwards the filter options on every mental-model page", async () => {
    fetchSpy.mockResolvedValueOnce(page(itemsOf(1), 1));

    await client.listAllMentalModels("bank-a", { tags: ["work"], detail: "metadata" });

    const url = String(fetchSpy.mock.calls[0][0]);
    expect(url).toContain("tags=work");
    expect(url).toContain("detail=metadata");
    expect(url).toContain("limit=1000");
  });

  it("stops on an empty page even when the total disagrees", async () => {
    // Rows deleted mid-page would otherwise leave the loop asking forever.
    fetchSpy.mockResolvedValueOnce(page(itemsOf(2), 99)).mockResolvedValueOnce(page([], 99));

    const all = await client.listAllMentalModels("bank-a");

    expect(all).toHaveLength(2);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it("pages directives to the reported total", async () => {
    fetchSpy
      .mockResolvedValueOnce(page(itemsOf(1000), 1200))
      .mockResolvedValueOnce(page(itemsOf(200, 1000), 1200));

    const all = await client.listAllDirectives("bank-a");

    expect(all).toHaveLength(1200);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });
});
