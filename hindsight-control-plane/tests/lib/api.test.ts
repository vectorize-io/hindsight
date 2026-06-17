import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { toast } from "sonner";
import { ControlPlaneClient } from "@/lib/api";

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    warning: vi.fn(),
  },
}));

describe("ControlPlaneClient error handling", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let client: ControlPlaneClient;

  beforeEach(() => {
    client = new ControlPlaneClient();
    fetchSpy = vi.spyOn(globalThis, "fetch");
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: {
          href: "",
          pathname: "/en/dashboard",
          search: "",
        },
      },
    });
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    vi.mocked(toast.error).mockReset();
    vi.mocked(toast.warning).mockReset();
    delete (globalThis as { window?: unknown }).window;
  });

  it("shows client-error details for 4xx validation failures", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: "Failed to update bank config",
          details: "retain_structured_chunk_size must be a positive integer",
        }),
        { status: 400 }
      )
    );

    await expect(client.getBankConfig("bank-a")).rejects.toMatchObject({
      message: "retain_structured_chunk_size must be a positive integer",
      status: 400,
      details: "retain_structured_chunk_size must be a positive integer",
    });

    expect(toast.warning).toHaveBeenCalledWith(
      "Client Error",
      expect.objectContaining({
        description: "retain_structured_chunk_size must be a positive integer",
      })
    );
  });

  it("does not show upstream response details for 5xx failures", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: "DiskFullError on shared memory",
          details: "internal stack trace",
        }),
        { status: 500 }
      )
    );

    await expect(client.getBankConfig("bank-a")).rejects.toMatchObject({
      message: "HTTP 500",
      status: 500,
    });

    expect(toast.error).toHaveBeenCalledWith(
      "Server Error",
      expect.objectContaining({
        description: "HTTP 500",
      })
    );
  });
});
