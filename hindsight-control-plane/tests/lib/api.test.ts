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

describe("ControlPlaneClient.deleteOperation", () => {
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

  it("issues DELETE to /api/banks/{bankId}/operations/{opId}", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          success: true,
          message: "Operation deleted",
          operation_id: "op-1",
        }),
        { status: 200 }
      )
    );

    await client.deleteOperation("bank-a", "op-1");

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/banks\/bank-a\/operations\/op-1$/),
      expect.objectContaining({ method: "DELETE" })
    );
  });
});

describe("ControlPlaneClient direct fetch error formatting", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let client: ControlPlaneClient;

  beforeEach(() => {
    client = new ControlPlaneClient();
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("preserves the transfer API's validation detail for the import view", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Invalid transfer archive: manifest.json is missing" }), {
        status: 400,
      })
    );

    const file = new File(["not a transfer archive"], "documents.zip", { type: "application/zip" });

    await expect(client.importDocuments("bank-a", file)).rejects.toMatchObject({
      message: "Invalid transfer archive: manifest.json is missing",
      status: 400,
    });
  });

  it("prefers an error message over a transfer API detail", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ error: "Transfer imports are disabled", detail: "Ignored detail" }), {
        status: 404,
      })
    );

    const file = new File(["transfer archive"], "documents.zip", { type: "application/zip" });

    await expect(client.importDocuments("bank-a", file)).rejects.toMatchObject({
      message: "Transfer imports are disabled",
      status: 404,
    });
  });

  it("formats structured validation details from direct upload requests", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: { violations: [{ message: "File type is not supported" }] } }), {
        status: 422,
      })
    );

    const file = new File(["unsupported"], "document.zip", { type: "application/zip" });

    await expect(
      client.uploadFiles({
        bank_id: "bank-a",
        files: [file],
      })
    ).rejects.toMatchObject({
      message: "File type is not supported",
      status: 422,
    });
  });

  it("formats structured validation details from binary download requests", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: { violations: [{ message: "Export is disabled" }] } }), {
        status: 404,
      })
    );

    await expect(client.exportDocuments("bank-a")).rejects.toMatchObject({
      message: "Export is disabled",
      status: 404,
    });
  });
});
