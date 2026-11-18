import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", () => ({
  DATAPLANE_URL: "http://dataplane",
  getDataplaneHeaders: () => ({}),
}));

import { GET } from "@/app/api/banks/[bankId]/attachments/[attachmentId]/route";

async function serve(contentType: string) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response("bytes", { headers: { "content-type": contentType } }))
  );
  return GET(new NextRequest("http://localhost/api/banks/b/attachments/0123456789ab"), {
    params: Promise.resolve({ bankId: "b", attachmentId: "0123456789ab" }),
  });
}

describe("GET /api/banks/[bankId]/attachments/[attachmentId]", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each(["image/png", "image/jpeg", "image/webp", "audio/mpeg", "video/mp4"])(
    "shows %s in the browser",
    async (type) => {
      const res = await serve(type);
      expect(res.headers.get("content-disposition")).toBe("inline");
      expect(res.headers.get("content-security-policy")).toContain("sandbox");
    }
  );

  it.each(["text/html", "image/svg+xml", "application/pdf", "application/xhtml+xml", "x-made/up"])(
    "sends %s as a download, sandboxed",
    async (type) => {
      const res = await serve(type);
      expect(res.headers.get("content-type")).toBe(type);
      expect(res.headers.get("content-disposition")).toBe("attachment");
      expect(res.headers.get("content-security-policy")).toContain("sandbox");
      expect(res.headers.get("x-content-type-options")).toBe("nosniff");
    }
  );

  it("reads the media type past its parameters and case", async () => {
    expect((await serve("Image/PNG; charset=binary")).headers.get("content-disposition")).toBe("inline");
    expect((await serve("text/html; charset=utf-8")).headers.get("content-disposition")).toBe("attachment");
  });
});
