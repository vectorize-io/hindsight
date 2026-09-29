import { describe, expect, it } from "vitest";
import { chunkApi, documentApi } from "@/lib/bank-url";

// Regression for #4586 / #4587: document and chunk ids routinely contain `/`
// (S3 keys, file paths). If the id lands in a path segment, ingress proxies that
// normalize `%2F` back to `/` split it across segments and the route 404s.
describe("slash-bearing ids stay out of the URL path", () => {
  const slashId = "folder/sub/doc.pdf";

  it("keeps the document id in the query string, encoded", () => {
    const url = documentApi(slashId, "bank/one");

    expect(url).toBe("/api/documents?bank_id=bank%2Fone&document_id=folder%2Fsub%2Fdoc.pdf");
    expect(new URL(url, "http://x").pathname).toBe("/api/documents");
    expect(new URL(url, "http://x").searchParams.get("document_id")).toBe(slashId);
  });

  it("keeps sub-actions on a static path with the id in the query string", () => {
    for (const action of ["/chunks", "/reprocess"]) {
      const url = documentApi(slashId, "b1", action);

      expect(new URL(url, "http://x").pathname).toBe(`/api/documents${action}`);
      expect(new URL(url, "http://x").searchParams.get("document_id")).toBe(slashId);
    }
  });

  it("keeps the chunk id in the query string", () => {
    const url = chunkApi("folder/sub/doc.pdf#0");

    expect(new URL(url, "http://x").pathname).toBe("/api/chunks");
    expect(new URL(url, "http://x").searchParams.get("chunk_id")).toBe("folder/sub/doc.pdf#0");
  });
});
