import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", async () => {
  const { createClient, createConfig, sdk } = await import("@vectorize-io/hindsight-client");
  return {
    DATAPLANE_URL: "http://dataplane",
    getDataplaneHeaders: (extra?: Record<string, string>) => ({
      Authorization: "Bearer secret",
      ...extra,
    }),
    sdk,
    lowLevelClient: createClient(createConfig({ baseUrl: "http://dataplane" })),
  };
});

import { GET as getDocuments } from "@/app/api/knowledge-banks/[bankId]/documents/route";
import { GET as getFile } from "@/app/api/knowledge-banks/[bankId]/documents/[documentId]/file/route";
import { GET as getMedia } from "@/app/api/knowledge-banks/[bankId]/documents/[documentId]/passages/[index]/media/route";
import { POST as ingestFiles } from "@/app/api/knowledge-banks/[bankId]/files/ingest/route";
import { GET as listOperations } from "@/app/api/knowledge-banks/[bankId]/operations/route";
import {
  DELETE as cancelOperation,
  GET as getOperation,
} from "@/app/api/knowledge-banks/[bankId]/operations/[operationId]/route";
import { POST as retryOperation } from "@/app/api/knowledge-banks/[bankId]/operations/[operationId]/retry/route";
import { DELETE as deleteOperation } from "@/app/api/knowledge-banks/[bankId]/operations/[operationId]/delete/route";

function req(url: string, init?: RequestInit): NextRequest {
  const request = new Request(url, init) as NextRequest & { nextUrl: URL };
  Object.defineProperty(request, "nextUrl", { value: new URL(url) });
  return request;
}

const ctx = <T extends Record<string, string>>(params: T) => ({ params: Promise.resolve(params) });
const media = ctx({ bankId: "docs", documentId: "d", index: "0" });
const file = ctx({ bankId: "docs", documentId: "d" });

afterEach(() => vi.restoreAllMocks());

describe("/api/knowledge-banks routes", () => {
  it("sends a JSON call through the SDK with the bank id encoded", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('{"documents":[],"total":0}', {
        headers: { "content-type": "application/json" },
      })
    );
    const res = await getDocuments(
      req("http://cp/api/knowledge-banks/my%2Fbank/documents?limit=25&status=ready"),
      ctx({ bankId: "my/bank" })
    );
    const sent = fetchSpy.mock.calls[0][0] as Request;
    expect(sent.url).toBe(
      "http://dataplane/v1/default/knowledge-banks/my%2Fbank/documents?limit=25&status=ready"
    );
    expect(await res.json()).toEqual({ documents: [], total: 0 });
  });

  it("proxies every operations call to the knowledge-bank endpoint, not the memory-bank one", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response("{}", { headers: { "content-type": "application/json" } })
    );
    const op = ctx({ bankId: "kb", operationId: "op1" });
    const base = "http://dataplane/v1/default/knowledge-banks/kb/operations";
    await listOperations(
      req("http://cp/x?limit=10&exclude_parents=true&status=failed"),
      ctx({ bankId: "kb" })
    );
    await getOperation(req("http://cp/x?include_payload=true"), op);
    await cancelOperation(req("http://cp/x", { method: "DELETE" }), op);
    await retryOperation(req("http://cp/x", { method: "POST" }), op);
    await deleteOperation(req("http://cp/x", { method: "DELETE" }), op);
    const sent = fetchSpy.mock.calls.map(([r]) => `${(r as Request).method} ${(r as Request).url}`);
    expect(sent).toEqual([
      `GET ${base}?status=failed&limit=10&exclude_parents=true`,
      `GET ${base}/op1?include_payload=true`,
      `DELETE ${base}/op1`,
      `POST ${base}/op1/retry`,
      `DELETE ${base}/op1/delete`,
    ]);
  });

  it("has a route behind every knowledge-bank call the UI makes", () => {
    // #6923: api.ts called /operations for a release with no route behind it, so the UI got a
    // 404 it swallowed. Every knowledgeApi(...) call must resolve to a route.ts: a quoted
    // segment is a folder of that name, a variable is a [param] folder.
    const root = join(__dirname, "../../../../src/app/api/knowledge-banks");
    const hasRoute = (dir: string, args: string[]): boolean => {
      if (args.length === 0) return existsSync(join(dir, "route.ts"));
      const [arg, ...rest] = args;
      const literal = /^"(.*)"$/.exec(arg)?.[1];
      const dirs = readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory());
      return dirs
        .filter((d) => (literal ? d.name === literal : d.name.startsWith("[")))
        .some((d) => hasRoute(join(dir, d.name), rest));
    };
    const api = readFileSync(join(__dirname, "../../../../src/lib/api.ts"), "utf8");
    const calls = [...api.matchAll(/knowledgeApi\(([^)]*)\)/g)].map((m) => m[1]);
    expect(calls.length).toBeGreaterThan(10);
    const missing = calls.filter(
      (c) => !hasRoute(root, c ? c.split(",").map((a) => a.trim()) : [])
    );
    expect(missing).toEqual([]);
  });

  it("passes an upstream error's reason and status through", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('{"detail":"knowledge bank not found"}', {
        status: 404,
        headers: { "content-type": "application/json" },
      })
    );
    const res = await getDocuments(req("http://cp/x"), ctx({ bankId: "nope" }));
    expect(res.status).toBe(404);
    expect((await res.json()).details).toBe("knowledge bank not found");
  });

  it("refuses dot segments that would escape the prefix", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const res = await getFile(req("http://cp/x"), ctx({ bankId: "..", documentId: "d" }));
    expect(res.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("streams a multipart upload with the browser's boundary", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response('{"document_ids":["d"]}', { status: 202 }));
    const form = new FormData();
    form.append("files", new File(["hello"], "a.md", { type: "text/markdown" }));
    const browserReq = new Request("http://cp/api/knowledge-banks/docs/files", {
      method: "POST",
      body: form,
    });
    const contentType = browserReq.headers.get("content-type")!;
    const res = await ingestFiles(
      req("http://cp/api/knowledge-banks/docs/files/ingest", {
        method: "POST",
        body: await browserReq.arrayBuffer(),
        headers: { "content-type": contentType },
      }),
      ctx({ bankId: "docs" })
    );
    expect(fetchSpy.mock.calls[0][0]).toBe(
      "http://dataplane/v1/default/knowledge-banks/docs/files/ingest"
    );
    const init = fetchSpy.mock.calls[0][1]!;
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe(contentType);
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer secret");
    expect(contentType).toMatch(/^multipart\/form-data; boundary=/);
    expect(res.status).toBe(202);
  });

  it("serves a byte range of passage media the dataplane sent whole", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), {
        headers: { "content-type": "video/mp4" },
      })
    );
    const res = await getMedia(
      req("http://cp/api/knowledge-banks/docs/documents/d/passages/0/media", {
        headers: { range: "bytes=2-4" },
      }),
      media
    );
    expect(res.status).toBe(206);
    expect(res.headers.get("content-range")).toBe("bytes 2-4/10");
    expect(res.headers.get("content-type")).toBe("video/mp4");
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
    expect([...new Uint8Array(await res.arrayBuffer())]).toEqual([2, 3, 4]);
  });

  it("answers 416 for a range past the end of the media", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(new Uint8Array(4)));
    const res = await getMedia(req("http://cp/x", { headers: { range: "bytes=10-" } }), media);
    expect(res.status).toBe(416);
    expect(res.headers.get("content-range")).toBe("bytes */4");
  });

  it("names an untyped original by its file name so the browser can preview it", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3]), {
        headers: {
          "content-type": "application/octet-stream",
          "content-disposition": "inline; filename*=UTF-8''demo%20clip.MP4",
        },
      })
    );
    const res = await getFile(req("http://cp/api/knowledge-banks/docs/documents/d/file"), file);
    expect(res.headers.get("content-type")).toBe("video/mp4");
    expect(res.headers.get("content-disposition")).toContain("inline");
  });

  it("sandboxes an uploaded original, but not a PDF, which the browser would then not show", async () => {
    for (const [type, sandboxed] of [
      ["text/html", true],
      ["application/pdf", false],
    ] as const) {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(new Uint8Array([1]), { headers: { "content-type": type } })
      );
      const res = await getFile(req("http://cp/api/knowledge-banks/docs/documents/d/file"), file);
      expect(res.headers.get("x-content-type-options")).toBe("nosniff");
      expect(res.headers.get("content-security-policy")).toBe(sandboxed ? "sandbox" : null);
    }
  });
});
