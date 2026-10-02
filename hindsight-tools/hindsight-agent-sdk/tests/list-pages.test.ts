import { createServer } from "node:http";
import { once } from "node:events";
import { describe, it, expect } from "vitest";
import { createKnowledgeTools } from "../src/index.js";

describe("list all knowledge pages", () => {
  it("collects every metadata page through the HTTP transport", async () => {
    const items = Array.from({ length: 205 }, (_, index) => ({
      id: `page-${index}`,
      name: `Page ${index}`,
    }));
    const offsets: number[] = [];
    const details: Array<string | null> = [];
    const server = createServer((request, response) => {
      const url = new URL(request.url!, "http://localhost");
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const limit = Number(url.searchParams.get("limit") ?? 100);
      offsets.push(offset);
      details.push(url.searchParams.get("detail"));
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          items: items.slice(offset, offset + limit),
          total: items.length,
          limit,
          offset,
        })
      );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    try {
      const tool = createKnowledgeTools({
        apiUrl: `http://127.0.0.1:${address.port}`,
        bankId: "test-bank",
      })[0];
      const result = JSON.parse((await tool.execute({})).content[0].text);
      expect(result.items).toEqual(items);
      expect(offsets).toEqual([0, 100, 200]);
      expect(details).toEqual(["metadata", "metadata", "metadata"]);
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });

  it("rejects a later page failure instead of returning an incomplete list", async () => {
    const server = createServer((request, response) => {
      const offset = new URL(request.url!, "http://localhost").searchParams.get("offset");
      response.writeHead(offset === "100" ? 500 : 200, { "content-type": "application/json" });
      response.end(
        JSON.stringify(
          offset === "100"
            ? { detail: "second page unavailable" }
            : { items: Array.from({ length: 100 }, (_, index) => ({ id: `page-${index}` })) }
        )
      );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    try {
      const tool = createKnowledgeTools({
        apiUrl: `http://127.0.0.1:${address.port}`,
        bankId: "test-bank",
      })[0];
      await expect(tool.execute({})).rejects.toThrow("second page unavailable");
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
