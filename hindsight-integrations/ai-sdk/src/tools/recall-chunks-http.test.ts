import { createServer } from "node:http";
import { once } from "node:events";
import { describe, it, expect, vi } from "vitest";
import { HindsightClient as ApiClient } from "../../../../hindsight-clients/typescript/src/index.js";
import { createHindsightTools } from "./index.js";

describe("source chunks through the AI SDK recall tool", () => {
  it("returns requested source text alongside its extracted fact", async () => {
    const chunks = {
      "chunk-1": {
        id: "chunk-1",
        text: "The fee is exactly 4.375%.",
        chunk_index: 0,
        truncated: false,
      },
    };
    let included = false;
    const server = createServer(async (request, response) => {
      let body = "";
      for await (const data of request) body += data.toString();
      included = JSON.parse(body).include?.chunks !== undefined;
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          results: [{ id: "fact-1", text: "Fee is about 4%", chunk_id: "chunk-1" }],
          ...(included ? { chunks } : {}),
        })
      );
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    try {
      const client = new ApiClient({ baseUrl: `http://127.0.0.1:${address.port}` });
      const tools = createHindsightTools({
        bankId: "test-bank",
        recall: { includeChunks: true },
        client: {
          retain: vi.fn(),
          reflect: vi.fn(),
          getDocument: vi.fn(),
          getMentalModel: vi.fn(),
          recall: client.recall.bind(client),
        },
      });
      const result = await tools.recall.execute!(
        { query: "exact fee" },
        { toolCallId: "test", messages: [] }
      );
      expect(included).toBe(true);
      expect(result).toMatchObject({ chunks });
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
