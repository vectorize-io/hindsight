import { createServer } from "node:http";
import { once } from "node:events";
import { describe, it, expect } from "vitest";
import { createKnowledgeTools } from "../src/index.js";

// Exercise the generated transport too: HTTP failures must never look like a
// successful tool response, including the methods that bypass the wrapper.
describe("knowledge tool HTTP failures", () => {
  it.each([
    ["agent_knowledge_list_pages", {}],
    [
      "agent_knowledge_create_page",
      { page_id: "prefs", name: "Preferences", source_query: "What matters?" },
    ],
  ])("%s propagates the server error", async (name, params) => {
    const server = createServer((_request, response) => {
      response.writeHead(500, { "content-type": "application/json" });
      response.end(JSON.stringify({ detail: "backend unavailable" }));
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("No HTTP address");
    try {
      const tools = createKnowledgeTools({
        apiUrl: `http://127.0.0.1:${address.port}`,
        bankId: "test-bank",
      });
      const tool = tools.find((candidate) => candidate.name === name)!;
      await expect(tool.execute(params)).rejects.toThrow("backend unavailable");
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
