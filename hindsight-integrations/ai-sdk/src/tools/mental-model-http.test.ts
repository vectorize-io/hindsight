import { createServer } from "node:http";
import { once } from "node:events";
import { describe, it, expect, vi } from "vitest";
import { HindsightClient as ApiClient } from "../../../../hindsight-clients/typescript/src/index.js";
import { createHindsightTools } from "./index.js";

// The API defaults to metadata; content must be requested explicitly even
// when the caller already knows the model ID. Exercise the real SDK transport.
describe("mental model content through the AI SDK tool", () => {
  it("asks for content and returns stored synthesis to the agent", async () => {
    let detail: string | null = null;
    const server = createServer((request, response) => {
      detail = new URL(request.url!, "http://localhost").searchParams.get("detail");
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          id: "preferences",
          name: "Preferences",
          content: detail === "content" ? "Prefers tea" : null,
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
        client: {
          retain: vi.fn(),
          recall: vi.fn(),
          reflect: vi.fn(),
          getDocument: vi.fn(),
          getMentalModel: client.getMentalModel.bind(client),
        },
      });
      const result = await tools.getMentalModel.execute!(
        { mentalModelId: "preferences" },
        { toolCallId: "test", messages: [] }
      );
      expect(detail).toBe("content");
      expect(result).toMatchObject({ name: "Preferences", content: "Prefers tea" });
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
