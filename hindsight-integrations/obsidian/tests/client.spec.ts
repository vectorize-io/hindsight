import { beforeEach, describe, expect, it } from "vitest";
import { HindsightClient } from "../src/client";
// Import the mock by path so tsc type-checks against it; vitest aliases
// "obsidian" → this same module, so it's the singleton the client calls.
import { requestUrl } from "./__mocks__/obsidian";

const mock = requestUrl;

function ok(json: unknown = {}) {
  return { status: 200, text: JSON.stringify(json), json };
}

function lastCall() {
  const call = mock.mock.calls.at(-1);
  if (!call) throw new Error("requestUrl was not called");
  return call[0];
}

describe("HindsightClient", () => {
  beforeEach(() => {
    mock.mockReset();
    mock.mockResolvedValue(ok());
  });

  it("retain posts an upsert item with document_id and replace mode", async () => {
    const client = new HindsightClient("https://api.example.com/", "secret");
    await client.retain("bank x", "Folder/Note.md", "body text", { tags: ["t1"] });

    const params = lastCall();
    expect(params.method).toBe("POST");
    expect(params.url).toBe("https://api.example.com/v1/default/banks/bank%20x/memories");
    expect(params.headers?.Authorization).toBe("Bearer secret");
    const body = JSON.parse(params.body ?? "{}");
    expect(body.items[0]).toMatchObject({
      content: "body text",
      document_id: "Folder/Note.md",
      update_mode: "replace",
      tags: ["t1"],
    });
  });

  it("deleteDocument encodes segments but preserves path slashes", async () => {
    const client = new HindsightClient("https://api.example.com");
    await client.deleteDocument("b", "Folder/My Note.md");
    const params = lastCall();
    expect(params.method).toBe("DELETE");
    expect(params.url).toBe(
      "https://api.example.com/v1/default/banks/b/documents/Folder/My%20Note.md"
    );
  });

  it("reflect requests citations + trace when asked", async () => {
    mock.mockResolvedValue(ok({ text: "answer", based_on: { memories: [] } }));
    const client = new HindsightClient("https://api.example.com");
    const res = await client.reflect("b", "what?", { budget: "high", includeCitations: true });

    const body = JSON.parse(lastCall().body ?? "{}");
    expect(body).toMatchObject({ query: "what?", budget: "high" });
    expect(body.include).toEqual({ facts: {}, tool_calls: {} });
    expect(res.text).toBe("answer");
  });

  it("omits the Authorization header when no token is set", async () => {
    const client = new HindsightClient("https://api.example.com");
    await client.reflect("b", "q");
    expect(lastCall().headers?.Authorization).toBeUndefined();
  });

  it("throws a useful error on non-2xx", async () => {
    mock.mockResolvedValue({ status: 500, text: "boom", json: {} });
    const client = new HindsightClient("https://api.example.com");
    await expect(client.reflect("b", "q")).rejects.toThrow(/HTTP 500: boom/);
  });

  it("propagates a transport rejection (network/timeout)", async () => {
    mock.mockRejectedValue(new Error("net::ERR_CONNECTION_REFUSED"));
    const client = new HindsightClient("https://api.example.com");
    await expect(client.reflect("b", "q")).rejects.toThrow(/ERR_CONNECTION_REFUSED/);
  });

  it("reflect sends tag_groups when provided", async () => {
    const client = new HindsightClient("https://api.example.com");
    await client.reflect("b", "q", {
      tagGroups: [{ tags: ["vault:notes"], match: "all_strict" }],
      tags: ["ignored"],
    });
    const body = JSON.parse(lastCall().body ?? "{}");
    expect(body.tag_groups).toEqual([{ tags: ["vault:notes"], match: "all_strict" }]);
    expect(body.tags).toBeUndefined();
  });

  it("reflect falls back to tags when only tags are given", async () => {
    const client = new HindsightClient("https://api.example.com");
    await client.reflect("b", "q", { tags: ["work"] });
    const body = JSON.parse(lastCall().body ?? "{}");
    expect(body.tags).toEqual(["work"]);
    expect(body.tag_groups).toBeUndefined();
  });

  it("reflect sends neither tags nor tag_groups when both are empty", async () => {
    const client = new HindsightClient("https://api.example.com");
    await client.reflect("b", "q", { tags: [], tagGroups: [] });
    const body = JSON.parse(lastCall().body ?? "{}");
    expect(body.tags).toBeUndefined();
    expect(body.tag_groups).toBeUndefined();
  });

  it("retain omits the tags field when no tags are given", async () => {
    const client = new HindsightClient("https://api.example.com");
    await client.retain("b", "Note.md", "body");
    const body = JSON.parse(lastCall().body ?? "{}");
    expect(body.items[0].tags).toBeUndefined();
  });
});
