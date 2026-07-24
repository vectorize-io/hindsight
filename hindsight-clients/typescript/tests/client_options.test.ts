import { DEFAULT_USER_AGENT, HindsightClient } from "../src";

describe("HindsightClient options", () => {
  test("flattens custom headers into the generated client configuration", () => {
    const client = new HindsightClient({
      baseUrl: "http://localhost:8888",
      apiKey: "test-api-key",
      headers: {
        "X-Request-Source": "test-suite",
      },
    });
    const internalClient = client as unknown as {
      client: { getConfig: () => { headers: Headers } };
    };

    const headers = internalClient.client.getConfig().headers;

    expect(headers.get("X-Request-Source")).toBe("test-suite");
    expect(headers.get("Authorization")).toBe("Bearer test-api-key");
    expect(headers.get("User-Agent")).toBe(DEFAULT_USER_AGENT);
  });
});
