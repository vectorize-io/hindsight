import { createClient } from "../generated/client";
import { getVersion } from "../generated/sdk.gen";

describe("generated fetch client", () => {
  test("does not leak the SDK client selector into native RequestInit", async () => {
    const NativeRequest = globalThis.Request;

    class StrictRequest extends NativeRequest {
      constructor(input: RequestInfo | URL, init?: RequestInit) {
        if (init && "client" in init) {
          throw new TypeError("SDK client selector leaked into RequestInit");
        }
        super(input, init);
      }
    }

    Object.defineProperty(globalThis, "Request", {
      configurable: true,
      value: StrictRequest,
      writable: true,
    });

    try {
      const fetchMock = jest.fn(async () =>
        new Response(JSON.stringify({ version: "test" }), {
          headers: { "Content-Type": "application/json" },
          status: 200,
        })
      );
      const client = createClient({
        baseUrl: "http://127.0.0.1:8888",
        fetch: fetchMock as typeof fetch,
      });

      await getVersion({ client });

      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(globalThis, "Request", {
        configurable: true,
        value: NativeRequest,
        writable: true,
      });
    }
  });
});
