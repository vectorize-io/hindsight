import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { respondWithSdk, type SdkResult } from "@/lib/sdk-response";

const HTTP_OK = 200;
const HTTP_CREATED = 201;
const HTTP_UPSTREAM_503 = 503;
const HTTP_UPSTREAM_429 = 429;
const HTTP_UPSTREAM_500 = 500;
const HTTP_DEFAULT_FAILURE = 502;

function makeResponse(status: number): Response {
  return new Response(null, { status });
}

function ok<T>(data: T, status = HTTP_OK): SdkResult<T> {
  return { data, error: undefined, response: makeResponse(status) };
}

function fail(error: unknown, status: number): SdkResult<never> {
  return { data: undefined, error, response: makeResponse(status) };
}

describe("respondWithSdk", () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  describe("success path", () => {
    it("returns 200 with the data when result is successful", async () => {
      const data = { foo: "bar", count: 42 };
      const response = respondWithSdk(ok(data), "Failed to fetch foo");
      expect(response.status).toBe(HTTP_OK);
      await expect(response.json()).resolves.toEqual(data);
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it("honors custom success status (e.g. 201 for creates)", async () => {
      const data = { id: "new-resource" };
      const response = respondWithSdk(ok(data), "Failed to create", HTTP_CREATED);
      expect(response.status).toBe(HTTP_CREATED);
      await expect(response.json()).resolves.toEqual(data);
    });

    it("returns 200 with array data", async () => {
      const data = [{ id: 1 }, { id: 2 }];
      const response = respondWithSdk(ok(data), "Failed to list");
      expect(response.status).toBe(HTTP_OK);
      await expect(response.json()).resolves.toEqual(data);
    });

    it("returns 200 with empty-object data (not treated as failure)", async () => {
      const data = {};
      const response = respondWithSdk(ok(data), "Failed to fetch");
      expect(response.status).toBe(HTTP_OK);
      await expect(response.json()).resolves.toEqual(data);
    });
  });

  describe("failure path", () => {
    it("does NOT throw TypeError when SDK returns undefined data (regression for upstream bug)", () => {
      // The original bug: NextResponse.json(undefined, ...) throws
      // `TypeError: Value is not JSON serializable`. This helper must short-
      // circuit before reaching that call site, so no throw should escape.
      expect(() =>
        respondWithSdk(fail({ detail: "boom" }, HTTP_UPSTREAM_500), "Failed to fetch stats")
      ).not.toThrow();
    });

    it("passes the upstream HTTP status through (5xx)", async () => {
      const response = respondWithSdk(
        fail({ detail: "internal server error" }, HTTP_UPSTREAM_500),
        "Failed to fetch stats"
      );
      expect(response.status).toBe(HTTP_UPSTREAM_500);
    });

    it("passes the upstream HTTP status through (503)", async () => {
      const response = respondWithSdk(
        fail({ detail: "unavailable" }, HTTP_UPSTREAM_503),
        "Failed to fetch banks"
      );
      expect(response.status).toBe(HTTP_UPSTREAM_503);
    });

    it("passes the upstream HTTP status through (429)", async () => {
      const response = respondWithSdk(
        fail({ detail: "rate limited" }, HTTP_UPSTREAM_429),
        "Failed to reflect"
      );
      expect(response.status).toBe(HTTP_UPSTREAM_429);
    });

    it("includes the upstream error detail in the response body", async () => {
      const upstreamError = { detail: "DiskFullError on shared memory" };
      const response = respondWithSdk(
        fail(upstreamError, HTTP_UPSTREAM_500),
        "Failed to fetch stats"
      );
      const body = await response.json();
      expect(body).toEqual({
        error: "Failed to fetch stats",
        upstream: {
          status: HTTP_UPSTREAM_500,
          detail: upstreamError,
        },
      });
    });

    it("logs the upstream status and error to console.error", () => {
      const upstreamError = { detail: "boom" };
      respondWithSdk(fail(upstreamError, HTTP_UPSTREAM_500), "Failed to fetch stats");
      expect(consoleErrorSpy).toHaveBeenCalledWith("Failed to fetch stats:", {
        upstreamStatus: HTTP_UPSTREAM_500,
        upstreamError,
      });
    });

    it("falls back to 502 when result has no Response (network-level failure)", async () => {
      // SDK call resolved but neither data nor a usable Response — treat as
      // "couldn't reach upstream" rather than masking as 500 (which implies
      // upstream answered).
      const result: SdkResult<unknown> = {
        data: undefined,
        error: new Error("ECONNREFUSED"),
      };
      const response = respondWithSdk(result, "Failed to reach API");
      expect(response.status).toBe(HTTP_DEFAULT_FAILURE);
      const body = await response.json();
      expect(body.upstream.status).toBe(HTTP_DEFAULT_FAILURE);
    });

    it("treats undefined data + undefined error as a failure (defensive)", async () => {
      // Should not happen with a well-behaved SDK, but if both are missing we
      // can't return undefined as JSON — treat as failure so the route doesn't
      // silently 200 with `null`.
      const result: SdkResult<unknown> = {
        data: undefined,
        error: undefined,
        response: makeResponse(HTTP_UPSTREAM_503),
      };
      const response = respondWithSdk(result, "Failed to fetch");
      expect(response.status).toBe(HTTP_UPSTREAM_503);
      const body = await response.json();
      expect(body.error).toBe("Failed to fetch");
      expect(body.upstream.detail).toBeNull();
    });
  });
});
