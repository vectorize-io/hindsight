import { afterEach, describe, expect, it, vi } from "vitest";

import { getDataplaneHeadersForRequest } from "@/lib/hindsight-client";

function requestWithCookie(cookie: string): Request {
  return new Request("http://localhost/api/test", {
    headers: { cookie },
  });
}

describe("getDataplaneHeadersForRequest", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("uses the fixed dataplane API key in access_key mode", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "access_key");
    vi.stubEnv("HINDSIGHT_CP_DATAPLANE_API_KEY", "fixed-key");

    const headers = getDataplaneHeadersForRequest(requestWithCookie(""));

    expect(headers).toMatchObject({ Authorization: "Bearer fixed-key" });
  });

  it("keeps existing no-auth dataplane behavior when no key is configured", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "disabled");
    vi.stubEnv("HINDSIGHT_CP_DATAPLANE_API_KEY", "");

    const headers = getDataplaneHeadersForRequest(requestWithCookie(""));

    expect(headers.Authorization).toBeUndefined();
  });

});
