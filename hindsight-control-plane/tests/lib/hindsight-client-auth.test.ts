import { afterEach, describe, expect, it, vi } from "vitest";

import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
} from "@/lib/auth/provider";
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

  it("uses the fixed dataplane API key outside supabase_org mode", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "access_key");
    vi.stubEnv("HINDSIGHT_CP_DATAPLANE_API_KEY", "fixed-key");

    const headers = getDataplaneHeadersForRequest(requestWithCookie(""));

    expect(headers).toMatchObject({ Authorization: "Bearer fixed-key" });
    expect(headers["X-Hindsight-Org-Id"]).toBeUndefined();
  });

  it("uses Supabase session cookies in supabase_org mode", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "supabase_org");
    vi.stubEnv("HINDSIGHT_CP_DATAPLANE_API_KEY", "fixed-key");

    const headers = getDataplaneHeadersForRequest(
      requestWithCookie(
        `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=jwt-token; ${SUPABASE_ORG_SELECTED_ORG_COOKIE}=org_123`
      )
    );

    expect(headers).toMatchObject({
      Authorization: "Bearer jwt-token",
      "X-Hindsight-Org-Id": "org_123",
    });
  });
});
