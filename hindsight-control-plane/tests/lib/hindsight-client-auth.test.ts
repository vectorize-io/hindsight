import { afterEach, describe, expect, it, vi } from "vitest";

import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
} from "@/lib/auth-profiles/supabase-org/constants";
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

  it("forwards Supabase JWT and selected tenant in supabase_org mode", () => {
    vi.stubEnv("HINDSIGHT_CP_AUTH_PROVIDER", "supabase_org");
    vi.stubEnv("HINDSIGHT_CP_DATAPLANE_API_KEY", "");

    const headers = getDataplaneHeadersForRequest(
      requestWithCookie(
        `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=jwt-token; ${SUPABASE_ORG_SELECTED_ORG_COOKIE}=org_123`
      )
    );

    expect(headers).toMatchObject({
      Authorization: "Bearer jwt-token",
      "X-Hindsight-Tenant-Id": "org_123",
    });
  });
});
