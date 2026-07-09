import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

import { POST } from "@/app/api/auth-profiles/supabase-org/auth/select-org/route";
import { SUPABASE_ORG_ACCESS_TOKEN_COOKIE } from "@/lib/auth-profiles/supabase-org/constants";
import { getAuthenticatedUserWithRefresh, listOrganizationsForUser } from "@/lib/supabase-org/store";

vi.mock("@/lib/supabase-org/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/supabase-org/store")>();
  return {
    ...actual,
    getAuthenticatedUserWithRefresh: vi.fn(),
    listOrganizationsForUser: vi.fn(),
  };
});

function request(body: unknown): NextRequest {
  return new NextRequest("http://control.local/api/auth/select-org", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

describe("supabase org select organization route", () => {
  beforeEach(() => {
    vi.mocked(getAuthenticatedUserWithRefresh).mockReset();
    vi.mocked(listOrganizationsForUser).mockReset();
  });

  it("preserves refreshed session cookies when membership validation fails", async () => {
    vi.mocked(getAuthenticatedUserWithRefresh).mockImplementation(async (_request, response) => {
      response.cookies.set(SUPABASE_ORG_ACCESS_TOKEN_COOKIE, "new-access-token");
      return { id: "user_1", email: "user@example.com" };
    });
    vi.mocked(listOrganizationsForUser).mockResolvedValue([
      { id: "org_1", name: "Org 1", role: "member" },
    ]);

    const response = await POST(request({ org_id: "org_2" }));

    expect(response.status).toBe(403);
    expect(response.cookies.get(SUPABASE_ORG_ACCESS_TOKEN_COOKIE)?.value).toBe("new-access-token");
  });
});
