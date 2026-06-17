import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
} from "@/lib/auth/provider";
import {
  acceptInvite,
  assertOrganizationRole,
  createApiKey,
  createInvite,
  createOrganization,
  getCurrentOrgContext,
  removeMember,
  updateOrganizationName,
  updateMemberRole,
} from "@/lib/supabase-org/store";
import type { CurrentOrgContext, SupabaseUser } from "@/lib/supabase-org/store";

const serviceEnv = {
  HINDSIGHT_CP_AUTH_PROVIDER: "supabase_org",
  HINDSIGHT_AUTH_SUPABASE_URL: "http://supabase.local",
  HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY: "service-key",
  HINDSIGHT_AUTH_PUBLIC_BASE_URL: "http://control.local",
};

function requestWithSession(token = "jwt-token", orgId = "org_1"): Request {
  return new Request("http://control.local/api/test", {
    headers: {
      cookie: `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=${token}; ${SUPABASE_ORG_SELECTED_ORG_COOKIE}=${orgId}`,
    },
  });
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function adminContext(role: "owner" | "admin" = "owner"): CurrentOrgContext {
  return {
    user: { id: "user_owner", email: "owner@example.com" },
    selectedOrgId: "org_1",
    membership: { org_id: "org_1", user_id: "user_owner", email: "owner@example.com", role },
  };
}

describe("supabase org store", () => {
  beforeEach(() => {
    for (const [key, value] of Object.entries(serviceEnv)) vi.stubEnv(key, value);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("resolves the current user and selected organization membership", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ id: "user_1", email: "member@example.com" }))
      .mockResolvedValueOnce(
        jsonResponse([{ org_id: "org_1", user_id: "user_1", email: "member@example.com", role: "member" }])
      );

    const context = await getCurrentOrgContext(requestWithSession());

    expect(context.user).toEqual({ id: "user_1", email: "member@example.com" });
    expect(context.selectedOrgId).toBe("org_1");
    expect(context.membership.role).toBe("member");
    expect(fetchMock.mock.calls[0][0]).toBe("http://supabase.local/auth/v1/user");
    expect(String(fetchMock.mock.calls[1][0])).toContain("/rest/v1/organization_members");
  });

  it("creates an organization and owner membership", async () => {
    const user: SupabaseUser = { id: "user_1", email: "owner@example.com" };
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "generated-org", name: "Acme", config: {} }]))
      .mockResolvedValueOnce(
        jsonResponse([{ org_id: "generated-org", user_id: "user_1", email: "owner@example.com", role: "owner" }])
      );

    const organization = await createOrganization(user, "  Acme  ");

    expect(organization).toEqual({ id: "generated-org", name: "Acme", config: {} });
    const membershipRequest = fetchMock.mock.calls[1][1] as RequestInit;
    expect(JSON.parse(String(membershipRequest.body))).toMatchObject({
      org_id: "generated-org",
      user_id: "user_1",
      role: "owner",
    });
  });

  it("allows only owners to rename the selected organization", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "org_1", name: "Renamed", config: {} }]));

    const organization = await updateOrganizationName(adminContext("owner"), "org_1", "  Renamed  ");

    expect(organization).toEqual({ id: "org_1", name: "Renamed", config: {} });
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({ name: "Renamed" });
    expect(String(fetchMock.mock.calls[0][0])).toContain("id=eq.org_1");
    await expect(updateOrganizationName(adminContext("admin"), "org_1", "Nope")).rejects.toThrow(
      /owner/i
    );
  });

  it("creates manual invites with normalized email and public URL", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "invite_1", expires_at: "2099-01-01T00:00:00.000Z" }]));

    const invite = await createInvite(adminContext("admin"), "  USER@Example.COM ", "member");

    expect(invite.id).toBe("invite_1");
    expect(invite.invite_url).toMatch(/^http:\/\/control\.local\/login\?invite=/);
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toMatchObject({
      org_id: "org_1",
      email: "user@example.com",
      role: "member",
    });
  });

  it("rejects expired invites before creating membership", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ id: "user_2", email: "user@example.com" }))
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "invite_1",
            org_id: "org_1",
            email: "user@example.com",
            role: "member",
            expires_at: "2000-01-01T00:00:00.000Z",
          },
        ])
      );

    await expect(acceptInvite(requestWithSession(), "raw-token")).rejects.toThrow(/expired/);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("only owners can change or remove members", async () => {
    await expect(updateMemberRole(adminContext("admin"), "user_2", "member")).rejects.toThrow(/owner/i);
    await expect(removeMember(adminContext("admin"), "user_2")).rejects.toThrow(/owner/i);
  });

  it("prevents owners from demoting or removing themselves", async () => {
    await expect(updateMemberRole(adminContext("owner"), "user_owner", "member")).rejects.toThrow(/own role/);
    await expect(removeMember(adminContext("owner"), "user_owner")).rejects.toThrow(/remove themselves/);
  });

  it("creates bank-scoped API keys with validated operations", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_1" }]))
      .mockResolvedValueOnce(jsonResponse([{ api_key_id: "api_key_1", bank_id: "bank_a" }]));

    const created = await createApiKey(adminContext("owner"), "Agent key", ["bank_a", "bank_a"], ["recall"]);

    expect(created).toMatchObject({ id: "api_key_1" });
    expect(created.key).toMatch(/^hs_/);
    const keyRequest = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(keyRequest.body))).toMatchObject({
      org_id: "org_1",
      name: "Agent key",
      role: "admin",
      allowed_operations: ["recall"],
    });
    const scopeRequest = fetchMock.mock.calls[1][1] as RequestInit;
    expect(JSON.parse(String(scopeRequest.body))).toEqual([{ api_key_id: "api_key_1", bank_id: "bank_a" }]);
  });

  it("validates roles and API key operations", async () => {
    expect(() => assertOrganizationRole("owner")).not.toThrow();
    expect(() => assertOrganizationRole("viewer")).toThrow(/Invalid organization role/);
    await expect(createApiKey(adminContext("owner"), "Bad key", null, ["delete"])).rejects.toThrow(
      /Invalid API key operation/
    );
  });

  it("blocks post-login organization creation when policy is direct signup only", async () => {
    vi.stubEnv("HINDSIGHT_AUTH_ORG_CREATION_POLICY", "direct_signup_only");
    const { POST } = await import("@/app/api/organizations/route");

    const response = await POST(
      new Request("http://control.local/api/organizations", {
        method: "POST",
        body: JSON.stringify({ name: "Extra org" }),
      })
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toMatchObject({
      error: "Organization creation is only allowed during direct signup",
    });
  });
});
