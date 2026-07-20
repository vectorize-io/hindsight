import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest, NextResponse } from "next/server";

import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_REFRESH_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
} from "@/lib/auth-profiles/supabase-org/constants";
import {
  acceptInvite,
  assertOrganizationRole,
  createApiKey,
  createInvite,
  createOrganization,
  getAuthenticatedUserWithRefresh,
  getCurrentOrgContext,
  listApiKeys,
  revealApiKey,
  removeMember,
  signOutSupabaseSession,
  updateSupabaseUserPassword,
  updateOrganizationName,
  updateMemberRole,
  updateApiKeyPermissions,
} from "@/lib/supabase-org/store";
import type { CurrentOrgContext, SupabaseUser } from "@/lib/supabase-org/store";
import {
  API_KEY_OPERATIONS,
  BANK_SCOPED_OPERATIONS,
  OPERATION_DEFINITIONS,
  OPERATION_GROUPS,
  UNSCOPED_DATAPLANE_OPERATIONS,
} from "@/lib/supabase-org/operations";

const serviceEnv = {
  HINDSIGHT_CP_AUTH_PROVIDER: "supabase_org",
  HINDSIGHT_AUTH_SUPABASE_URL: "http://supabase.local",
  HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY: "service-key",
  HINDSIGHT_AUTH_API_KEY_ENCRYPTION_KEY: "test-api-key-encryption-secret",
  HINDSIGHT_AUTH_PUBLIC_BASE_URL: "http://control.local",
};

function requestWithSession(token = "jwt-token", orgId = "org_1"): Request {
  return new Request("http://control.local/api/test", {
    headers: {
      cookie: `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=${token}; ${SUPABASE_ORG_SELECTED_ORG_COOKIE}=${orgId}`,
    },
  });
}

function nextRequestWithSession(
  accessToken = "jwt-token",
  refreshToken = "refresh-token",
  orgId = "org_1"
): NextRequest {
  return new NextRequest("http://control.local/api/test", {
    headers: {
      cookie: `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=${accessToken}; ${SUPABASE_ORG_REFRESH_TOKEN_COOKIE}=${refreshToken}; ${SUPABASE_ORG_SELECTED_ORG_COOKIE}=${orgId}`,
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

function memberContext(): CurrentOrgContext {
  return {
    user: { id: "user_member", email: "member@example.com" },
    selectedOrgId: "org_1",
    membership: { org_id: "org_1", user_id: "user_member", email: "member@example.com", role: "member" },
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

  it("keeps API key operations and UI groups aligned", () => {
    expect(API_KEY_OPERATIONS).toHaveLength(67);
    expect(new Set(API_KEY_OPERATIONS).size).toBe(API_KEY_OPERATIONS.length);
    expect(API_KEY_OPERATIONS).not.toContain("consolidate");
    expect(API_KEY_OPERATIONS).not.toContain("get_entity_state");
    expect(API_KEY_OPERATIONS).not.toContain("run_consolidation");
    expect(API_KEY_OPERATIONS).not.toContain("set_bank_mission");
    expect(UNSCOPED_DATAPLANE_OPERATIONS).toEqual(["create_bank"]);
    expect(BANK_SCOPED_OPERATIONS).not.toContain("create_bank");
    expect(OPERATION_DEFINITIONS.every((operation) => API_KEY_OPERATIONS.includes(operation.name))).toBe(
      true
    );
    expect(
      OPERATION_DEFINITIONS.every(
        (operation) => !("group" in operation) && !("section" in operation) && !("description" in operation)
      )
    ).toBe(true);

    const groupedOperations = OPERATION_GROUPS.flatMap((group) => group.operations);
    expect(new Set(groupedOperations)).toEqual(new Set(API_KEY_OPERATIONS));
    expect(OPERATION_GROUPS.map((group) => group.label)).toEqual([
      "Create bank",
      "Bank management",
      "Memory & mental models",
      "Operations & automation",
    ]);
    for (const group of OPERATION_GROUPS) {
      expect(group.labelKey).toBeTruthy();
      const sectionOperations = group.sections?.flatMap((section) => section.operations) ?? [];
      if (group.sections) expect(new Set(sectionOperations)).toEqual(new Set(group.operations));
    }
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
    expect(String(fetchMock.mock.calls[1][0])).toContain("removed_at=is.null");
  });

  it("refreshes an expired Supabase access token and updates session cookies", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ message: "expired" }, 401))
      .mockResolvedValueOnce(
        jsonResponse({
          access_token: "new-access-token",
          refresh_token: "new-refresh-token",
          expires_in: 123,
          user: { id: "user_1", email: "member@example.com" },
        })
      );
    const response = NextResponse.json({});

    const user = await getAuthenticatedUserWithRefresh(nextRequestWithSession(), response);

    expect(user).toEqual({ id: "user_1", email: "member@example.com" });
    expect(fetchMock.mock.calls[0][0]).toBe("http://supabase.local/auth/v1/user");
    expect(fetchMock.mock.calls[1][0]).toBe("http://supabase.local/auth/v1/token?grant_type=refresh_token");
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body))).toEqual({
      refresh_token: "refresh-token",
    });
    expect(response.cookies.get(SUPABASE_ORG_ACCESS_TOKEN_COOKIE)?.value).toBe("new-access-token");
    expect(response.cookies.get(SUPABASE_ORG_REFRESH_TOKEN_COOKIE)?.value).toBe("new-refresh-token");
  });

  it("clears Supabase session cookies when refresh fails", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ message: "expired" }, 401))
      .mockResolvedValueOnce(jsonResponse({ message: "invalid refresh" }, 401));
    const response = NextResponse.json({});

    await expect(getAuthenticatedUserWithRefresh(nextRequestWithSession(), response)).rejects.toThrow(
      /refresh failed/
    );

    expect(response.cookies.get(SUPABASE_ORG_ACCESS_TOKEN_COOKIE)?.value).toBe("");
    expect(response.cookies.get(SUPABASE_ORG_REFRESH_TOKEN_COOKIE)?.value).toBe("");
    expect(response.cookies.get(SUPABASE_ORG_SELECTED_ORG_COOKIE)?.value).toBe("");
  });

  it("signs out a Supabase session with local scope", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse({}));

    await signOutSupabaseSession("access-token", "local");

    expect(fetchMock.mock.calls[0][0]).toBe("http://supabase.local/auth/v1/logout?scope=local");
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((fetchMock.mock.calls[0][1] as RequestInit).headers).toMatchObject({
      Authorization: "Bearer access-token",
      apikey: "service-key",
    });
  });

  it("updates the Supabase user password with the current access token", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse({}));

    await updateSupabaseUserPassword("access-token", "new-password");

    expect(fetchMock.mock.calls[0][0]).toBe("http://supabase.local/auth/v1/user");
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("PUT");
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({
      password: "new-password",
    });
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

  it("allows only owners to create owner invites", async () => {
    await expect(createInvite(adminContext("admin"), "owner@example.com", "owner")).rejects.toThrow(
      /Only organization owners/
    );

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "invite_owner", expires_at: "2099-01-01T00:00:00.000Z" }]));

    await expect(createInvite(adminContext("owner"), "owner@example.com", "owner")).resolves.toMatchObject({
      id: "invite_owner",
    });
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toMatchObject({ role: "owner" });
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

  it("removes a member and revokes their API keys through one RPC", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(jsonResponse(null));

    await removeMember(adminContext("owner"), "user_2");

    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "/rest/v1/rpc/remove_organization_member"
    );
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({
      p_org_id: "org_1",
      p_user_id: "user_2",
      p_removed_by_user_id: "user_owner",
    });
  });

  it("creates operation-scoped API keys with validated operations", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_1" }]));

    const created = await createApiKey(adminContext("owner"), "Agent key", "scoped", [
      { operation: "recall", bank_scope_mode: "all", bank_scopes: [] },
    ]);

    expect(created).toMatchObject({ id: "api_key_1" });
    expect(created.key).toMatch(/^hs_/);
    const keyRequest = fetchMock.mock.calls[0][1] as RequestInit;
    const keyBody = JSON.parse(String(keyRequest.body));
    expect(keyBody).toMatchObject({
      p_org_id: "org_1",
      p_name: "Agent key",
      p_permission_mode: "scoped",
      p_allowed_operations: ["recall"],
      p_operation_scopes: [
        { operation: "recall", bank_scope_mode: "all", bank_scopes: [] },
      ],
    });
    expect(keyBody.p_encrypted_key).toMatch(/^v1\./);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/rest/v1/rpc/create_hindsight_api_key");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("creates selected-bank API key scopes", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_1" }]));

    const created = await createApiKey(
      adminContext("owner"),
      "Scoped key",
      "scoped",
      [
        {
          operation: "recall",
          bank_scope_mode: "selected",
          bank_scopes: [
            { bank_id: "bank_a", bank_internal_id: "internal_a" },
            { bank_id: "bank_a", bank_internal_id: "internal_a" },
          ],
        },
      ]
    );

    expect(created).toMatchObject({ id: "api_key_1" });
    const keyRequest = fetchMock.mock.calls[0][1] as RequestInit;
    const keyBody = JSON.parse(String(keyRequest.body));
    expect(keyBody).toMatchObject({
      p_org_id: "org_1",
      p_name: "Scoped key",
      p_permission_mode: "scoped",
      p_allowed_operations: ["recall"],
      p_operation_scopes: [
        {
          operation: "recall",
          bank_scope_mode: "selected",
          bank_scopes: [{ bank_id: "bank_a", bank_internal_id: "internal_a" }],
        },
      ],
    });
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("creates full-access API keys without persisted operation scopes", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_full" }]));

    const created = await createApiKey(
      adminContext("owner"),
      "Full key",
      "full_access",
      null
    );

    expect(created).toMatchObject({ id: "api_key_full" });
    const body = JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body));
    expect(body).toMatchObject({
      p_permission_mode: "full_access",
      p_allowed_operations: null,
      p_operation_scopes: [],
    });
  });

  it("allows members to create bank-scoped API keys", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_member" }]));

    const created = await createApiKey(memberContext(), "Read key", "scoped", null);

    expect(created).toMatchObject({ id: "api_key_member" });
    const keyRequest = fetchMock.mock.calls[0][1] as RequestInit;
    const keyBody = JSON.parse(String(keyRequest.body));
    expect(keyBody).not.toHaveProperty("p_role");
    expect(keyBody.p_allowed_operations).toContain("list_documents");
    expect(keyBody.p_allowed_operations).toContain("recall");
    expect(keyBody.p_allowed_operations).toContain("reflect");
    expect(keyBody.p_allowed_operations).toContain("retain");
    expect(keyBody.p_allowed_operations).toContain("create_mental_model");
    expect(keyBody.p_allowed_operations).toContain("update_mental_model");
    expect(keyBody.p_allowed_operations).toContain("update_document");
    expect(keyBody.p_allowed_operations).toContain("update_memory_unit");
    expect(keyBody.p_allowed_operations).toContain("delete_bank");
    expect(keyBody.p_allowed_operations).toContain("mental_model_refresh");
    expect(keyBody.p_allowed_operations).not.toContain("create_bank");
    expect(keyBody.p_allowed_operations).not.toContain("consolidate");
  });

  it("prevents member API keys from exceeding member operations", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse([{ id: "api_key_member" }])
    );
    await expect(
      createApiKey(memberContext(), "Bank scoped key", "scoped", [
        { operation: "delete_bank", bank_scope_mode: "all", bank_scopes: [] },
      ])
    ).resolves.toMatchObject({
      id: "api_key_member",
    });
    await expect(
      createApiKey(memberContext(), "Bad create key", "scoped", [
        { operation: "create_bank", bank_scope_mode: "all", bank_scopes: [] },
      ])
    ).rejects.toThrow(
      /exceeds creator permissions/
    );
  });

  it("updates API key permissions through one transactional RPC", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "api_key_1",
            org_id: "org_1",
            created_by_user_id: "user_owner",
            name: "Agent key",
            role: "owner",
            created_at: "2026-01-01T00:00:00Z",
          },
        ])
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));

    await updateApiKeyPermissions(adminContext("owner"), "api_key_1", "scoped", [
      {
        operation: "recall",
        bank_scope_mode: "selected",
        bank_scopes: [{ bank_id: "bank_a", bank_internal_id: "internal_a" }],
      },
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1][0])).toContain(
      "/rest/v1/rpc/replace_hindsight_api_key_permissions"
    );
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body))).toEqual({
      p_api_key_id: "api_key_1",
      p_org_id: "org_1",
      p_permission_mode: "scoped",
      p_allowed_operations: ["recall"],
      p_operation_scopes: [
        {
          operation: "recall",
          bank_scope_mode: "selected",
          bank_scopes: [{ bank_id: "bank_a", bank_internal_id: "internal_a" }],
        },
      ],
    });
  });

  it("switches an API key to full access through the transactional RPC", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "api_key_1",
            org_id: "org_1",
            created_by_user_id: "user_owner",
            name: "Agent key",
            role: "owner",
            permission_mode: "scoped",
            created_at: "2026-01-01T00:00:00Z",
          },
        ])
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));

    await updateApiKeyPermissions(
      adminContext("owner"),
      "api_key_1",
      "full_access",
      null
    );

    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body))).toEqual({
      p_api_key_id: "api_key_1",
      p_org_id: "org_1",
      p_permission_mode: "full_access",
      p_allowed_operations: null,
      p_operation_scopes: [],
    });
  });

  it("reveals stored API key secrets to admins and owning members only", async () => {
    const createdFetch = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_1" }]));
    const created = await createApiKey(adminContext("owner"), "Agent key", "scoped", [
      { operation: "recall", bank_scope_mode: "all", bank_scopes: [] },
    ]);
    const keyRequest = createdFetch.mock.calls[0][1] as RequestInit;
    const encryptedKey = JSON.parse(String(keyRequest.body)).p_encrypted_key;
    vi.restoreAllMocks();

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse([{ id: "api_key_1", org_id: "org_1", name: "Agent key", encrypted_key: encryptedKey }])
    );
    await expect(revealApiKey(adminContext("admin"), "api_key_1")).resolves.toEqual(created);
    vi.restoreAllMocks();

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse([
        {
          id: "api_key_1",
          org_id: "org_1",
          created_by_user_id: "user_other",
          name: "Agent key",
          encrypted_key: encryptedKey,
        },
      ])
    );
    await expect(revealApiKey(memberContext(), "api_key_1")).rejects.toThrow(/not owned/);
    vi.restoreAllMocks();

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      jsonResponse([
        {
          id: "api_key_1",
          org_id: "org_1",
          created_by_user_id: "user_member",
          name: "Agent key",
          encrypted_key: encryptedKey,
        },
      ])
    );
    await expect(revealApiKey(memberContext(), "api_key_1")).resolves.toEqual(created);
  });

  it("lists all API keys for admins and only owned keys for members", async () => {
    const keys = [
      {
        id: "key_a",
        org_id: "org_1",
        created_by_user_id: "user_owner",
        name: "A",
        created_at: "2026-01-01T00:00:00Z",
      },
      {
        id: "key_b",
        org_id: "org_1",
        created_by_user_id: "user_member",
        name: "B",
        created_at: "2026-01-01T00:00:00Z",
      },
    ];

    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(keys))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(
        jsonResponse([
          {
            api_key_id: "key_b",
            bank_id: "bank_owned",
            bank_internal_id: "internal_owned",
            created_at: "2026-01-02T00:00:00Z",
          },
        ])
      );
    await expect(listApiKeys(memberContext())).resolves.toMatchObject([
      {
        id: "key_b",
        operation_scopes: [],
        owned_banks: [{ bank_id: "bank_owned", bank_internal_id: "internal_owned" }],
        can_view_secret: true,
      },
    ]);
    vi.restoreAllMocks();

    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(keys))
      .mockResolvedValueOnce(
        jsonResponse([{ api_key_id: "key_a", operation: "recall", bank_scope_mode: "selected" }])
      )
      .mockResolvedValueOnce(
        jsonResponse([
          {
            api_key_id: "key_a",
            operation: "recall",
            bank_id: "bank_a",
            bank_internal_id: "internal_a",
          },
        ])
      )
      .mockResolvedValueOnce(jsonResponse([]));
    await expect(listApiKeys(adminContext("owner"))).resolves.toMatchObject([
      {
        id: "key_a",
        operation_scopes: [
          { operation: "recall", bank_scope_mode: "selected", scoped_bank_ids: ["bank_a"] },
        ],
        can_view_secret: true,
      },
      { id: "key_b", operation_scopes: [], can_view_secret: true },
    ]);
  });

  it("validates roles and API key operations", async () => {
    expect(() => assertOrganizationRole("owner")).not.toThrow();
    expect(() => assertOrganizationRole("viewer")).toThrow(/Invalid organization role/);
    await expect(
      createApiKey(adminContext("owner"), "Bad key", "scoped", [
        { operation: "delete" as never, bank_scope_mode: "all", bank_scopes: [] },
      ])
    ).rejects.toThrow(
      /Invalid API key operation/
    );
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse([{ id: "api_key_1" }]));
    await expect(
      createApiKey(adminContext("owner"), "Create key", "scoped", [
        { operation: "create_bank", bank_scope_mode: "all", bank_scopes: [] },
        { operation: "mental_model_refresh", bank_scope_mode: "all", bank_scopes: [] },
      ])
    ).resolves.toMatchObject({
      id: "api_key_1",
    });
    const keyRequest = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(keyRequest.body)).p_allowed_operations).toEqual([
      "create_bank",
      "mental_model_refresh",
    ]);
  });

  it("blocks post-login organization creation when policy is direct signup only", async () => {
    vi.stubEnv("HINDSIGHT_AUTH_ORG_CREATION_POLICY", "direct_signup_only");
    const { POST } = await import("@/app/api/auth-profiles/supabase-org/organizations/route");

    const response = await POST(
      new Request("http://control.local/api/auth-profiles/supabase-org/organizations", {
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
