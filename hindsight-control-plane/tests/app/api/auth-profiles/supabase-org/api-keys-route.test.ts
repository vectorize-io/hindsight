import { beforeEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "@/app/api/auth-profiles/supabase-org/api-keys/route";
import { sdk } from "@/lib/hindsight-client";
import {
  createApiKey,
  getCurrentOrgContext,
  listApiKeys,
} from "@/lib/supabase-org/store";

vi.mock("@/lib/hindsight-client", () => ({
  createDataplaneClientForRequest: vi.fn(() => ({})),
  sdk: { listBanks: vi.fn() },
}));

vi.mock("@/lib/supabase-org/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/supabase-org/store")>();
  return {
    ...actual,
    createApiKey: vi.fn(),
    getCurrentOrgContext: vi.fn(),
    listApiKeys: vi.fn(),
  };
});

describe("supabase org API key list route", () => {
  beforeEach(() => {
    vi.mocked(createApiKey).mockReset();
    vi.mocked(getCurrentOrgContext).mockReset();
    vi.mocked(listApiKeys).mockReset();
    vi.mocked(sdk.listBanks).mockReset();
  });

  it("creates a full-access key without resolving Bank scopes", async () => {
    const context = {
      user: { id: "user_owner", email: "owner@example.com" },
      selectedOrgId: "org_1",
      membership: {
        org_id: "org_1",
        user_id: "user_owner",
        email: "owner@example.com",
        role: "owner" as const,
      },
    };
    vi.mocked(getCurrentOrgContext).mockResolvedValue(context);
    vi.mocked(createApiKey).mockResolvedValue({ id: "key_full", key: "hs_secret" });

    const response = await POST(
      new Request("http://control.local/api/api-keys", {
        method: "POST",
        body: JSON.stringify({ name: "Full key", permission_mode: "full_access" }),
      })
    );

    expect(response.status).toBe(201);
    expect(createApiKey).toHaveBeenCalledWith(context, "Full key", "full_access", null);
    expect(sdk.listBanks).not.toHaveBeenCalled();
  });

  it("filters stale Bank references from the response", async () => {
    const context = {
      user: { id: "user_owner", email: "owner@example.com" },
      selectedOrgId: "org_1",
      membership: {
        org_id: "org_1",
        user_id: "user_owner",
        email: "owner@example.com",
      },
    };
    const apiKeys = [
      {
        id: "key_1",
        org_id: "org_1",
        name: "Agent key",
        created_at: "2026-01-01T00:00:00Z",
        operation_scopes: [
          {
            operation: "recall",
            bank_scope_mode: "selected" as const,
            scoped_bank_ids: ["active", "deleted"],
            scoped_bank_internal_ids: ["internal_active", "internal_deleted"],
          },
        ],
        owned_banks: [
          { bank_id: "active", bank_internal_id: "internal_active" },
          { bank_id: "deleted", bank_internal_id: "internal_deleted" },
        ],
      },
    ];
    vi.mocked(getCurrentOrgContext).mockResolvedValue(context);
    vi.mocked(listApiKeys).mockResolvedValue(apiKeys);
    vi.mocked(sdk.listBanks).mockResolvedValue({
      data: {
        banks: [{ bank_id: "active", internal_id: "internal_active", name: "Active bank" }],
      },
    } as never);
    const response = await GET(new Request("http://control.local/api/api-keys"));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.api_keys[0].operation_scopes[0]).toEqual({
      operation: "recall",
      bank_scope_mode: "selected",
      scoped_bank_ids: ["active"],
    });
    expect(body.api_keys[0].owned_banks).toEqual([
      {
        bank_id: "active",
        bank_internal_id: "internal_active",
        name: "Active bank",
      },
    ]);
  });
});
