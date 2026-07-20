import { beforeEach, describe, expect, it, vi } from "vitest";

import { PATCH } from "@/app/api/auth-profiles/supabase-org/api-keys/[id]/route";
import { sdk } from "@/lib/hindsight-client";
import { getCurrentOrgContext, updateApiKeyPermissions } from "@/lib/supabase-org/store";

vi.mock("@/lib/hindsight-client", () => ({
  createDataplaneClientForRequest: vi.fn(() => ({})),
  sdk: { listBanks: vi.fn() },
}));

vi.mock("@/lib/supabase-org/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/supabase-org/store")>();
  return {
    ...actual,
    getCurrentOrgContext: vi.fn(),
    updateApiKeyPermissions: vi.fn(),
  };
});

describe("supabase org API key route", () => {
  beforeEach(() => {
    vi.mocked(getCurrentOrgContext).mockReset();
    vi.mocked(updateApiKeyPermissions).mockReset();
    vi.mocked(sdk.listBanks).mockReset();
  });

  it("switches an existing key to full access without resolving Bank scopes", async () => {
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

    const response = await PATCH(
      new Request("http://control.local/api/api-keys/key_1", {
        method: "PATCH",
        body: JSON.stringify({ permission_mode: "full_access" }),
      }),
      { params: Promise.resolve({ id: "key_1" }) }
    );

    expect(response.status).toBe(200);
    expect(updateApiKeyPermissions).toHaveBeenCalledWith(
      context,
      "key_1",
      "full_access",
      null
    );
    expect(sdk.listBanks).not.toHaveBeenCalled();
  });
});
