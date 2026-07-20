import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest, NextResponse } from "next/server";

import { supabaseOrgAuthProvider } from "@/lib/auth-profiles/supabase-org/provider";
import { SUPABASE_ORG_ACCESS_TOKEN_COOKIE } from "@/lib/auth-profiles/supabase-org/constants";
import {
  acceptInviteForUser,
  clearSupabaseOrgSessionCookies,
  ensureInitialOrganizationForSignup,
  listOrganizationsForUser,
  signOutSupabaseSession,
  signInWithPassword,
  signUpWithPassword,
} from "@/lib/supabase-org/store";

vi.mock("@/lib/supabase-org/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/supabase-org/store")>();
  return {
    ...actual,
    acceptInviteForUser: vi.fn(),
    clearSupabaseOrgSessionCookies: vi.fn(),
    ensureInitialOrganizationForSignup: vi.fn(),
    listOrganizationsForUser: vi.fn(),
    signOutSupabaseSession: vi.fn(),
    signInWithPassword: vi.fn(),
    signUpWithPassword: vi.fn(),
  };
});

const user = { id: "user_1", email: "user@example.com" };
const invitedOrg = { id: "org_invited", name: "Invited Org", role: "member" as const };

function loginRequest(body: unknown): NextRequest {
  return new NextRequest("http://control.local/api/auth/login", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

describe("supabase org auth provider", () => {
  beforeEach(() => {
    vi.mocked(signInWithPassword).mockReset();
    vi.mocked(signUpWithPassword).mockReset();
    vi.mocked(acceptInviteForUser).mockReset();
    vi.mocked(clearSupabaseOrgSessionCookies).mockReset();
    vi.mocked(ensureInitialOrganizationForSignup).mockReset();
    vi.mocked(listOrganizationsForUser).mockReset();
    vi.mocked(signOutSupabaseSession).mockReset();
  });

  it("does not create an extra initial organization when signup accepts an invite", async () => {
    vi.mocked(signUpWithPassword).mockResolvedValue({
      access_token: "access-token",
      refresh_token: "refresh-token",
      user,
    });
    vi.mocked(acceptInviteForUser).mockResolvedValue({ org_id: invitedOrg.id });
    vi.mocked(listOrganizationsForUser).mockResolvedValue([invitedOrg]);

    const response = await supabaseOrgAuthProvider.login(
      loginRequest({
        mode: "signup",
        email: user.email,
        password: "password",
        organization_name: "Extra Org",
        invite_token: "invite-token",
      })
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      success: true,
      selected_org_id: invitedOrg.id,
      organizations: [invitedOrg],
    });
    expect(acceptInviteForUser).toHaveBeenCalledWith(user, "invite-token");
    expect(listOrganizationsForUser).toHaveBeenCalledWith(user.id);
    expect(ensureInitialOrganizationForSignup).not.toHaveBeenCalled();
  });

  it("creates an initial organization for direct signup without an invite", async () => {
    vi.mocked(signUpWithPassword).mockResolvedValue({
      access_token: "access-token",
      user,
    });
    vi.mocked(ensureInitialOrganizationForSignup).mockResolvedValue({
      id: "org_direct",
      name: "Direct Org",
      role: "owner",
    });

    const response = await supabaseOrgAuthProvider.login(
      loginRequest({
        mode: "signup",
        email: user.email,
        password: "password",
        organization_name: "Direct Org",
      })
    );

    expect(response.status).toBe(200);
    expect(ensureInitialOrganizationForSignup).toHaveBeenCalledWith(user, "Direct Org");
    expect(listOrganizationsForUser).not.toHaveBeenCalled();
  });

  it("signs out the Supabase session before clearing local cookies", async () => {
    const request = new NextRequest("http://control.local/api/auth/logout", {
      headers: { cookie: `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=access-token` },
    });
    const response = NextResponse.json({ success: true });

    await supabaseOrgAuthProvider.logout(response, request);

    expect(signOutSupabaseSession).toHaveBeenCalledWith("access-token", "local");
    expect(clearSupabaseOrgSessionCookies).toHaveBeenCalledWith(response, request);
  });

  it("still clears local cookies when Supabase logout fails", async () => {
    vi.mocked(signOutSupabaseSession).mockRejectedValueOnce(new Error("logout failed"));
    const request = new NextRequest("http://control.local/api/auth/logout", {
      headers: { cookie: `${SUPABASE_ORG_ACCESS_TOKEN_COOKIE}=access-token` },
    });
    const response = NextResponse.json({ success: true });

    await expect(supabaseOrgAuthProvider.logout(response, request)).resolves.toBeUndefined();

    expect(clearSupabaseOrgSessionCookies).toHaveBeenCalledWith(response, request);
  });
});
