import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

import { POST } from "@/app/api/auth-profiles/supabase-org/auth/change-password/route";
import {
  clearSupabaseOrgSessionCookies,
  getValidSupabaseSession,
  signInWithPassword,
  signOutSupabaseSession,
  updateSupabaseUserPassword,
} from "@/lib/supabase-org/store";

vi.mock("@/lib/supabase-org/store", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/supabase-org/store")>();
  return {
    ...actual,
    clearSupabaseOrgSessionCookies: vi.fn(),
    getValidSupabaseSession: vi.fn(),
    signInWithPassword: vi.fn(),
    signOutSupabaseSession: vi.fn(),
    updateSupabaseUserPassword: vi.fn(),
  };
});

function request(body: unknown): NextRequest {
  return new NextRequest("http://control.local/api/auth/change-password", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

describe("supabase org change password route", () => {
  beforeEach(() => {
    vi.mocked(clearSupabaseOrgSessionCookies).mockReset();
    vi.mocked(getValidSupabaseSession).mockReset();
    vi.mocked(signInWithPassword).mockReset();
    vi.mocked(signOutSupabaseSession).mockReset();
    vi.mocked(updateSupabaseUserPassword).mockReset();
  });

  it("reauthenticates, updates the password, globally signs out, and clears cookies", async () => {
    vi.mocked(getValidSupabaseSession).mockResolvedValue({
      accessToken: "access-token",
      user: { id: "user_1", email: "user@example.com" },
    });
    vi.mocked(signInWithPassword).mockResolvedValue({
      access_token: "reauth-access-token",
      user: { id: "user_1", email: "user@example.com" },
    });

    const response = await POST(
      request({ current_password: "old-password", new_password: "new-password" })
    );

    expect(response.status).toBe(200);
    expect(signInWithPassword).toHaveBeenCalledWith("user@example.com", "old-password");
    expect(updateSupabaseUserPassword).toHaveBeenCalledWith("access-token", "new-password");
    expect(signOutSupabaseSession).toHaveBeenCalledWith("access-token", "global");
    expect(signOutSupabaseSession).toHaveBeenCalledWith("reauth-access-token", "local");
    expect(clearSupabaseOrgSessionCookies).toHaveBeenCalledOnce();
  });

  it("cleans up the reauthentication session when password update fails", async () => {
    vi.mocked(getValidSupabaseSession).mockResolvedValue({
      accessToken: "access-token",
      user: { id: "user_1", email: "user@example.com" },
    });
    vi.mocked(signInWithPassword).mockResolvedValue({
      access_token: "reauth-access-token",
      user: { id: "user_1", email: "user@example.com" },
    });
    vi.mocked(updateSupabaseUserPassword).mockRejectedValueOnce(new Error("password rejected"));

    const response = await POST(
      request({ current_password: "old-password", new_password: "new-password" })
    );

    expect(response.status).toBe(400);
    expect(signOutSupabaseSession).toHaveBeenCalledWith("reauth-access-token", "local");
    expect(clearSupabaseOrgSessionCookies).not.toHaveBeenCalled();
  });

  it("still clears local cookies when global sign out fails after password update", async () => {
    vi.mocked(getValidSupabaseSession).mockResolvedValue({
      accessToken: "access-token",
      user: { id: "user_1", email: "user@example.com" },
    });
    vi.mocked(signInWithPassword).mockResolvedValue({
      access_token: "reauth-access-token",
      user: { id: "user_1", email: "user@example.com" },
    });
    vi.mocked(signOutSupabaseSession).mockRejectedValueOnce(new Error("global signout failed"));

    const response = await POST(
      request({ current_password: "old-password", new_password: "new-password" })
    );

    expect(response.status).toBe(200);
    expect(signOutSupabaseSession).toHaveBeenCalledWith("reauth-access-token", "local");
    expect(clearSupabaseOrgSessionCookies).toHaveBeenCalledOnce();
  });

  it("requires both current and new passwords", async () => {
    const response = await POST(request({ current_password: "old-password" }));

    expect(response.status).toBe(400);
    expect(getValidSupabaseSession).not.toHaveBeenCalled();
  });
});
