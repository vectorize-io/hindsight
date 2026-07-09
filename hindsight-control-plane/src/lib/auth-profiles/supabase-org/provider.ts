import { NextResponse, type NextRequest } from "next/server";

import type { ControlPlaneAuthProviderAdapter } from "@/lib/auth/provider";
import {
  acceptInviteForUser,
  clearSupabaseOrgSessionCookies,
  ensureInitialOrganizationForSignup,
  listOrganizationsForUser,
  setSupabaseOrgSessionCookies,
  signOutSupabaseSession,
  signInWithPassword,
  signUpWithPassword,
} from "@/lib/supabase-org/store";
import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
} from "@/lib/auth-profiles/supabase-org/constants";

export const supabaseOrgAuthProvider: ControlPlaneAuthProviderAdapter = {
  id: "supabase_org",
  expectedDataplaneAuthProfile: "supabase_org",
  settingsPath: "/auth-profiles/supabase-org/settings",
  settingsLabel: "Organization settings",
  logoutEnabled: true,

  validateConfig() {
    if (process.env.HINDSIGHT_CP_ACCESS_KEY) {
      throw new Error(
        "HINDSIGHT_CP_AUTH_PROVIDER=supabase_org must not be combined with HINDSIGHT_CP_ACCESS_KEY"
      );
    }
  },

  async isAuthenticated(request) {
    return isSupabaseOrgSessionPresent(
      request.cookies.get(SUPABASE_ORG_ACCESS_TOKEN_COOKIE)?.value,
      request.cookies.get(SUPABASE_ORG_SELECTED_ORG_COOKIE)?.value
    );
  },

  getDataplaneHeaders(request, extra) {
    const headers: Record<string, string> = { ...extra };
    const accessToken = getCookie(request, SUPABASE_ORG_ACCESS_TOKEN_COOKIE);
    const selectedOrgId = getCookie(request, SUPABASE_ORG_SELECTED_ORG_COOKIE);
    if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
    if (selectedOrgId) headers["X-Hindsight-Tenant-Id"] = selectedOrgId;
    return headers;
  },

  getLoginConfig() {
    return {
      provider: "supabase_org",
      modes: [
        { id: "login", label: "Sign in" },
        { id: "signup", label: "Sign up" },
      ],
      defaultMode: "login",
      fields: [
        {
          name: "email",
          type: "email",
          placeholder: "Email",
          autocomplete: "email",
          required: true,
        },
        {
          name: "password",
          type: "password",
          placeholder: "Password",
          autocomplete: "current-password",
          required: true,
        },
        {
          name: "organization_name",
          type: "text",
          placeholder: "Organization name",
          autocomplete: "organization",
          modes: ["signup"],
          hiddenWhenInvite: true,
        },
      ],
      submitLabel: "Sign in",
      submitLabelsByMode: {
        login: "Sign in",
        signup: "Sign up",
      },
    };
  },

  async login(request) {
    let body: {
      mode?: "login" | "signup";
      email?: string;
      password?: string;
      organization_name?: string;
      selected_org_id?: string;
      invite_token?: string;
    };
    try {
      body = await request.json();
    } catch {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }

    if (!body.email || !body.password) {
      return NextResponse.json({ error: "email and password are required" }, { status: 400 });
    }

    try {
      const mode = body.mode || "login";
      const session =
        mode === "signup"
          ? await signUpWithPassword(body.email, body.password)
          : await signInWithPassword(body.email, body.password);

      if (!session) {
        return NextResponse.json(
          {
            success: false,
            pending_email_confirmation: true,
            error: "Email confirmation is required before login.",
          },
          { status: 202 }
        );
      }

      const acceptedInvite = body.invite_token
        ? await acceptInviteForUser(session.user, body.invite_token)
        : null;
      const organizations =
        mode === "signup" && !acceptedInvite
          ? [await ensureInitialOrganizationForSignup(session.user, body.organization_name)]
          : await listOrganizationsForUser(session.user.id);
      if (organizations.length === 0) {
        return NextResponse.json(
          {
            error:
              "No organization is available for this user. Accept an invite or sign up with organization creation enabled.",
          },
          { status: 403 }
        );
      }

      const selectedOrgId = acceptedInvite?.org_id || body.selected_org_id;
      const selectedOrg = selectedOrgId
        ? organizations.find((organization) => organization.id === selectedOrgId)
        : organizations[0];
      if (!selectedOrg) {
        return NextResponse.json(
          { error: "User is not a member of the selected organization" },
          { status: 403 }
        );
      }

      const response = NextResponse.json({
        success: true,
        user: session.user,
        organizations,
        selected_org_id: selectedOrg.id,
      });
      setSupabaseOrgSessionCookies(response, request, session, selectedOrg.id);
      return response;
    } catch (error) {
      return NextResponse.json(
        { error: error instanceof Error ? error.message : "Supabase login failed" },
        { status: 401 }
      );
    }
  },

  async logout(response, request) {
    try {
      await signOutSupabaseSession(
        request.cookies.get(SUPABASE_ORG_ACCESS_TOKEN_COOKIE)?.value,
        "local"
      );
    } catch {
      // Local logout must still complete even if the remote Supabase session is already gone.
    }
    clearSupabaseOrgSessionCookies(response, request);
  },
};

function isSupabaseOrgSessionPresent(
  accessToken: string | undefined,
  selectedOrgId: string | undefined
): boolean {
  return Boolean(accessToken && selectedOrgId);
}

function getCookie(request: NextRequest | Request, name: string): string | undefined {
  if ("cookies" in request) {
    return request.cookies.get(name)?.value;
  }

  const cookieHeader = request.headers.get("cookie");
  if (!cookieHeader) return undefined;
  for (const part of cookieHeader.split(";")) {
    const [key, ...valueParts] = part.trim().split("=");
    if (key === name) return decodeURIComponent(valueParts.join("="));
  }
  return undefined;
}
