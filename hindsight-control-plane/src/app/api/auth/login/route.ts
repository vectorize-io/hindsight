import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

import {
  ACCESS_KEY_COOKIE,
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  sessionCookieOptions,
} from "@/lib/auth/session";
import { getControlPlaneAuthProvider } from "@/lib/auth/provider";
import {
  acceptInviteForUser,
  ensureInitialOrganizationForSignup,
  listOrganizationsForUser,
  setSupabaseOrgSessionCookies,
  signInWithPassword,
  signUpWithPassword,
} from "@/lib/supabase-org/store";

export async function POST(request: NextRequest) {
  const provider = getControlPlaneAuthProvider();
  if (provider === "supabase_org") {
    return loginWithSupabaseOrg(request);
  }

  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;

  // If no access key is configured, return 503
  if (!accessKey) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Access key not configured",
        errorKey: "api.errors.auth.accessKeyNotConfigured",
      }),
      { status: 503 }
    );
  }

  let body: { key?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid request body",
        errorKey: "api.errors.auth.invalidRequestBody",
      }),
      { status: 400 }
    );
  }

  const providedKey = body.key;

  // Constant-time comparison to prevent timing attacks
  const isValid = providedKey && constantTimeCompare(providedKey, accessKey);

  if (!isValid) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid access key",
        errorKey: "api.errors.auth.invalidAccessKey",
      }),
      { status: 401 }
    );
  }

  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: await createSessionToken(accessKey),
    ...sessionCookieOptions(request),
    maxAge: SESSION_MAX_AGE_SECONDS,
  });

  return response;
}

async function loginWithSupabaseOrg(request: NextRequest): Promise<NextResponse> {
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
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid request body",
        errorKey: "api.errors.auth.invalidRequestBody",
      }),
      { status: 400 }
    );
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
      mode === "signup"
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
}

/**
 * Constant-time string comparison to prevent timing attacks.
 */
function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}
