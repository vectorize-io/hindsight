import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { getSessionPrefix } from "@/lib/auth/session";
import { bankAllowed } from "@/lib/auth/tokens";

export function forbiddenResponse(request: NextRequest): NextResponse {
  return NextResponse.json(
    localizeApiErrorPayload(request, {
      error: "Forbidden",
      errorKey: "api.errors.auth.forbidden",
    }),
    { status: 403 }
  );
}

/**
 * For routes that read the bank id from the request body (which middleware
 * cannot inspect without consuming it): returns a 403 response when the session
 * is scoped to a prefix that does not cover `bankId`, else null. Admin sessions
 * (empty prefix) and setups without an access key resolve to null and pass.
 */
export async function assertBankAllowed(
  request: NextRequest,
  bankId: string
): Promise<NextResponse | null> {
  const prefix = await getSessionPrefix(request);
  if (prefix === null) return null;
  if (bankAllowed(prefix, bankId)) return null;
  return forbiddenResponse(request);
}
