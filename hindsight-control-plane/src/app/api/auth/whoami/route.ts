import { NextRequest, NextResponse } from "next/server";

import { getSessionPrefix } from "@/lib/auth/session";
import { labelForPrefix } from "@/lib/auth/tokens";

/**
 * Reports the current session's bank scope so the client can tailor the UI
 * (hide admin-only actions, foreign chrome). Requires a valid session: it lives
 * under `/api/` and is not in PUBLIC_PATTERNS, so middleware already gates it.
 */
export async function GET(request: NextRequest) {
  const prefix = (await getSessionPrefix(request)) ?? "";
  return NextResponse.json({
    isAdmin: prefix === "",
    prefix,
    label: labelForPrefix(prefix),
  });
}
