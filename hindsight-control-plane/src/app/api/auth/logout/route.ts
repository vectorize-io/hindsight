import { NextRequest, NextResponse } from "next/server";

import { getAuthProviderAdapter } from "@/lib/auth/provider";

export async function POST(request: NextRequest) {
  const response = NextResponse.json({ success: true });
  await getAuthProviderAdapter().logout(response, request);
  return response;
}
